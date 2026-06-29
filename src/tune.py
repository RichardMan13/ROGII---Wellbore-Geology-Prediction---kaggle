import argparse
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error

from src import config, data_loader, features


def objective(trial, model_name, mode="classifier"):
    if model_name == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1000, 5000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": config.SEED,
            "n_jobs": -1,
            "enable_categorical": True,
        }
        if mode == "classifier":
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "auc"
            model = XGBClassifier(**params)
        else:
            params["objective"] = "reg:squarederror"
            params["eval_metric"] = "rmse"
            model = XGBRegressor(**params)

    elif model_name == "logistic":
        if mode == "classifier":
            params = {
                "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "solver": trial.suggest_categorical(
                    "solver", ["lbfgs", "liblinear", "saga"]
                ),
                "max_iter": trial.suggest_int("max_iter", 500, 2000),
                "random_state": config.SEED,
                "n_jobs": -1,
            }
            if params["solver"] in ["liblinear", "saga"]:
                params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])
                if params["solver"] == "liblinear":
                    del params["n_jobs"]  # liblinear não suporta processamento paralelo
            else:
                params["penalty"] = "l2"
            model = LogisticRegression(**params)
        else:
            params = {
                "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
                "solver": trial.suggest_categorical(
                    "solver",
                    ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                ),
                "random_state": config.SEED,
            }
            model = Ridge(**params)

    elif model_name == "mlp":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f"n_units_l{i}", 32, 256, log=True))

        params = {
            "hidden_layer_sizes": tuple(layers),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-1, log=True
            ),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "solver": "adam",
            "max_iter": trial.suggest_int("max_iter", 200, 500),
            "early_stopping": True,
            "random_state": config.SEED,
        }
        if mode == "classifier":
            model = MLPClassifier(**params)
        else:
            model = MLPRegressor(**params)

    elif model_name == "knn":
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical(
                "metric", ["euclidean", "manhattan", "minkowski"]
            ),
            "n_jobs": -1,
        }
        if mode == "classifier":
            model = KNeighborsClassifier(**params)
        else:
            model = KNeighborsRegressor(**params)

    elif model_name in ["realmlp", "tabm", "tabpfn", "tabicl"]:
        raise ValueError(
            f"O modelo '{model_name}' é um Foundation Model ou utiliza Tuned Defaults "
            "profundos. Otimização Bayesiana manual não é recomendada ou necessária. "
            "Utilize os defaults fornecidos no config.py."
        )

    else:
        raise ValueError(f"Tuning para o modelo '{model_name}' não implementado.")

    # Carregar dados
    df_train = data_loader.load_train_data()
    # Aplicar engenharia de features
    df_train = features.engineer_features(df_train, is_train=True)

    features_cols = [
        c for c in df_train.columns if c not in [config.TARGET_COL, config.ID_COL]
    ]
    X = df_train[features_cols]
    y = df_train[config.TARGET_COL]

    if mode == "classifier":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=config.SEED)

    scores = []

    # Validacao Cruzada robusta para evitar overfitting no tuning
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(
            f"    [Trial {trial.number}] Treinando {model_name} - Fold {fold + 1}/5...",
            end="\r",
        )
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Prunar colinearidade apenas baseado no X_train
        pruner = features.CollinearityPruner(threshold=0.95)
        X_train = pruner.fit_transform(X_train, y_train)
        X_val = pruner.transform(X_val)

        if model_name in ["logistic", "mlp", "knn"]:
            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.pipeline import Pipeline

            num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = X_train.select_dtypes(
                include=["category", "object"]
            ).columns.tolist()

            num_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            cat_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ohe",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_transformer, num_cols),
                    ("cat", cat_transformer, cat_cols),
                ]
            )

            clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            clf.fit(X_train, y_train)

            if mode == "classifier":
                preds = clf.predict_proba(X_val)[:, 1]
                scores.append(roc_auc_score(y_val, preds))
            else:
                preds = clf.predict(X_val)
                scores.append(mean_squared_error(y_val, preds))
        else:
            if model_name == "xgboost":
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_train, y_train)

            if mode == "classifier":
                preds = model.predict_proba(X_val)[:, 1]
                scores.append(roc_auc_score(y_val, preds))
            else:
                preds = model.predict(X_val)
                scores.append(mean_squared_error(y_val, preds))

    return np.mean(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Otimização de Hiperparâmetros")
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=[
            "xgboost",
            "logistic",
            "mlp",
            "knn",
            "realmlp",
            "tabm",
            "tabpfn",
            "tabicl",
        ],
        help="Nome do modelo para otimizar",
    )
    parser.add_argument(
        "--mode", type=str, default="classifier", choices=["classifier", "regressor"]
    )
    args = parser.parse_args()

    print(
        f"Iniciando busca de hiperparametros para '{args.model}' com Optuna no modo '{args.mode}'..."
    )
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        study_name=f"tune_{args.model}_{args.mode}",
        storage="sqlite:///tuning_history.db",
        load_if_exists=True,
        direction="maximize" if args.mode == "classifier" else "minimize",
    )

    # Função auxiliar para limpar a linha antes do Optuna logar
    def trial_callback(study, trial):
        print(" " * 60, end="\r")  # Limpa o texto do Fold progress

    # Lambda wrapper para passar o model_name para a objective function
    study.optimize(
        lambda trial: objective(trial, args.model, args.mode),
        n_trials=10,
        callbacks=[trial_callback],
    )

    print("\n========================================")
    print("Busca Completa!")
    metric_name = "AUC" if args.mode == "classifier" else "MSE"
    print(f"Melhor Score {metric_name} OOF: {study.best_value:.5f}")
    print("Melhores Hiperparametros encontrados:")
    for k, v in study.best_params.items():
        print(f"  '{k}': {v}")
    print("========================================")
