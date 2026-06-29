import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss, accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src import config, data_loader, features, models


def get_score(y_true, y_pred, metric_name):
    if metric_name == "auc":
        if config.PROBLEM_TYPE == "multiclass":
            return roc_auc_score(y_true, y_pred, multi_class='ovr')
        return roc_auc_score(y_true, y_pred)
    elif metric_name == "logloss":
        return log_loss(y_true, y_pred)
    elif metric_name == "accuracy":
        if config.PROBLEM_TYPE == "multiclass":
            y_pred_classes = np.argmax(y_pred, axis=1)
            return accuracy_score(y_true, y_pred_classes)
        else:
            return accuracy_score(y_true, (y_pred > 0.5).astype(int))
    elif metric_name == "rmse":
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric_name == "mae":
        return mean_absolute_error(y_true, y_pred)
    else:
        raise ValueError(f"Métrica {metric_name} não suportada ainda. Adicione-a em get_score().")


def train_pipeline(model_name="lightgbm", mode="classifier"):
    print(f"Iniciando Pipeline de Treinamento: {model_name} ({mode} - {config.PROBLEM_TYPE})")

    # Garantir que a pasta de artefatos exista para salvar modelos e logs
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carregar dados de treino
    try:
        df_train = data_loader.load_train_data()
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        print(
            "Crie um arquivo train.csv dummy em data/raw/ para testar o pipeline executando o script."
        )
        return

    # 2. Separar Features e Target
    features_cols = [
        c for c in df_train.columns if c not in [config.TARGET_COL, config.ID_COL]
    ]
    X = df_train[features_cols]
    y = df_train[config.TARGET_COL]

    # 3. Configurar Arrays OOF e Splits
    if config.PROBLEM_TYPE == "multiclass":
        n_classes = y.nunique()
        oof_predictions = np.zeros((len(df_train), n_classes))
    else:
        oof_predictions = np.zeros(len(df_train))
        
    cv_splits = data_loader.get_cv_splits(df_train, mode=mode)

    scores = []

    # 4. Loop de Treino por Fold
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n--- Treinando Fold {fold + 1}/{config.N_SPLITS} ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Obter modelo
        model = models.get_model(model_name, mode=mode)

        # Instanciar a engenharia de features e o pruner
        fe = features.FeatureEngineer()
        pruner = features.CollinearityPruner(threshold=0.95)

        # Lógica de Preprocessamento para Modelos Lineares, Redes Neurais e KNN
        if model_name in ["logistic", "mlp", "knn"]:
            # Identificar colunas numericas e categoricas dinamicamente
            num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = X_train.select_dtypes(
                include=["category", "object"]
            ).columns.tolist()

            # Definir transformadores robustos
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

            # Unificar pre-processamento e classificador no mesmo pipeline, blindado de Leakage!
            clf = Pipeline(
                steps=[
                    ("fe", fe),
                    ("pruner", pruner),
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            # Treinar pipeline completo
            clf.fit(X_train, y_train)

            # Predições de validação usando o pipeline completo
            if mode == "classifier":
                if config.PROBLEM_TYPE == "multiclass":
                    val_preds = clf.predict_proba(X_val)
                else:
                    val_preds = clf.predict_proba(X_val)[:, 1]
            else:
                val_preds = clf.predict(X_val)

            fitted_obj = clf

        elif model_name in ["lightgbm", "xgboost", "catboost"]:
            # Aplicar fe manualmente para obter os arrays processados
            X_train_fe = fe.fit_transform(X_train, y_train)
            X_val_fe = fe.transform(X_val)
            
            # Para modelos de árvore, aplicamos o pruner para preservar o eval_set
            X_train_pruned = pruner.fit_transform(X_train_fe, y_train)
            X_val_pruned = pruner.transform(X_val_fe)

            # Lógica especial de Early Stopping ou fit dependendo do framework
            if model_name in ["lightgbm", "xgboost"]:
                model.fit(
                    X_train_pruned,
                    y_train,
                    eval_set=[(X_val_pruned, y_val)],
                    callbacks=[],  # Adicione callbacks de early stopping se desejado
                )
            else:
                model.fit(X_train_pruned, y_train, eval_set=[(X_val_pruned, y_val)])

            # Predições de validação
            if mode == "classifier":
                if config.PROBLEM_TYPE == "multiclass":
                    val_preds = model.predict_proba(X_val_pruned)
                else:
                    val_preds = model.predict_proba(X_val_pruned)[:, 1]
            else:
                val_preds = model.predict(X_val_pruned)

            # Re-empacotar em um Pipeline padronizado incluindo a FeatureEngineer
            fitted_obj = Pipeline(steps=[("fe", fe), ("pruner", pruner), ("model", model)])

        elif model_name in ["realmlp", "tabm", "tabpfn", "tabicl"]:
            # Deep Tabular / Foundation Models
            X_train_fe = fe.fit_transform(X_train, y_train)
            X_val_fe = fe.transform(X_val)
            
            X_train_pruned = pruner.fit_transform(X_train_fe, y_train)
            X_val_pruned = pruner.transform(X_val_fe)

            model.fit(X_train_pruned, y_train)

            # Predições de validação
            if mode == "classifier":
                if config.PROBLEM_TYPE == "multiclass":
                    val_preds = model.predict_proba(X_val_pruned)
                else:
                    val_preds = model.predict_proba(X_val_pruned)[:, 1]
            else:
                val_preds = model.predict(X_val_pruned)

            # Re-empacotar em um Pipeline padronizado
            fitted_obj = Pipeline(steps=[("fe", fe), ("pruner", pruner), ("model", model)])

        else:
            raise ValueError(f"Modelo desconhecido no loop de treino: {model_name}")

        # Avaliar score do fold dinamicamente
        fold_score = get_score(y_val, val_preds, config.EVAL_METRIC)
        print(f"Fold {fold + 1} {config.EVAL_METRIC.upper()}: {fold_score:.5f}")

        # Armazenar OOF e score
        oof_predictions[val_idx] = val_preds
        scores.append(fold_score)

        # Salvar o objeto treinado (pipeline final com FeatureEngineer injetada)
        model_path = config.ARTIFACTS_DIR / f"{model_name}_fold_{fold}.pkl"
        joblib.dump(fitted_obj, model_path)
        print(f"Saved model to {model_path}")

    # 5. Avaliar Consistência OOF consolidada
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    print("\n========================================")
    overall_score = get_score(y, oof_predictions, config.EVAL_METRIC)
    print(f"Consolidado OOF {config.EVAL_METRIC.upper()}: {overall_score:.5f}")
    print(f"CV {config.EVAL_METRIC.upper()} Medio: {mean_score:.5f} +/- {std_score:.5f}")
    print("========================================")

    # Salvar predições Out-of-Fold para blendings futuros
    if config.PROBLEM_TYPE == "multiclass":
        oof_df = pd.DataFrame(oof_predictions, columns=[f"pred_class_{i}" for i in range(oof_predictions.shape[1])])
        oof_df[config.ID_COL] = df_train[config.ID_COL].values
        oof_df[config.TARGET_COL] = y.values
    else:
        oof_df = pd.DataFrame(
            {
                config.ID_COL: df_train[config.ID_COL],
                "oof_pred": oof_predictions,
                config.TARGET_COL: y,
            }
        )
    oof_df.to_csv(config.ARTIFACTS_DIR / f"{model_name}_oof.csv", index=False)
    print(f"Saved OOF predictions to {config.ARTIFACTS_DIR / f'{model_name}_oof.csv'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=[
            "lightgbm",
            "xgboost",
            "catboost",
            "logistic",
            "mlp",
            "knn",
            "realmlp",
            "tabm",
            "tabpfn",
            "tabicl",
        ],
    )
    # mode pode ser inferido do config, mas mantemos para logica CLI retro-compativel
    parser.add_argument(
        "--mode", type=str, default="classifier", choices=["classifier", "regressor"]
    )
    args = parser.parse_args()

    train_pipeline(model_name=args.model, mode=args.mode)
