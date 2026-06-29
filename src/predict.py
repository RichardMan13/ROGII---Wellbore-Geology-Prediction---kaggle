import joblib
import numpy as np
import pandas as pd

from src import config, data_loader


def predict_pipeline(model_name="lightgbm", mode="classifier"):
    print(f"Iniciando Pipeline de Inferência: {model_name} ({mode} - {config.PROBLEM_TYPE})")

    # 1. Carregar dados de teste
    try:
        df_test = data_loader.load_test_data()
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        print("Crie um arquivo test.csv dummy em data/raw/ para testar a inferência.")
        return

    # 2. Separar as colunas usadas no treino
    features_cols = [
        c for c in df_test.columns if c not in [config.TARGET_COL, config.ID_COL]
    ]
    X_test = df_test[features_cols]

    # 3. Preparar array de predições (tratar multiclasse)
    if config.PROBLEM_TYPE == "multiclass":
        # Descobrir o número de classes lendo o primeiro modelo
        model_path = config.ARTIFACTS_DIR / f"{model_name}_fold_0.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo {model_path} não encontrado.")
        first_model = joblib.load(model_path)
        # Hack para pegar o número de classes do modelo final no pipeline
        n_classes = len(first_model.classes_) if hasattr(first_model, "classes_") else first_model.steps[-1][1].classes_.shape[0]
        test_preds = np.zeros((len(df_test), n_classes))
    else:
        test_preds = np.zeros(len(df_test))

    # 4. Predição com a média dos modelos de cada fold (Ensemble robusto)
    for fold in range(config.N_SPLITS):
        model_path = config.ARTIFACTS_DIR / f"{model_name}_fold_{fold}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo para o fold {fold} não encontrado em {model_path}. Treine os modelos primeiro!"
            )

        model = joblib.load(model_path)

        if mode == "classifier":
            if config.PROBLEM_TYPE == "multiclass":
                fold_preds = model.predict_proba(X_test)
            else:
                fold_preds = model.predict_proba(X_test)[:, 1]
        else:
            fold_preds = model.predict(X_test)

        test_preds += fold_preds / config.N_SPLITS
        print(f"Predicoes obtidas com fold {fold}")

    # 5. Criar arquivo de submissão formatado
    if config.PROBLEM_TYPE == "multiclass":
        submission = pd.DataFrame(test_preds, columns=[f"pred_class_{i}" for i in range(test_preds.shape[1])])
        submission.insert(0, config.ID_COL, df_test[config.ID_COL].values)
    else:
        submission = pd.DataFrame(
            {config.ID_COL: df_test[config.ID_COL], config.TARGET_COL: test_preds}
        )

    # 6. Submission Sanity Check (Skills)
    try:
        sample_path = config.DATA_DIR / "raw" / "sample_submission.csv"
        if sample_path.exists():
            sample_sub = pd.read_csv(sample_path)
            assert len(submission) == len(sample_sub), (
                f"Erro de Sanidade: Mismatch de linhas! Test={len(submission)} != Sample={len(sample_sub)}"
            )

        assert not submission.isnull().values.any(), (
            "Erro de Sanidade: NaNs encontrados na submissao!"
        )
        print("[Sanity Check] OK! Submissao pronta para envio e livre de NaNs.")
    except Exception as e:
        print(
            f"\n[FALHA DE SANIDADE]: {e}\nA submissao foi bloqueada para previnir erros no Kaggle."
        )
        return

    config.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = config.SUBMISSIONS_DIR / f"submission_{model_name}.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Submissao criada com sucesso em: {sub_path}!")


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
    parser.add_argument(
        "--mode", type=str, default="classifier", choices=["classifier", "regressor"]
    )
    args = parser.parse_args()

    predict_pipeline(model_name=args.model, mode=args.mode)
