import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.optimize import minimize

from src import config


def blend_pipeline(mode="classifier"):
    print(
        f"Iniciando Pipeline de Blending Dinamico ({'Robust Rank Ensemble' if mode == 'classifier' else 'Weighted Average'})..."
    )

    # 1. Detectar OOFs disponiveis na pasta de artefatos
    supported_models = list(config.MODEL_PARAMS.keys())
    oof_dfs = {}

    for model in supported_models:
        path = config.ARTIFACTS_DIR / f"{model}_oof.csv"
        if path.exists():
            oof_dfs[model] = pd.read_csv(path)

    if len(oof_dfs) < 2:
        print(
            "\n[ERRO] Sao necessarios pelo menos 2 modelos treinados com arquivos OOF para realizar o blending!"
        )
        print(f"Modelos suportados: {supported_models}")
        print(f"Modelos encontrados atualmente: {list(oof_dfs.keys())}")
        return

    models_list = list(oof_dfs.keys())
    print(f"Modelos detectados para blending: {models_list}")

    y_true = oof_dfs[models_list[0]][config.TARGET_COL].values

    # Conversao das probabilidades brutas para Ranks Percentuais para classificacao
    ranks = []
    for model in models_list:
        pred = oof_dfs[model]["oof_pred"].values
        if mode == "classifier":
            rank = pd.Series(pred).rank(pct=True).values
            ranks.append(rank)
            print(f"  * {model} OOF AUC: {roc_auc_score(y_true, pred):.5f}")
        else:
            ranks.append(pred)
            print(
                f"  * {model} OOF RMSE: {np.sqrt(mean_squared_error(y_true, pred)):.5f}"
            )

    # Otimizar os pesos usando scipy minimize
    # O último peso é determinado deterministicamente para garantir soma 1.0.
    M = len(models_list)

    def objective(x):
        w = list(x)
        w_last = 1.0 - sum(w)
        w.append(w_last)

        # Funcao de barreira rigida para garantir limites [0.0, 1.0]
        if any(val < 0.0 or val > 1.0 for val in w):
            return 99.0  # Penalidade extrema se violar os limites impostos

        blend_preds = np.zeros_like(ranks[0])
        for i in range(M):
            blend_preds += w[i] * ranks[i]

        if mode == "classifier":
            return -roc_auc_score(
                y_true, blend_preds
            )  # Minimizar negativo = Maximizar AUC
        else:
            return mean_squared_error(y_true, blend_preds)

    # Chute inicial uniforme
    init_weights = [1.0 / M] * (M - 1)
    bounds = [(0.0, 1.0)] * (M - 1)

    # Uso de algoritmo livre de derivadas (Gradient-Free) Powell
    # Essencial para otimizar superficies nao-suaves e degraus como a metrica do AUC.
    res = minimize(objective, init_weights, method="Powell", bounds=bounds)

    w_opt = list(res.x)
    w_last_opt = 1.0 - sum(w_opt)
    w_opt.append(w_last_opt)

    # Normalizar para garantir soma exata a 1.0 por seguranca numerica
    w_opt = np.array(w_opt) / np.sum(w_opt)
    best_metric = -res.fun if mode == "classifier" else np.sqrt(res.fun)
    metric_name = "AUC" if mode == "classifier" else "RMSE"

    print("\n========================================")
    print("Otimizacao de Blending Completa!")
    print(f"Pesos Ideais:")
    for i, model in enumerate(models_list):
        print(f"  * {model}: {w_opt[i]:.4f}")
    print(f"Consolidado OOF {metric_name} Blended: {best_metric:.5f}")
    print("========================================")

    # 3. Carregar submissoes de teste e combinar
    sub_dfs = {}
    for model in models_list:
        path = config.SUBMISSIONS_DIR / f"submission_{model}.csv"
        if path.exists():
            sub_dfs[model] = pd.read_csv(path)

    if len(sub_dfs) < M:
        print("\n[AVISO] Nem todas as submissoes de teste foram encontradas.")
        print(f"Esperadas submissoes para: {models_list}")
        print("Inferencia de teste incompleta. Previsoes OOF otimizadas com sucesso!")
        return

    # Unificar predicoes via Blending
    sub_blend = sub_dfs[models_list[0]].copy()
    blend_preds_test = np.zeros(len(sub_blend))

    for i, model in enumerate(models_list):
        pred_test = sub_dfs[model][config.TARGET_COL].values
        if mode == "classifier":
            val_test = pd.Series(pred_test).rank(pct=True).values
        else:
            val_test = pred_test
        blend_preds_test += w_opt[i] * val_test

    sub_blend[config.TARGET_COL] = blend_preds_test

    blend_path = config.SUBMISSIONS_DIR / "submission_blend.csv"
    sub_blend.to_csv(blend_path, index=False)
    print(f"Submissao blended criada com sucesso em: {blend_path}!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="classifier", choices=["classifier", "regressor"]
    )
    args = parser.parse_args()
    blend_pipeline(mode=args.mode)
