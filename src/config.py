import os
from pathlib import Path

from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Carrega o token do arquivo access_token se KAGGLE_API_TOKEN não estiver definido
if "KAGGLE_API_TOKEN" not in os.environ:
    ACCESS_TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
    if ACCESS_TOKEN_PATH.exists():
        try:
            token = ACCESS_TOKEN_PATH.read_text(encoding="utf-8").strip()
            if token:
                os.environ["KAGGLE_API_TOKEN"] = token
        except Exception:
            pass

# Valida se o token foi configurado com sucesso
if "KAGGLE_API_TOKEN" not in os.environ:
    token_path = Path.home() / ".kaggle" / "access_token"
    print(
        f"\n[AVISO] A chave de API do Kaggle ('KAGGLE_API_TOKEN') não está configurada!\nPor favor, configure o seu token de acesso salvando-o no arquivo: {token_path}\nou definindo KAGGLE_API_TOKEN no arquivo '.env' do seu projeto se for usar os downloads do Kaggle.\n"
    )

# Caminhos do Projeto
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

# Configurações Globais de Reprodutibilidade
SEED = 25844181

# Configuração de Cross-Validation (Mudar conforme a competição)
N_SPLITS = 5
SHUFFLE = True
STRATIFIED = False  # True para classificação, False para regressão

# Configurações da Competição
PROBLEM_TYPE = "regression"  # "binary", "multiclass" ou "regression"
EVAL_METRIC = "rmse"      # ex: "auc", "logloss", "rmse", "mae", "accuracy"

# Configurações de Dados
TARGET_COL = "TVT"
ID_COL = "id"

# Parâmetros dos Modelos
MODEL_PARAMS = {
    "lightgbm": {
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "random_state": SEED,
        "verbose": -1,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "random_state": SEED,
        "n_jobs": -1,
    },
    "catboost": {
        "iterations": 1000,
        "learning_rate": 0.05,
        "random_seed": SEED,
        "verbose": 0,
        "thread_count": -1,
    },
    "logistic": {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": SEED,
    },
    "mlp": {
        "hidden_layer_sizes": (100,),
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
        "activation": "relu",
        "solver": "adam",
        "max_iter": 500,
        "early_stopping": True,
        "random_state": SEED,
    },
    "knn": {
        "n_neighbors": 5,
        "weights": "uniform",
        "metric": "euclidean",
        "n_jobs": -1,
    },
    "realmlp": {
        "device": "auto",
        "verbosity": 0,
    },
    "tabm": {
        "device": "auto",
        "verbosity": 0,
    },
    "tabpfn": {
        "device": "auto",
        "N_ensemble_configurations": 3,
    },
    "tabicl": {
        "device": "auto",
    },
}
