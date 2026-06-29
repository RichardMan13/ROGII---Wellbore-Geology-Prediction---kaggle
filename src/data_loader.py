import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from src import config


def load_train_data():
    """Carrega o conjunto de dados de treino."""
    train_path = config.RAW_DATA_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Arquivo de treino não encontrado em {train_path}. Por favor, baixe os dados usando a API do Kaggle."
        )
    return pd.read_csv(train_path)


def load_test_data():
    """Carrega o conjunto de dados de teste."""
    test_path = config.RAW_DATA_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Arquivo de teste não encontrado em {test_path}. Por favor, baixe os dados usando a API do Kaggle."
        )
    return pd.read_csv(test_path)


def get_cv_splits(df, mode="classifier"):
    """
    Gera as dobras de Cross-Validation com base na configuração e modo.

    Retorna:
        Gerador de tuplas (train_idx, val_idx)
    """
    is_stratified = config.STRATIFIED if mode == "classifier" else False

    if is_stratified:
        kf = StratifiedKFold(
            n_splits=config.N_SPLITS, shuffle=config.SHUFFLE, random_state=config.SEED
        )
        return list(kf.split(df, df[config.TARGET_COL]))
    else:
        kf = KFold(
            n_splits=config.N_SPLITS, shuffle=config.SHUFFLE, random_state=config.SEED
        )
        return list(kf.split(df))
