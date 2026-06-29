import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from src import config


def load_train_data():
    """Carrega o conjunto de dados de treino (dicionário de poços)."""
    train_path = config.RAW_DATA_DIR / "train"
    if not train_path.exists() or not train_path.is_dir():
        raise FileNotFoundError(
            f"Pasta de treino não encontrada em {train_path}. Por favor, rode 'inv download-data'."
        )
        
    wells = {}
    for h_path in train_path.glob("*__horizontal_well.csv"):
        well_id = h_path.name.split("__")[0]
        t_path = train_path / f"{well_id}__typewell.csv"
        
        wells[well_id] = {
            "horizontal": pd.read_csv(h_path),
            "typewell": pd.read_csv(t_path) if t_path.exists() else None
        }
    return wells


def load_test_data():
    """Carrega o conjunto de dados de teste (dicionário de poços)."""
    test_path = config.RAW_DATA_DIR / "test"
    if not test_path.exists() or not test_path.is_dir():
        raise FileNotFoundError(
            f"Pasta de teste não encontrada em {test_path}. Por favor, rode 'inv download-data'."
        )
        
    wells = {}
    for h_path in test_path.glob("*__horizontal_well.csv"):
        well_id = h_path.name.split("__")[0]
        wells[well_id] = {
            "horizontal": pd.read_csv(h_path)
        }
    return wells


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
