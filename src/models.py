from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from src import config

from pytabkit.models.sklearn.sklearn_interfaces import (
    RealMLP_TD_Classifier,
    RealMLP_TD_Regressor,
    TabM_TD_Classifier,
    TabM_TD_Regressor,
)
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabicl import TabICLClassifier, TabICLRegressor


def get_model(model_name: str, mode: str = "classifier"):
    """
    Retorna a instância do modelo desejado baseado no nome e configuração.

    Args:
        model_name: Nome do algoritmo ('lightgbm', 'xgboost', 'catboost', 'logistic', 'mlp', 'knn')
        mode: Tipo de tarefa ('classifier' ou 'regressor')
    """
    params = config.MODEL_PARAMS.get(model_name, {})

    if mode == "classifier":
        if model_name == "lightgbm":
            return LGBMClassifier(**params)
        elif model_name == "xgboost":
            return XGBClassifier(**params)
        elif model_name == "catboost":
            return CatBoostClassifier(**params)
        elif model_name == "logistic":
            return LogisticRegression(**params)
        elif model_name == "mlp":
            return MLPClassifier(**params)
        elif model_name == "knn":
            return KNeighborsClassifier(**params)
        elif model_name == "realmlp":
            return RealMLP_TD_Classifier(**params)
        elif model_name == "tabm":
            return TabM_TD_Classifier(**params)
        elif model_name == "tabpfn":
            return TabPFNClassifier(**params)
        elif model_name == "tabicl":
            return TabICLClassifier(**params)
        else:
            raise ValueError(f"Modelo {model_name} desconhecido para classificação.")

    elif mode == "regressor":
        if model_name == "lightgbm":
            return LGBMRegressor(**params)
        elif model_name == "xgboost":
            return XGBRegressor(**params)
        elif model_name == "catboost":
            return CatBoostRegressor(**params)
        elif model_name == "logistic":
            return Ridge(**params)
        elif model_name == "mlp":
            return MLPRegressor(**params)
        elif model_name == "knn":
            return KNeighborsRegressor(**params)
        elif model_name == "realmlp":
            return RealMLP_TD_Regressor(**params)
        elif model_name == "tabm":
            return TabM_TD_Regressor(**params)
        elif model_name == "tabpfn":
            return TabPFNRegressor(**params)
        elif model_name == "tabicl":
            return TabICLRegressor(**params)
        else:
            raise ValueError(
                f"Modelo {model_name} desconhecido para regressão ou não suportado."
            )
    else:
        raise ValueError(
            f"Modo {mode} desconhecido. Escolha 'classifier' ou 'regressor'."
        )
