import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Realiza a engenharia de features no DataFrame. 
    Como é uma classe compativel com scikit-learn, o estado 
    salvo no `fit` previne vazamento de dados no `transform`.
    """
    
    def __init__(self):
        # Exemplo: variáveis para salvar estados
        self.state_ = {}

    def fit(self, X, y=None):
        # -------------------------------------------------------------
        # TODO: Se suas features exigem estatísticas do treino
        # calcule-as aqui e salve no self.state_ (ex: Target Encoding)
        # -------------------------------------------------------------
        return self

    def transform(self, X, y=None):
        X = X.copy()
        
        # -------------------------------------------------------------
        # TODO: Implemente suas features aqui
        # Exemplo: X['feature_comb'] = X['feat_1'] * X['feat_2']
        # Exemplo: X['log_feature'] = np.log1p(X['numeric_feat'])
        # -------------------------------------------------------------
        return X


class CollinearityPruner(BaseEstimator, TransformerMixin):
    """
    Remove colunas numéricas que possuem correlação de Pearson (em valor absoluto)
    maior que o threshold configurado. O cálculo é feito no `fit` e as colunas a serem
    removidas são lembradas para o `transform`.
    """

    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.drop_cols_ = []

    def fit(self, X, y=None):
        # Seleciona apenas colunas numéricas
        X_num = X.select_dtypes(include=[np.number])
        if X_num.empty:
            self.drop_cols_ = []
            return self

        # Calcula a matriz de correlação em valor absoluto
        corr_matrix = X_num.corr().abs()

        # Seleciona o triângulo superior da matriz de correlação
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Encontra colunas com correlação maior que o limite
        self.drop_cols_ = [
            column for column in upper.columns if any(upper[column] > self.threshold)
        ]

        if self.drop_cols_:
            print(
                f"CollinearityPruner identificou {len(self.drop_cols_)} coluna(s) colinear(es) para remover."
            )

        return self

    def transform(self, X, y=None):
        if self.drop_cols_:
            # Remove as colunas colineares (ignorando erros se elas já não estiverem presentes por algum motivo)
            return X.drop(columns=self.drop_cols_, errors="ignore")
        return X
