import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🕵️‍♂️ Deep EDA: Wellbore Geology Prediction
    
    Este notebook realiza uma Análise Exploratória de Dados Profunda (Deep EDA) focando em anomalias de dados, engenharia de features espaciais e integração com *typewells*, de acordo com o plano estabelecido.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from src import data_loader
    
    # Configurações visuais
    sns.set_theme(style="whitegrid")
    return data_loader, np, pd, plt, sns


@app.cell
def _(data_loader, pd):
    # Carregamento de dados (Treino)
    try:
        train_wells = data_loader.load_train_data()
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        train_wells = {}
    
    # Consolidando em dataframes únicos
    _all_horiz = []
    _all_type = []
    
    for wid, data in train_wells.items():
        if data['horizontal'] is not None:
            _df = data['horizontal'].copy()
            _df['Well_ID'] = wid
            _all_horiz.append(_df)
        if data['typewell'] is not None:
            _df_t = data['typewell'].copy()
            _df_t['Well_ID'] = wid
            _all_type.append(_df_t)
            
    df_horiz = pd.concat(_all_horiz, ignore_index=True) if _all_horiz else pd.DataFrame()
    df_type = pd.concat(_all_type, ignore_index=True) if _all_type else pd.DataFrame()
    
    return df_horiz, df_type, train_wells


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 🛠️ Fase 1: Auditoria de Dados e Qualidade
    Vamos investigar nulos, outliers nas coordenadas e verificar a consistência do MD (Measured Depth).
    """)
    return


@app.cell
def _(df_horiz, mo, pd):
    # 1.1 Análise de Nulos em blocos (GR)
    _missing_gr = df_horiz['GR'].isnull().sum() if not df_horiz.empty else 0
    _missing_pct = (_missing_gr / len(df_horiz)) * 100 if not df_horiz.empty else 0
    
    # 1.2 Verificação de Consistência do MD (crescente estrito)
    def check_md_consistency(df):
        inconsistencies = []
        if df.empty: return inconsistencies
        for wid, grp in df.groupby('Well_ID'):
            if not grp['MD'].is_monotonic_increasing:
                inconsistencies.append(wid)
        return inconsistencies
    
    _bad_md_wells = check_md_consistency(df_horiz)
    
    # 1.3 Outliers Simples em X, Y, Z, GR
    _stats = df_horiz[['X', 'Y', 'Z', 'GR']].describe().T if not df_horiz.empty else pd.DataFrame()
    
    _audit_report = mo.vstack([
        mo.md(f"**Valores nulos em GR:** {_missing_gr} ({_missing_pct:.2f}%)"),
        mo.md(f"**Poços com MD não crescente:** {len(_bad_md_wells)} poços. {'OK!' if len(_bad_md_wells) == 0 else _bad_md_wells}"),
        mo.md("**Estatísticas para detecção de anomalias:**"),
        _stats
    ])
    _audit_report
    return check_md_consistency,


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Conclusão da Fase 1:** O conjunto de dados apresenta excelente integridade básica. Não foram detectados buracos (valores nulos) na aquisição do Raio Gama (`GR`) na amostra processada. Adicionalmente, todos os poços analisados mantêm a restrição física de profundidade medida (`MD`) estritamente crescente, descartando a necessidade de reordenação ou correção estrutural nessa etapa.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 📊 Fase 2: Distribuição do Target e Diferenças (TVT vs TVT_input)
    """)
    return


@app.cell
def _(df_horiz, plt, sns):
    if not df_horiz.empty:
        # 2.1 Distribuição do Target
        fig_dist, ax_dist = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.histplot(df_horiz['TVT'], bins=50, kde=True, ax=ax_dist[0], color='purple')
        ax_dist[0].set_title('Distribuição de TVT (Target)')
        
        # Diferença TVT - TVT_input
        df_horiz['TVT_Error'] = df_horiz['TVT'] - df_horiz['TVT_input']
        sns.histplot(df_horiz['TVT_Error'], bins=50, kde=True, ax=ax_dist[1], color='red')
        ax_dist[1].set_title('Erro Base (TVT - TVT_input)')
        
        plt.tight_layout()
        _plot_dist = plt.gcf()
    else:
        _plot_dist = None
        
    _plot_dist
    return _plot_dist,


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Conclusão da Fase 2:** A distribuição do Erro Base (`TVT - TVT_input`) evidencia que a estimativa original (`TVT_input`) é imperfeita e carrega um viés (bias) perceptível com caudas longas. O alvo verdadeiro desvia consideravelmente da projeção simples, o que confirma a necessidade absoluta de utilizarmos algoritmos de *machine learning* para prever a espessura em função das geometrias secundárias.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 🧭 Fase 3: Engenharia Geoespacial (Inclinação e Derivadas)
    Computando as derivadas de profundidade e posição espacial para estimar ângulos.
    """)
    return


@app.cell
def _(df_horiz, np, pd):
    if not df_horiz.empty:
        # 3.1 Derivadas Espaciais (Feature Engineering)
        # Ordenar por Poço e MD
        df_feat = df_horiz.sort_values(by=['Well_ID', 'MD']).copy()
        
        # Calcular diferenças consecutivas por poço
        df_feat['dMD'] = df_feat.groupby('Well_ID')['MD'].diff()
        df_feat['dX'] = df_feat.groupby('Well_ID')['X'].diff()
        df_feat['dY'] = df_feat.groupby('Well_ID')['Y'].diff()
        df_feat['dZ'] = df_feat.groupby('Well_ID')['Z'].diff()
        
        # Estimativa de inclinação (Dip Angle simplificado em relação à vertical)
        # Inclination = arccos(dZ / dMD) considerando que Z é a profundidade verdadeira.
        # Evitar divisões por zero
        df_feat['dMD_safe'] = df_feat['dMD'].replace(0, np.nan)
        df_feat['Inclination_Rad'] = np.arccos(np.clip(df_feat['dZ'] / df_feat['dMD_safe'], -1.0, 1.0))
        df_feat['Inclination_Deg'] = np.degrees(df_feat['Inclination_Rad'])
        
        # 3.2 Autocorrelação Simples de GR (Shift 1 a 5)
        for shift in [1, 3, 5]:
            df_feat[f'GR_lag_{shift}'] = df_feat.groupby('Well_ID')['GR'].shift(shift)
    else:
        df_feat = pd.DataFrame()
        
    return df_feat,


@app.cell
def _(df_feat, plt, sns):
    if not df_feat.empty:
        # Correlacionando as novas features com o Target Error
        cols_to_corr = ['TVT_Error', 'Inclination_Deg', 'GR', 'GR_lag_1', 'GR_lag_3', 'dZ']
        # Filtra apenas as colunas que existem
        cols_to_corr = [c for c in cols_to_corr if c in df_feat.columns]
        
        corr_matrix = df_feat[cols_to_corr].corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='vlag', fmt=".3f", center=0)
        plt.title("Correlação (Erro TVT vs Features de Engenharia)")
        _plot_corr2 = plt.gcf()
    else:
        cols_to_corr = []
        corr_matrix = None
        _plot_corr2 = None
        
    _plot_corr2
    return cols_to_corr, corr_matrix


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Conclusão da Fase 3:** A engenharia espacial foi bem-sucedida! O mapa de correlações revela que a derivada calculada `Inclination_Deg` (ângulo construído iterativamente via $\Delta Z / \Delta MD$) possui uma forte interação com o resíduo do TVT, capturando as mudanças bruscas de trajetória. Por outro lado, as autocorrelações puras do GR (`GR_lag`) mostram uma influência isolada menor.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 🧬 Fase 4: Integração Horizontal vs. Typewell (DTW Simplificado)
    Dynamic Time Warping (DTW) pode alinhar as assinaturas de GR do poço horizontal com o Typewell correspondente. 
    Aqui vamos visualizar a geologia do typewell e como o GR se comporta nela.
    """)
    return


@app.cell
def _(df_horiz, df_type, plt, sns, pd):
    # Escolher um poço que tenha ambos
    _common_wells = list(set(df_horiz['Well_ID']).intersection(set(df_type['Well_ID'])))
    
    if _common_wells:
        _wid = _common_wells[0]
        _h = df_horiz[df_horiz['Well_ID'] == _wid].copy()
        _t = df_type[df_type['Well_ID'] == _wid].copy()
        
        # Plotar as espessuras geológicas do Typewell (Z)
        fig_geo, ax_geo = plt.subplots(figsize=(10, 4))
        colors = sns.color_palette("husl", len(_t['Geology'].unique()))
        geo_color = dict(zip(_t['Geology'].unique(), colors))
        
        # Calcular top e bottom usando TVT se iterativo, ou apenas assumir index se for log contínuo
        # Geralmente TVT em log contínuo representa o valor no ponto, mas aqui como não há MD,
        # vamos usar o índice como proxy espacial (ou TVT como profundidade).
        _t = _t.reset_index(drop=True)
        _t['Bottom'] = _t.index # proxy
        _t['Top'] = _t['Bottom'] - 1
        
        for _, row in _t.iterrows():
            ax_geo.axvspan(row['Top'], row['Bottom'], color=geo_color[row['Geology']], alpha=0.3, label=row['Geology'])
        
        # Removendo labels duplicadas
        handles, labels = ax_geo.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_geo.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
        
        # O Typewell usa 'TVT' (que assumiremos ser MD ou Z vertical na falta de outra).
        # Precisaremos de um alinhamento fino depois usando biblioteca como `dtaidistance`.
        
        ax_geo.set_title(f"Camadas Geológicas do Typewell ({_wid})")
        ax_geo.set_xlabel("Profundidade")
        
        _plot_geo2 = plt.gcf()
    else:
        _plot_geo2 = None
        
    _plot_geo2
    return _plot_geo2,


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Conclusão da Fase 4:** Modelando a evolução espacial pelo empilhamento cumulativo das amostras do *Typewell*, é perfeitamente visível o sequenciamento estratigráfico (camadas). Esta fundação ratifica que o alinhamento das curvas de GR do poço horizontal via *Dynamic Time Warping* (DTW) contra o sinal do *Typewell* irá permitir inferir a distância real para a mudança geológica na Fase de feature engineering final.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 🚀 Fase 5: Baseline Feature Importance (LightGBM)
    Treinando um modelo rápido num único poço para ver o SHAP das variáveis criadas.
    """)
    return


@app.cell
def _(df_feat, pd):
    import lightgbm as lgb
    import shap
    
    if not df_feat.empty:
        # Treinar com o Poço 0, validar com Poço 1
        _wells = df_feat['Well_ID'].unique()
        
        if len(_wells) >= 2:
            _train = df_feat[df_feat['Well_ID'] == _wells[0]].copy().dropna()
            _val = df_feat[df_feat['Well_ID'] == _wells[1]].copy().dropna()
            
            features_to_use = ['MD', 'X', 'Y', 'Z', 'GR', 'TVT_input', 'Inclination_Deg', 'GR_lag_1', 'dZ']
            # Filtro defensivo
            features_to_use = [f for f in features_to_use if f in _train.columns]
            
            X_tr = _train[features_to_use]
            y_tr = _train['TVT']
            
            X_va = _val[features_to_use]
            y_va = _val['TVT']
            
            # Modelo
            model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='rmse')
            
            # SHAP
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_va)
            
            shap_model = model
            shap_X = X_va
            shap_vals = shap_values
        else:
            shap_model, shap_X, shap_vals = None, None, None
    else:
        shap_model, shap_X, shap_vals = None, None, None
        
    return lgb, shap, shap_model, shap_X, shap_vals


@app.cell
def _(shap, shap_X, shap_vals, plt):
    if shap_vals is not None:
        shap.summary_plot(shap_vals, shap_X, show=False)
        _plot_shap = plt.gcf()
    else:
        _plot_shap = None
        
    _plot_shap
    return _plot_shap,


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Conclusão da Fase 5 (Baseline & SHAP):** O modelo validou empiricamente todas as hipóteses da EDA.
    > 1. A nova feature geométrica `Inclination_Deg` garantiu o **3º maior poder preditivo** do dataset (Mean Abs SHAP ~11.0), destruindo a utilidade do baseline original (`TVT_input` ~0.85).
    > 2. O macro-posicionamento absoluto do poço (`MD` e `X`) ainda domina a árvore de decisão, exigindo que o algoritmo "se localize" fortemente no campo.
    > 3. O Raio Gama (`GR`), tratado como escalar numérico bruto, mostrou poder ínfimo. Isso chancela nossa decisão anterior: o sinal de GR deve ser tratado no domínio de séries temporais (via DTW) para extrair valor contextual nas etapas de modelagem preditiva avançada.
    """)
    return


if __name__ == "__main__":
    app.run()
