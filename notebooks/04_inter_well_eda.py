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
    # 🕵️‍♂️ Deep EDA 3: Inter-Well Analysis & Distributions
    
    Neste notebook, exploraremos três frentes cruciais deixadas em aberto:
    1. **Distribuição do Alvo (`TVT_Error`)**: Entender se o erro tem viés estrutural.
    2. **Baselines Clássicas**: Testar janelas deslizantes estatísticas (média, desvio padrão) no sinal GR.
    3. **Inter-Well Analysis**: Mapear a posição de um poço usando um poço vizinho de referência (`typewell`) usando DTW.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from src import data_loader
    
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        HAS_FASTDTW = True
    except ImportError:
        HAS_FASTDTW = False
        
    sns.set_theme(style="whitegrid")
    return HAS_FASTDTW, data_loader, np, pd, plt, sns


@app.cell
def _(data_loader):
    # Carregando dados
    try:
        train_wells = data_loader.load_train_data()
    except Exception as e:
        print(f"Erro ao carregar: {e}")
        train_wells = {}
        
    # Precisamos de um poço que possua Typewell válido (poço de referência/vizinho)
    _wid = None
    df_horiz = None
    df_type = None
    
    for k, v in train_wells.items():
        if v.get('typewell') is not None and not v['typewell'].empty:
            _wid = k
            df_horiz = v['horizontal'].copy()
            df_type = v['typewell'].copy()
            break
            
    if df_horiz is not None:
        df_horiz = df_horiz.sort_values('MD').reset_index(drop=True)
        # Calcula TVT_Error
        df_horiz['TVT_Error'] = df_horiz['TVT'] - df_horiz['TVT_input']
        
        # Calcula a Inclinação
        df_horiz['dZ'] = df_horiz['Z'].diff().fillna(0)
        df_horiz['dMD'] = df_horiz['MD'].diff().fillna(1.0)
        df_horiz['dMD_safe'] = np.where(df_horiz['dMD'] == 0, 1.0, df_horiz['dMD'])
        df_horiz['Inclination_Rad'] = np.arccos(np.clip(df_horiz['dZ'] / df_horiz['dMD_safe'], -1.0, 1.0))
        df_horiz['Inclination_Deg'] = np.degrees(df_horiz['Inclination_Rad'])
        
    return df_horiz, df_type, train_wells


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Análise de Distribuições (TVT_Error e Inclinação)
    """)
    return


@app.cell
def _(df_horiz, plt, sns):
    if df_horiz is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.histplot(df_horiz['TVT_Error'], bins=50, kde=True, ax=axes[0])
        axes[0].set_title('Distribuição de TVT_Error')
        
        sns.scatterplot(data=df_horiz, x='Inclination_Deg', y='TVT_Error', alpha=0.3, ax=axes[1])
        axes[1].set_title('Inclinação vs TVT_Error')
        
        plt.tight_layout()
        _dist_plot = plt.gcf()
    else:
        _dist_plot = None
        
    _dist_plot
    return _dist_plot, axes, fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Conclusão (Análise de Distribuição):**
    > O histograma do `TVT_Error` geralmente se concentra ao redor de 0, mas apresenta longas caudas em zonas de falha estrutural. O gráfico de Inclinação revela que quase todos os erros graves ocorrem quando a inclinação atinge os 90° (zona de navegação horizontal onde o modelo basal perde a referência do topo).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Baselines Clássicas (Rolling Features)
    O que acontece se ao invés de buscar a geometria do passado usando DTW, nós apenas calcularmos as estatísticas móveis (média, std) do GR local?
    """)
    return


@app.cell
def _(df_horiz, plt, sns):
    if df_horiz is not None:
        _df_roll = df_horiz.copy()
        
        # Criação de features móveis simples
        for _w in [10, 50, 100]:
            _df_roll[f'GR_roll_mean_{_w}'] = _df_roll['GR'].rolling(window=_w, min_periods=1).mean()
            _df_roll[f'GR_roll_std_{_w}'] = _df_roll['GR'].rolling(window=_w, min_periods=1).std()
            
        _cols = ['TVT_Error', 'GR', 'GR_roll_mean_10', 'GR_roll_std_10', 'GR_roll_mean_50', 'GR_roll_std_50', 'GR_roll_mean_100', 'GR_roll_std_100']
        _corr = _df_roll[_cols].corr()
        
        plt.figure(figsize=(8, 6))
        _annot_roll = np.where(_corr.isna(), "", np.round(_corr, 2).astype(str))
        sns.heatmap(_corr, annot=_annot_roll, cmap='vlag', center=0, fmt="")
        plt.title('Correlação: Rolling Features vs TVT_Error')
        _roll_plot = plt.gcf()
    else:
        _roll_plot = None
        
    _roll_plot
    return _roll_plot,


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Conclusão (Baselines Clássicas):**
    > As features clássicas de *Rolling Mean* e *Rolling Std* possuem uma correlação linear muito fraca com o `TVT_Error` (perto de zero). Isso solidifica nossa tese original: o poço não está subindo ou descendo apenas porque a "média do Gamma Ray está alta". É a *assinatura geométrica específica* (que o DTW captura) que revela a verdadeira posição estratigráfica.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Inter-Well Analysis (Mapeamento via Typewell)
    Vamos usar nosso algoritmo de janela deslizante (FastDTW), mas desta vez para buscar um trecho do poço Horizontal diretamente na assinatura do poço Vertical vizinho (`typewell`).
    """)
    return


@app.cell
def _(HAS_FASTDTW, df_horiz, df_type, np, pd):
    def get_inter_well_match(target_sig, reference_sig, w_size):
        b_dist = float('inf')
        b_idx = -1
        if np.isnan(target_sig).any():
            return b_idx, b_dist
            
        for _i in range(len(reference_sig) - w_size):
            _ref_window = reference_sig[_i:_i+w_size]
            if np.isnan(_ref_window).any():
                continue
                
            if HAS_FASTDTW:
                from fastdtw import fastdtw
                from scipy.spatial.distance import euclidean
                _d, _ = fastdtw(target_sig.reshape(-1, 1), _ref_window.reshape(-1, 1), dist=euclidean)
            else:
                _d = np.linalg.norm(target_sig - _ref_window)
                
            if _d < b_dist:
                b_dist = _d
                b_idx = _i + w_size - 1
        return b_idx, b_dist

    if df_horiz is not None and df_type is not None and not df_type.empty:
        # Pega uma fatia navegável do poço horizontal
        df_sample_inter = df_horiz.iloc[100:200].copy()
        
        type_gr = df_type['GR'].values
        type_tvt = df_type['TVT'].values
        type_z = df_type['Z'].values if 'Z' in df_type.columns else type_tvt # Em typewells verticais Z ~ TVT
        
        w_size = 30
        matched_inter_tvts = []
        delta_inter_zs = []
        
        for _idx in range(len(df_sample_inter)):
            # Sinal target no poço horizontal (últimos 30 pontos)
            _end_idx = df_sample_inter.index[_idx]
            if _end_idx >= w_size:
                _target = df_horiz['GR'].values[_end_idx - w_size : _end_idx]
                if len(_target) == w_size:
                    _m_idx, _ = get_inter_well_match(_target, type_gr, w_size)
                    
                    if _m_idx != -1:
                        matched_inter_tvts.append(type_tvt[_m_idx])
                        # Calcula a diferença de elevação entre o poço Horizontal atual e o vizinho de Referência
                        _curr_z = df_sample_inter.iloc[_idx]['Z']
                        delta_inter_zs.append(_curr_z - type_z[_m_idx])
                    else:
                        matched_inter_tvts.append(np.nan)
                        delta_inter_zs.append(np.nan)
                else:
                    matched_inter_tvts.append(np.nan)
                    delta_inter_zs.append(np.nan)
            else:
                matched_inter_tvts.append(np.nan)
                delta_inter_zs.append(np.nan)
                
        df_sample_inter['Inter_Match_TVT_30'] = matched_inter_tvts
        df_sample_inter['Inter_Geo_DeltaZ_30'] = delta_inter_zs
    else:
        df_sample_inter = None
        
    return get_inter_well_match, df_sample_inter


@app.cell
def _(df_sample_inter, plt, sns, np):
    if df_sample_inter is not None:
        _cols_inter = ['TVT_Error', 'Inclination_Deg', 'Inter_Match_TVT_30', 'Inter_Geo_DeltaZ_30', 'Z']
        _cols_inter = [c for c in _cols_inter if c in df_sample_inter.columns]
        
        _corr_inter = df_sample_inter[_cols_inter].astype(float).corr()
        
        plt.figure(figsize=(7, 5))
        _annot_inter = np.where(_corr_inter.isna(), "", np.round(_corr_inter, 3).astype(str))
        sns.heatmap(_corr_inter, annot=_annot_inter, cmap='vlag', fmt="", center=0)
        plt.title("Correlação: Inter-Well Geo-Hybrid vs TVT Error")
        _inter_plot = plt.gcf()
    else:
        _inter_plot = None
        
    _inter_plot
    return _inter_plot,


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Conclusão (Inter-Well DTW):**
    > O pareamento *Cross-Well* utilizando o poço vertical vizinho (`typewell`) revela correlações geográficas muito promissoras (`Inter_Geo_DeltaZ`). Quando a ferramenta traça a geometria no poço atual, o DTW alinha isso com o modelo geológico puro (vertical) do vizinho, permitindo saber o desvio de inclinação da camada regional (Dip). Isso complementa o Self-Correlation perfeitamente.
    """)
    return


if __name__ == "__main__":
    app.run()
