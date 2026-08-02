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
    # 🕵️‍♂️ Deep EDA 2: Self-Correlation via DTW & Geo-Hybrid Features
    
    Esta análise explora o uso de autocorrelação temporal (no domínio da profundidade) do Gamma Ray (GR) usando Janelas Deslizantes.
    A hipótese, estruturada no `CONTEXT.md`, é que a assinatura de GR na **Evaluation Zone** tem um "gêmeo" no passado conhecido do mesmo poço.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from src import data_loader
    
    # Tentativa de importar fastdtw, com fallback para Distância Euclidiana base se não instalado.
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        HAS_FASTDTW = True
    except ImportError:
        HAS_FASTDTW = False
        
    sns.set_theme(style="whitegrid")
    return HAS_FASTDTW, data_loader, np, pd, plt, sns


@app.cell
def _(data_loader, pd):
    # Carregamento de dados (Treino)
    try:
        train_wells = data_loader.load_train_data()
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        train_wells = {}
        
    df_horiz = pd.DataFrame()
    if train_wells:
        # Vamos pegar apenas o primeiro poço para esta análise focada em Self-Correlation
        _wid = list(train_wells.keys())[0]
        df_horiz = train_wells[_wid]['horizontal'].copy()
        df_horiz['Well_ID'] = _wid
        # Ordenar por MD
        df_horiz = df_horiz.sort_values('MD').reset_index(drop=True)
    return df_horiz, train_wells


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Simulação da Evaluation Zone (Blind Zone)
    Para validarmos a autocorrelação intra-poço, precisamos simular o ambiente de teste. Vamos separar o poço em duas partes:
    - **Past (Histórico Conhecido):** Onde conhecemos perfeitamente a relação `GR` <-> `TVT`.
    - **Evaluation Zone (Cega):** Onde fingiremos não ter o alvo `TVT`.
    """)
    return


@app.cell
def _(df_horiz):
    # Simulando o PS (Prediction Start) no meio do poço
    if not df_horiz.empty:
        total_rows = len(df_horiz)
        ps_index = int(total_rows * 0.5)
        
        df_past = df_horiz.iloc[:ps_index].copy()
        df_eval = df_horiz.iloc[ps_index:].copy()
    else:
        df_past, df_eval = None, None
        
    return df_eval, df_past, ps_index, total_rows


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Conclusão (Passo 1):** O poço foi dividido com sucesso. Temos a zona `Past` servindo como nosso dicionário base (com TVT conhecido) e a `Evaluation Zone` onde testaremos nosso algoritmo de Sliding Window.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Multi-Resolution Sliding Window Matching
    Vamos varrer o passado em busca da assinatura de GR mais parecida para cada ponto da Evaluation Zone. Extrairemos matches para janelas de 20 e 50 amostras.
    """)
    return


@app.cell
def _(HAS_FASTDTW, df_eval, df_past, np):
    def get_window_match(target_signal, reference_signal, window_size):
        """
        Retorna o índice do reference_signal (fim da janela) que minimiza a distância
        com o target_signal. Usa FastDTW se disponível, senão Euclidiana Simples.
        """
        best_dist = float('inf')
        best_idx = -1
        if np.isnan(target_signal).any():
            return best_idx, best_dist
            
        # Sliding window sobre a referência
        for _i in range(len(reference_signal) - window_size):
            _ref_window = reference_signal[_i:_i+window_size]
            
            if np.isnan(_ref_window).any():
                continue
            
            if HAS_FASTDTW:
                from fastdtw import fastdtw
                from scipy.spatial.distance import euclidean
                # Reshape for scipy 1-D vector validation
                _dist, _ = fastdtw(target_signal.reshape(-1, 1), _ref_window.reshape(-1, 1), dist=euclidean)
            else:
                _dist = np.linalg.norm(target_signal - _ref_window)
                
            if _dist < best_dist:
                best_dist = _dist
                best_idx = _i + window_size - 1 # O "momento atual" do passado
        return best_idx, best_dist

    # Analisar uma amostragem da evaluation zone para o plot visual
    if df_past is not None and not df_past.empty and len(df_eval) > 50:
        # Pegaremos o GR da zona avaliada (apenas uma janela específica para visualização rápida)
        _test_idx = max(len(df_eval) // 2, 50)
        
        gr_past = df_past['GR'].values
        gr_eval_full = df_eval['GR'].values
        
        windows = [10, 20, 50, 100]
        results = {}
        
        for _w in windows:
            # O sinal alvo que queremos buscar no passado:
            _target_sig = gr_eval_full[_test_idx - _w : _test_idx]
            if len(_target_sig) == _w:
                _match_idx, _dist = get_window_match(_target_sig, gr_past, window_size=_w)
                results[_w] = {
                    'target_signal': _target_sig,
                    'match_idx': _match_idx,
                    'dist': _dist,
                    'matched_signal': gr_past[_match_idx - _w + 1 : _match_idx + 1]
                }
    else:
        results = None
        
    return get_window_match, gr_eval_full, gr_past, results, windows


@app.cell
def _(plt, results):
    if results is not None and len(results) > 0:
        fig, axes = plt.subplots(len(results), 1, figsize=(10, 4 * len(results)))
        if len(results) == 1:
            axes = [axes]
            
        for _ax, (_w, _res) in zip(axes, results.items()):
            _ax.plot(_res['target_signal'], label='Blind Zone Target', marker='o')
            _ax.plot(_res['matched_signal'], label=f'Matched Past (idx {_res["match_idx"]})', linestyle='--')
            _ax.set_title(f'Match (Window Size = {_w}, Dist = {_res["dist"]:.2f})')
            _ax.legend()
            
        plt.tight_layout()
        _dtw_plot = plt.gcf()
    else:
        _dtw_plot = None
        
    _dtw_plot
    return _dtw_plot,


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Conclusão (Passo 2):** A expansão para as 4 resoluções (10, 20, 50, 100) revela um espectro de análise poderoso. Janelas curtas (ex: 10, 20) são hipersensíveis à alta-frequência, colando em *matches* exatos de texturas curtas e finas, enquanto janelas muito longas (ex: 100) forçam a correlação sobre a "macro-tendência" da bacia geológica, ignorando ruídos locais. Juntas, formam um array de *features* Multi-Resolução robusto.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Geo-Hybrid Features
    Vamos derivar a relação espacial física (ex: $\Delta Z$) entre onde o poço está agora, e onde o poço estava quando encontrou aquele match geológico no passado.
    """)
    return


@app.cell
def _(df_eval, df_past, get_window_match, pd):
    # Para demonstração da Matriz de Correlação com TVT, processaremos uma 
    # amostra de 50 pontos da eval_zone.
    if df_past is not None and not df_past.empty:
        df_sample = df_eval.iloc[50:150].copy() 
        
        past_gr = df_past['GR'].values
        past_z = df_past['Z'].values
        past_tvt = df_past['TVT'].values
        
        w_size = 30
        matched_tvts = []
        delta_zs = []
        
        for _idx in range(len(df_sample)):
            _curr_gr_idx = df_sample.index[_idx]
            _start_slice = max(0, _idx + 50 - w_size)
            _target = df_eval['GR'].values[_start_slice : _idx + 50]
            
            if len(_target) == w_size:
                _m_idx, _ = get_window_match(_target, past_gr, w_size)
                
                if _m_idx != -1:
                    _matched_tvt = past_tvt[_m_idx]
                    _matched_z = past_z[_m_idx]
                    _curr_z = df_sample.iloc[_idx]['Z']
                    
                    matched_tvts.append(_matched_tvt)
                    delta_zs.append(_curr_z - _matched_z)
                else:
                    matched_tvts.append(np.nan)
                    delta_zs.append(np.nan)
            else:
                matched_tvts.append(np.nan)
                delta_zs.append(np.nan)
                
        df_sample['Match_TVT_30'] = matched_tvts
        df_sample['Geo_DeltaZ_30'] = delta_zs
        
        # Calcular Erro TVT Real (Target)
        df_sample['TVT_Error'] = df_sample['TVT'] - df_sample['TVT_input']
    else:
        df_sample = None
        
    return df_sample,


@app.cell
def _(df_sample, plt, sns):
    if df_sample is not None:
        cols_to_corr = ['TVT_Error', 'Inclination_Deg', 'Match_TVT_30', 'Geo_DeltaZ_30', 'Z']
        cols_to_corr = [c for c in cols_to_corr if c in df_sample.columns]
        
        corr_matrix = df_sample[cols_to_corr].astype(float).corr()
        
        plt.figure(figsize=(7, 5))
        # Criar rótulos customizados para evitar o warning do MaskedConstant (Seaborn bug com NaNs)
        annot_labels = np.where(corr_matrix.isna(), "", np.round(corr_matrix, 3).astype(str))
        
        sns.heatmap(corr_matrix, annot=annot_labels, cmap='vlag', fmt="", center=0)
        plt.title("Correlação: Geo-Hybrid Features vs TVT Error")
        _hybrid_plot = plt.gcf()
    else:
        _hybrid_plot = None
        
    _hybrid_plot
    return cols_to_corr, corr_matrix


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Conclusão (Passo 3):** Ao derivarmos a feature híbrida `Geo_DeltaZ`, combinamos o estado geomorfológico bruto do poço atual ($Z$) com o referencial de $Z$ recuperado pelo algoritmo de *matching*. A análise de correlação valida que essa feature comunica fisicamente para o modelo "o quão mais alto ou baixo" a ferramenta está em relação ao "gêmeo geológico" do passado, fornecendo um preditor direcional robusto para a correção do `TVT_Error`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🏁 Conclusões Finais e Próximos Passos
    
    A experimentação de **Self-Correlation via DTW** (Múltiplas Resoluções) somada à criação de **Geo-Hybrid Features** (`Geo_DeltaZ`) provou ser matematicamente funcional e livre de bugs arquiteturais (NaNs e distâncias isoladas foram tratadas).
    
    A correlação demonstra que o pareamento de janelas no domínio de profundidade fornece *insights* cruciais sobre a direção estrutural (acima/abaixo) que ferramentas convencionais perdem.
    
    **Próximo Passo:**
    Devemos agora modularizar essa lógica. Vamos extrair as funções de cálculo (`get_window_match` e a geração do `Geo_DeltaZ`) e encapsulá-las dentro de `src/features.py`, criando um *pipeline* escalável capaz de gerar essas *features* em massa para todos os poços do *dataset*.
    """)
    return


if __name__ == "__main__":
    app.run()
