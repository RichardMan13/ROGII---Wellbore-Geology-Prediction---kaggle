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
    # Análise Exploratória de Dados (EDA)

    Notebook inicial estruturado para investigação de dados, engenharia rápida de features e testes de hipóteses.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Requer que o pacote local 'src' tenha sido instalado via 'pip install -e .'
    from src import config, data_loader, features

    return data_loader, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Carregamento de Dados
    """)
    return


@app.cell
def _(data_loader):
    try:
        train_wells = data_loader.load_train_data()
        test_wells = data_loader.load_test_data()
        print(f"Número de poços no Treino: {len(train_wells)}")
        print(f"Número de poços no Teste: {len(test_wells)}")

        # Exemplo de inspeção do primeiro poço
        first_well_id = list(train_wells.keys())[0]
        first_well = train_wells[first_well_id]
        print(f"\nExemplo (Poço {first_well_id}):")
        print(f"  Horizontal shape: {first_well['horizontal'].shape}")
        if first_well['typewell'] is not None:
            print(f"  Typewell shape: {first_well['typewell'].shape}")
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        train_wells, test_wells = {}, {}
        first_well_id, first_well = None, None
    return first_well, first_well_id, train_wells


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspeção Inicial
    """)
    return


@app.cell
def _(first_well, first_well_id, mo):
    # Visualizar primeiras linhas do poço selecionado
    _layout = mo.vstack([
        mo.md(f"### Primeiras Linhas - Horizontal (Poço {first_well_id})"),
        first_well['horizontal'].head(),
        mo.md(f"### Primeiras Linhas - Typewell (Poço {first_well_id})"),
        first_well['typewell'].head() if first_well['typewell'] is not None else mo.md("Sem typewell")
    ])

    _layout
    return


@app.cell
def _(plt, sns, train_wells):
    import random
    # Selecionar um poço aleatório para visualização detalhada
    _well_id = random.choice(list(train_wells.keys()))
    _df_horiz = train_wells[_well_id]['horizontal']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- Gráfico 1: TVT vs MD ---
    sns.lineplot(data=_df_horiz, x='MD', y='TVT', label='TVT Real (Target)', color='blue', linewidth=2, ax=ax1)
    sns.lineplot(data=_df_horiz, x='MD', y='TVT_input', label='TVT Input (Features)', color='orange', linestyle='--', linewidth=2.5, ax=ax1)

    ax1.set_title(f"Visualização de TVT e GR ao longo do poço: {_well_id}")
    ax1.set_ylabel("True Vertical Thickness (TVT)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Gráfico 2: GR vs MD ---
    sns.lineplot(data=_df_horiz, x='MD', y='GR', label='Gamma Ray (GR)', color='green', linewidth=1.5, ax=ax2)

    ax2.set_xlabel("Measured Depth (MD)")
    ax2.set_ylabel("Gamma Ray (GR)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    _plot = plt.gcf()
    _plot
    return (random,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Qualidade de Dados e Estatística Descritiva
    """)
    return


@app.cell
def _(mo, pd, train_wells):
    _all_horiz = []
    _all_type = []
    _shapes_data = []

    # Coletar dados e shapes de todos os poços
    for _wid, _data in train_wells.items():
        _h_df = _data['horizontal']
        _t_df = _data['typewell']

        _shapes_data.append({
            "Well_ID": _wid,
            "Horizontal_Rows": _h_df.shape[0] if _h_df is not None else 0,
            "Typewell_Rows": _t_df.shape[0] if _t_df is not None else 0
        })

        if _h_df is not None:
            _all_horiz.append(_h_df)
        if _t_df is not None:
            _all_type.append(_t_df)

    df_horiz_all = pd.concat(_all_horiz, ignore_index=True)
    df_type_all = pd.concat(_all_type, ignore_index=True)
    df_shapes = pd.DataFrame(_shapes_data)

    # Valores Nulos
    _miss_h = (df_horiz_all.isnull().sum() / len(df_horiz_all)) * 100
    _miss_h_df = pd.DataFrame({"Nulos (%)": _miss_h}).sort_values("Nulos (%)", ascending=False)

    _miss_t = (df_type_all.isnull().sum() / len(df_type_all)) * 100
    _miss_t_df = pd.DataFrame({"Nulos (%)": _miss_t}).sort_values("Nulos (%)", ascending=False)

    _layout_dq = mo.vstack([
        mo.md("### Resumo do Tamanho (Shape) dos Poços"),
        mo.md("Abaixo vemos a estatística da quantidade de linhas por poço:"),
        df_shapes.describe(),
        mo.md("---"),
        mo.md("### Qualidade e Descritiva: Dados Horizontais"),
        mo.hstack([_miss_h_df, df_horiz_all.describe().T]),
        mo.md("---"),
        mo.md("### Qualidade e Descritiva: Dados Typewell"),
        mo.hstack([_miss_t_df, df_type_all.describe(include='all').T])
    ])

    _layout_dq
    return df_horiz_all, df_type_all


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Análises Aprofundadas
    Distribuição Geológica, Matriz de Correlações e Visualização da Trajetória 3D.
    """)
    return


@app.cell
def _(df_type_all, plt, sns):
    # 1. Distribuição das Formações Geológicas nos Typewells
    plt.figure(figsize=(10, 5))
    _ax_geo = sns.countplot(data=df_type_all, x='Geology', order=df_type_all['Geology'].value_counts().index, palette="viridis")
    _ax_geo.set_title("Distribuição das Camadas Geológicas (Typewells)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    _plot_geo = plt.gcf()
    _plot_geo
    return


@app.cell
def _(df_horiz_all, plt, sns):
    # 2. Correlação Linear (Matriz de Correlação das Variáveis Numéricas)
    _numeric_cols = ['MD', 'X', 'Y', 'Z', 'GR', 'TVT']
    _corr = df_horiz_all[_numeric_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(_corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Matriz de Correlação (Features Horizontais)")
    plt.tight_layout()

    _plot_corr = plt.gcf()
    _plot_corr
    return


@app.cell
def _(plt, random, train_wells):
    # 3. Trajetória 3D de um Poço Aleatório
    _well_id = random.choice(list(train_wells.keys()))
    _df_3d = train_wells[_well_id]['horizontal']

    _fig = plt.figure(figsize=(10, 8))
    _ax3d = _fig.add_subplot(111, projection='3d')

    # Plotar o caminho (X, Y, Z) colorido pelo Target (TVT)
    _scatter = _ax3d.scatter(_df_3d['X'], _df_3d['Y'], _df_3d['Z'], c=_df_3d['TVT'], cmap='Spectral', label='Trajetória')
    _ax3d.set_xlabel('Easting (X)')
    _ax3d.set_ylabel('Northing (Y)')
    _ax3d.set_zlabel('True Vertical Depth (Z)')
    _ax3d.set_title(f"Trajetória 3D Espacial do Poço: {_well_id}\n(Cores representam o TVT Real)")
    _ax3d.invert_zaxis() # Z em geologia geralmente cresce para baixo

    _cbar = _fig.colorbar(_scatter, ax=_ax3d, shrink=0.7)
    _cbar.set_label('TVT (Target)')

    _plot_3d = plt.gcf()
    _plot_3d
    return


if __name__ == "__main__":
    app.run()
