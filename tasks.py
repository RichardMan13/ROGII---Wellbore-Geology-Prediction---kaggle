import shutil
import zipfile
from pathlib import Path

from invoke import task

from src import config

BASE_DIR = Path(__file__).resolve().parent

ALL_MODELS = list(config.MODEL_PARAMS.keys())
# Filtra modelos que não devem passar pelo Optuna tuning quando '--model=all' for chamado
TUNABLE_MODELS = [
    m for m in ALL_MODELS if m not in ["realmlp", "tabm", "tabpfn", "tabicl"]
]


@task
def clean(c):
    """Limpa arquivos temporários do Python, caches de Jupyter, logs e builds desnecessários."""
    print("Iniciando limpeza do workspace...")

    # 1. Limpar caches do Python (__pycache__, *.pyc, etc.)
    pycache_count = 0
    for p in BASE_DIR.rglob("__pycache__"):
        shutil.rmtree(p)
        pycache_count += 1

    pyc_count = 0
    for p in BASE_DIR.rglob("*.py[co]"):
        p.unlink()
        pyc_count += 1

    # 2. Limpar caches do Marimo
    marimo_count = 0
    for p in BASE_DIR.rglob("__marimo__"):
        shutil.rmtree(p)
        marimo_count += 1

    print(
        f"Removidos {pycache_count} diretórios __pycache__ e {pyc_count} arquivos .pyc/.pyo."
    )
    print(f"Removidos {marimo_count} diretórios de cache do Marimo.")


@task
def format(c):
    """Formata o código-fonte nas pastas src/ e notebooks/ utilizando o Ruff."""
    print("Formatando código com Ruff...")
    c.run("ruff format src/ notebooks/ tasks.py", warn=True)


@task
def lint(c):
    """Executa a verificação estática de código com o Ruff."""
    print("Executando análise estática com Ruff...")
    c.run("ruff check src/ notebooks/ tasks.py", warn=True)


@task(pre=[format, lint])
def check(c):
    """Executa a formatação e a verificação estática consecutivamente com o Ruff."""
    print("Verificação completa com Ruff concluída com sucesso!")


@task
def download_data(c, competition):
    """
    Baixa os dados da competição via KaggleHub e os salva na pasta data/raw/.
    Exemplo: inv download-data --competition=titanic
    """
    import kagglehub
    import shutil

    raw_dir = BASE_DIR / "data" / "raw"
    if raw_dir.exists():
        print(f"Limpando a pasta {raw_dir.name} antes do download...")
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Iniciando download dos dados da competicao '{competition}' via KaggleHub..."
    )
    # kagglehub automaticamente faz o download, lida com autenticação e retorna o caminho
    path = kagglehub.competition_download(competition, output_dir=str(raw_dir))
    print(f"Dados baixados com sucesso em: {path}")


@task(pre=[clean])
def train(c, model="all", mode="classifier"):
    """
    Roda o pipeline principal de treino cruzado (CV).
    Exemplo de uso: inv train (roda todos leves) ou inv train --model=tabpfn --mode=classifier
    """
    models_to_run = ALL_MODELS if model == "all" else [model]

    for m in models_to_run:
        print(f"Disparando pipeline de treino para o modelo: '{m}' no modo '{mode}'...")
        print("=" * 50 + "\n")
        c.run(f"python -m src.train --model {m} --mode {mode}", pty=False)


@task
def predict(c, model="all", mode="classifier"):
    """
    Roda o pipeline de inferência final com os modelos treinados.
    Exemplo de uso: inv predict (roda todos leves) ou inv predict --model=tabpfn
    """
    models_to_run = ALL_MODELS if model == "all" else [model]

    for m in models_to_run:
        print(
            f"Disparando pipeline de inferência para o modelo: '{m}' no modo '{mode}'..."
        )
        print("=" * 50 + "\n")
        c.run(f"python -m src.predict --model {m} --mode {mode}", pty=False)


@task
def tune(c, model="all"):
    """
    Roda a otimizacao de hiperparametros com Optuna.
    Exemplo de uso: inv tune (roda todos) ou inv tune --model=mlp
    """
    models_to_run = TUNABLE_MODELS if model == "all" else [model]

    for m in models_to_run:
        print("\n" + "=" * 50)
        print(f"Disparando otimizacao com Optuna para o modelo: '{m}'")
        print("=" * 50 + "\n")
        c.run(f"python -m src.tune --model {m}", pty=False)


@task
def blend(c, mode="classifier"):
    """
    Executa o blending das predicoes dos modelos da trindade.
    Exemplo de uso: inv blend --mode=regressor
    """
    print(f"Disparando blending de modelos no modo '{mode}'...")
    c.run(f"python -m src.blend --mode {mode}", pty=False)


# A task 'submit' foi removida. O envio da submissão deve ser feito manualmente pelo site do Kaggle.


@task
def notebook(c):
    """
    Inicia o editor Marimo na pasta notebooks/.
    """
    print("Iniciando Marimo...")
    c.run("marimo edit notebooks/", pty=False)
