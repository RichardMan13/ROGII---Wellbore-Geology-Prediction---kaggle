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

    # 2. Limpar checkpoints de Jupyter Notebooks
    jupyter_count = 0
    for p in BASE_DIR.rglob(".ipynb_checkpoints"):
        shutil.rmtree(p)
        jupyter_count += 1

    print(
        f"Removidos {pycache_count} diretórios __pycache__ e {pyc_count} arquivos .pyc/.pyo."
    )
    print(f"Removidos {jupyter_count} diretórios de checkpoints do Jupyter.")


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
    Baixa os dados da competição via API do Kaggle e descompacta na pasta data/raw/.
    Exemplo: inv download-data --competition=titanic
    """
    raw_dir = BASE_DIR / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Iniciando download dos dados da competicao '{competition}' via Kaggle API..."
    )
    c.run(f"kaggle competitions download -c {competition} -p {raw_dir}")

    zip_file = raw_dir / f"{competition}.zip"
    if zip_file.exists():
        print(f"Descompactando {zip_file.name} em {raw_dir}...")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        zip_file.unlink()
        print("Dados baixados e descompactados com sucesso!")
    else:
        # Se a API baixou os arquivos CSV ou outros tipos diretamente sem zipar
        print("Download concluido! Nenhum arquivo ZIP principal precisou ser extraido.")


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


@task
def submit(c, competition, file="submissions/submission.csv", message="My submission"):
    """
    Envia uma submissao para a competicao do Kaggle.
    Exemplo de uso: inv submit --competition=nome-da-competicao --file=submissions/submission.csv --message="Minha submissao"
    """
    file_path = BASE_DIR / file
    if not file_path.exists():
        print(f"[ERRO] O arquivo {file_path} não existe!")
        return

    print(f"Enviando arquivo {file_path.name} para a competicao '{competition}'...")
    c.run(
        f'kaggle competitions submit -c {competition} -f {file_path} -m "{message}"',
        pty=False,
    )
