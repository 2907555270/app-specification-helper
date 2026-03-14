# nl2sql_v3

A CLI tool for table retrieval in nl2sql applications.

## Installation

```bash
pip install -e .
```

## Usage

```bash
python main.py build-index
python main.py recall "card" --top-k 5
python main.py evaluate
```

## Configuration

Edit `config.yaml` to configure services.
