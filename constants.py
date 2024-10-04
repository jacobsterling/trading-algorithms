from pathlib import Path
import os

# Base project directory
PROJECT_ROOT = Path(__file__).parent


# set cwd to project root inside jupyter notebook
def set_cwd_to_project_root():
    os.chdir(PROJECT_ROOT)


# Data directories
DATA_DIR = PROJECT_ROOT / "data"
BACKTEST_RESULTS_DIR = PROJECT_ROOT / "backtest_results"

# Model directories
MODEL_DIR = PROJECT_ROOT / "freqaimodels"

# Config files
CONFIG_DIR = PROJECT_ROOT / "config.json"
CONFIGURATION_DIR = PROJECT_ROOT / "configuration"


# Results and output
TRADES_DB_FILE = PROJECT_ROOT / "tradesv3.sqlite"

LOGS_DIR = PROJECT_ROOT / "logs"

DB_URL = "sqlite:////freqtrade/tradesv3.sqlite"


def get_model_path(model_name: str) -> Path:
    """Get the full path for a trained model."""
    return MODEL_DIR / f"{model_name}.pkl"
