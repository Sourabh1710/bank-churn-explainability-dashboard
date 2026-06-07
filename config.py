# config.py
import os

DATA_PATH = "Churn_Modelling.csv"
MODELS_DIR = "models"
PLOTS_DIR = "outputs"

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)