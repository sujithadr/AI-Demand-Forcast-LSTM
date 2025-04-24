import os

# Define project folder structure
folders = [
    "demand_forecasting_lstm/data/raw",
    "demand_forecasting_lstm/data/processed",
    "demand_forecasting_lstm/data/external",
    "demand_forecasting_lstm/notebooks",
    "demand_forecasting_lstm/src",
    "demand_forecasting_lstm/models",
    "demand_forecasting_lstm/outputs/figures",
    "demand_forecasting_lstm/outputs/results",
    "demand_forecasting_lstm/app"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")

# Create placeholder files
files = {
    "demand_forecasting_lstm/README.md": "# Demand Forecasting with LSTM\n\nProject overview.",
    "demand_forecasting_lstm/requirements.txt": "# Add Python dependencies here\npandas\nnumpy\nscikit-learn\ntensorflow\nmatplotlib\nfastapi\nuvicorn",
    "demand_forecasting_lstm/.gitignore": "*.pyc\n__pycache__/\nmodels/\noutputs/\n",
    "demand_forecasting_lstm/src/__init__.py": "",
    "demand_forecasting_lstm/src/data_preprocessing.py": "# Data preprocessing functions",
    "demand_forecasting_lstm/src/model.py": "# LSTM model definition",
    "demand_forecasting_lstm/src/predict.py": "# Prediction and evaluation functions",
    "demand_forecasting_lstm/src/utils.py": "# Utility functions",
    "demand_forecasting_lstm/app/main.py": "# FastAPI application"
}

# Create files with initial content
for file_path, content in files.items():
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"Created file: {file_path}")
