import os
import sys

print("🔍 Checking deployment readiness...\n")

checks = {
    "app/app.py": "Main Streamlit app",
    "app/model.pkl": "Trained ML model",
    "app/scaler.pkl": "Feature scaler",
    "app/columns.pkl": "Column names",
    "src/preprocess.py": "Preprocessing module",
    "src/train.py": "Training module",
    "src/predict.py": "Prediction module",
    "data/churn.csv": "Dataset",
    "requirements.txt": "Dependencies",
    ".gitignore": "Git ignore file",
    "README.md": "Project readme",
    ".streamlit/config.toml": "Streamlit config",
}

all_good = True
for filepath, description in checks.items():
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌ MISSING"
    print(f"{status} {description}: {filepath}")
    if not exists:
        all_good = False

print()
if all_good:
    print("🚀 ALL CHECKS PASSED — Ready to deploy!")
else:
    print("⚠️  Fix missing files before deploying.")
