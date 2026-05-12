"""Load and preprocess the churn dataset for model training and app use."""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess():
	"""Load churn data, preprocess it, and return train/test splits."""
	base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	data_path = os.path.join(base_dir, "data", "churn.csv")
	app_dir = os.path.join(base_dir, "app")
	os.makedirs(app_dir, exist_ok=True)

	# Step 1: Load CSV from data/churn.csv
	print("Step 1/9: Loading data...")
	df = pd.read_csv(data_path)

	# Step 2: Drop the customerID column
	print("Step 2/9: Dropping customerID...")
	if "customerID" in df.columns:
		df = df.drop(columns=["customerID"])

	# Step 3: Convert TotalCharges to float after handling blanks
	print("Step 3/9: Converting TotalCharges to float...")
	if "TotalCharges" in df.columns:
		df["TotalCharges"] = df["TotalCharges"].replace("", np.nan)
		df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
		df = df.dropna(subset=["TotalCharges"])

	# Step 4: Encode the Churn column (Yes=1, No=0)
	print("Step 4/9: Encoding Churn...")
	if "Churn" in df.columns:
		df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

	# Step 5: One-hot encode all object/categorical columns
	print("Step 5/9: One-hot encoding categoricals...")
	categorical_cols = df.select_dtypes(include=["object"]).columns
	df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

	# Step 6: Apply StandardScaler only to selected numeric columns
	print("Step 6/9: Scaling numeric features...")
	scale_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
	scaler = StandardScaler()
	existing_scale_cols = [col for col in scale_cols if col in df.columns]
	if existing_scale_cols:
		df[existing_scale_cols] = scaler.fit_transform(df[existing_scale_cols])

	# Step 7: Split into train/test with stratification
	print("Step 7/9: Splitting train/test...")
	if "Churn" not in df.columns:
		raise ValueError("Churn column is missing after preprocessing.")
	X = df.drop(columns=["Churn"])
	y = df["Churn"]
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		stratify=y,
		random_state=42,
	)

	# Step 8: Return all 4 variables from load_and_preprocess()
	print("Step 8/9: Preparing return values...")

	# Step 9: Save scaler and column names for the app
	print("Step 9/9: Saving scaler and column names...")
	joblib.dump(scaler, os.path.join(app_dir, "scaler.pkl"))
	joblib.dump(list(X.columns), os.path.join(app_dir, "columns.pkl"))

	return X_train, X_test, y_train, y_test

