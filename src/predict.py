"""Load a saved churn model and run predictions."""

from __future__ import annotations

import os

import joblib
import pandas as pd


_CACHED_ARTIFACTS = None


def set_cached_artifacts(model, scaler, columns):
	"""Set preloaded artifacts to avoid reloading from disk."""
	global _CACHED_ARTIFACTS
	_CACHED_ARTIFACTS = (model, scaler, columns)


def _load_artifacts(base_dir):
	"""Load model artifacts from disk or return cached versions."""
	global _CACHED_ARTIFACTS
	if _CACHED_ARTIFACTS is not None:
		return _CACHED_ARTIFACTS

	model_path = os.path.join(base_dir, "app", "model.pkl")
	scaler_path = os.path.join(base_dir, "app", "scaler.pkl")
	columns_path = os.path.join(base_dir, "app", "columns.pkl")

	model = joblib.load(model_path)
	scaler = joblib.load(scaler_path)
	columns = joblib.load(columns_path)

	_CACHED_ARTIFACTS = (model, scaler, columns)
	return _CACHED_ARTIFACTS


def predict_churn(input_dict):
	"""Return (prediction, probability) for a single raw input dict.

	Args:
		input_dict (dict): Raw user inputs, one example at a time.

	Returns:
		tuple: (prediction, probability) where prediction is 0/1 and
		probability is a float between 0 and 1.
	"""
	try:
		if not isinstance(input_dict, dict):
			raise ValueError("input_dict must be a Python dictionary.")

		base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		model, scaler, columns = _load_artifacts(base_dir)

		df = pd.DataFrame([input_dict])
		df = pd.get_dummies(df, drop_first=True)

		df = df.reindex(columns=columns, fill_value=0)

		scale_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
		existing_scale_cols = [col for col in scale_cols if col in df.columns]
		if existing_scale_cols:
			df[existing_scale_cols] = scaler.transform(df[existing_scale_cols])

		prediction = int(model.predict(df)[0])
		probability = float(model.predict_proba(df)[:, 1][0])

		return prediction, probability
	except Exception as exc:
		print(f"Prediction error: {exc}")
		return 0, 0.0

