"""Train and save a Logistic Regression churn model."""

# Import standard library tools for file paths.
import os
import sys

# Import third-party tools for saving models and training.
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	roc_auc_score,
)

# Add project root to sys.path so src imports work when run as a script.
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
	sys.path.insert(0, base_dir)

# Import the preprocessing function from our project.
from src.preprocess import load_and_preprocess


# Define the main training routine.
def main():
	# Load and preprocess the dataset, then get train/test splits.
	X_train, X_test, y_train, y_test = load_and_preprocess()

	# Create a Logistic Regression model with the required settings.
	model = LogisticRegression(
		solver="lbfgs",
		max_iter=1000,
		class_weight="balanced",
		random_state=42,
	)

	# Train the model on the training data.
	model.fit(X_train, y_train)

	# Make predictions on the test data.
	y_pred = model.predict(X_test)

	# Get probability scores for ROC-AUC evaluation.
	y_proba = model.predict_proba(X_test)[:, 1]

	# Calculate and print the accuracy score.
	print("Accuracy:", accuracy_score(y_test, y_pred))

	# Calculate and print the ROC-AUC score.
	print("ROC-AUC:", roc_auc_score(y_test, y_proba))

	# Print the full classification report.
	print("\nClassification Report:\n", classification_report(y_test, y_pred))

	# Print the confusion matrix.
	print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

	# Save the trained model to the app folder for later use.
	app_dir = os.path.join(
		os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
		"app",
	)
	os.makedirs(app_dir, exist_ok=True)
	joblib.dump(model, os.path.join(app_dir, "model.pkl"))

	# Print a success message for the user.
	print("✅ Model trained and saved successfully!")


# Run the main function when this file is executed directly.
if __name__ == "__main__":
	main()

