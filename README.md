# 📊 Customer Churn Prediction Dashboard

A Streamlit dashboard that predicts telecom customer churn using a Logistic
Regression model trained on the IBM Telco Customer Churn dataset. Upload any
customer CSV and get per-customer churn risk scores, the top factors driving
each prediction, and visual breakdowns of what's driving churn across your
whole customer base.

## 🚀 Live Demo
[Try it here](https://customer-churn-prediction-hnd2ifkefunmc4uyvzhmyb.streamlit.app/) 

## 📌 Features

- Interactive churn predictor — enter a customer's details and get a live risk score
- Upload your own CSV of customers and get predictions for every row
- Feature-importance chart showing what's driving each prediction
- Risk segmentation (High / Medium / Low) with per-segment stats
- "Similar customers" lookup — compares a customer against others with a similar profile
- Churn-by-contract-type, tenure heatmap, and other exploratory charts

## 🛠️ Run Locally

```bash
git clone https://github.com/saksham-AI500/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
python src/train.py       # trains the model and saves it to app/
streamlit run app/app.py  # launches the dashboard
```

The app will open at `http://localhost:8501`.

## 📁 Project Structure

```
├── app/          → Streamlit dashboard (app.py) + saved model files
├── src/          → ML pipeline (preprocess.py, train.py, predict.py)
├── data/         → Dataset (churn.csv)
└── requirements.txt
```

## 🧰 Tech Stack

Python · Scikit-learn · Streamlit · Plotly · Pandas · Joblib

🔭 **Future Scope**

**Model improvements**: Compare Logistic Regression against Random Forest, XGBoost, and LightGBM to see if accuracy/ROC-AUC can be improved

**Explainability**: Add SHAP values for more rigorous, per-customer prediction explanations (currently uses raw model coefficients)

**Automated testing**: Add unit tests for preprocess.py and predict.py so changes don't silently break the pipeline

**CI/CD**: Add a GitHub Actions workflow to run tests automatically on every push

**Retention tab**: Build out the "Retention" and "Reports" tabs (currently placeholders) — e.g. a retention-offer simulator showing how discounts/upgrades change predicted churn risk

**Multi-model support**: Let users pick which trained model to use for predictions, instead of one fixed model

**Better data handling**: Support datasets beyond the Telco schema — auto-detect column meaning instead of assuming fixed column names

**Deployment**: Add Docker support for easier self-hosting

**Authentication**: If ever used with real customer data rather than practice datasets, add login/auth and data-handling safeguards

## 🤝 Contributing

Issues and pull requests are welcome — this is an actively maintained
learning project and feedback helps it improve.

## 📝 License

MIT
