# 📊 Customer Churn Prediction Dashboard

An enterprise-grade ML dashboard that predicts customer churn 
in the telecom industry using Logistic Regression.

## 🚀 Live Demo
[Click here to view the app](YOUR_STREAMLIT_URL)

## 📌 Features
- Real-time churn risk prediction
- Interactive customer risk dashboard  
- Retention strategy simulator
- Similar customer matching
- Feature importance visualization

## 🛠️ Run Locally
```
pip install -r requirements.txt
python src/train.py
streamlit run app/app.py
```

## 📁 Project Structure
```
├── app/          → Streamlit dashboard + model files
├── src/          → ML pipeline (preprocess, train, predict)
├── data/         → Dataset (churn.csv)
└── requirements.txt
```

## 🧰 Tech Stack
Python | Scikit-learn | Streamlit | Plotly | Pandas | Joblib
