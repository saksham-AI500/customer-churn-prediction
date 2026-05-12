"""Streamlit app entry point for churn prediction."""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import f1_score, recall_score, roc_auc_score

try:
    from src.predict import predict_churn
except Exception:
    predict_churn = None


st.set_page_config(
    layout="wide",
    page_title="AuraAnalytics - Churn Prediction",
    page_icon="📊",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "app", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "app", "scaler.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "app", "columns.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "churn.csv")


@st.cache_resource
def load_artifacts():
    """Load model artifacts once and cache them for the app."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        columns = joblib.load(COLUMNS_PATH)
        return model, scaler, columns, True, ""
    except Exception as exc:
        return None, None, None, False, str(exc)


@st.cache_data
def load_data():
    """Load and preprocess the churn data for the dashboard."""
    df = pd.read_csv(DATA_PATH)

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)

    # Encode Churn as 0/1
    df["Churn_Binary"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Create Risk Score from actual model predictions
    model, scaler, columns, model_loaded, _ = load_artifacts()

    if model_loaded:
        df_encoded = pd.get_dummies(df.drop(["customerID", "Churn"], axis=1), drop_first=True)
        df_encoded = df_encoded.reindex(columns=columns, fill_value=0)

        scale_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        existing_scale_cols = [col for col in scale_cols if col in df_encoded.columns]
        if existing_scale_cols:
            df_encoded[existing_scale_cols] = scaler.transform(df_encoded[existing_scale_cols])

        df["Risk Score"] = (model.predict_proba(df_encoded)[:, 1] * 100).round(1)
    else:
        tenure_norm = df["tenure"] / max(df["tenure"].max(), 1)
        charges_norm = df["MonthlyCharges"] / max(df["MonthlyCharges"].max(), 1)
        df["Risk Score"] = ((1 - tenure_norm) * 60 + charges_norm * 40).round(1)

    # Create Risk Label
    df["Risk Label"] = df["Risk Score"].apply(
        lambda x: "High" if x >= 70 else ("Medium" if x >= 40 else "Low")
    )

    # Create Activity Score from tenure + MonthlyCharges (normalized 0-100)
    df["Activity Score"] = (
        (df["tenure"] / df["tenure"].max() * 50)
        + (df["MonthlyCharges"] / df["MonthlyCharges"].max() * 50)
    ).round(0).astype(int)

    # Create display Customer ID like #00001
    df["Customer ID"] = ["#" + str(i).zfill(5) for i in range(1, len(df) + 1)]

    # Format tenure as Xmo
    df["Tenure Display"] = df["tenure"].astype(int).astype(str) + "mo"

    # Plan from Contract column
    df["Plan"] = df["Contract"].map(
        {
            "Month-to-month": "Basic",
            "One year": "Gold",
            "Two year": "Platinum",
        }
    )

    return df


@st.cache_data
def predict_new_csv(df):
    """Predict churn for any uploaded CSV without crashing the app."""
    try:
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            columns = joblib.load(COLUMNS_PATH)
        except Exception:
            st.error("Please run src/train.py first")
            df["Risk Score"] = np.nan
            df["Prediction"] = "Unknown"
            df["Risk Level"] = "Unknown"
            return df

        df_encoded = pd.get_dummies(df, drop_first=True)
        available_cols = [c for c in columns if c in df_encoded.columns]
        if len(available_cols) < len(columns):
            st.warning("Some columns missing — using available ones")

        df_encoded = df_encoded.reindex(columns=columns, fill_value=0)

        scale_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        existing_scale = [c for c in scale_cols if c in df_encoded.columns]
        if existing_scale:
            df_encoded[existing_scale] = scaler.transform(df_encoded[existing_scale])

        probabilities = model.predict_proba(df_encoded)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        df["Risk Score"] = (probabilities * 100).round(1)
        df["Prediction"] = ["Will Churn" if p == 1 else "Will Stay" for p in predictions]
        df["Risk Level"] = df["Risk Score"].apply(
            lambda x: "High" if x >= 70 else ("Medium" if x >= 40 else "Low")
        )
        return df
    except Exception as exc:
        st.error("Error: " + str(exc))
        df["Risk Score"] = np.nan
        df["Prediction"] = "Unknown"
        df["Risk Level"] = "Unknown"
        return df


def render_predictor_section(df, model, scaler, columns):
    """Render the interactive churn predictor UI and results."""
    st.markdown("## 🔍 Customer Churn Predictor")
    st.markdown("Select customer parameters below to predict churn probability")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 👤 Personal Info")
        gender = st.selectbox("Gender", ["Male", "Female"], key="pred_gender")
        senior = st.selectbox("Senior Citizen", ["No", "Yes"], key="pred_senior")
        partner = st.selectbox("Has Partner", ["Yes", "No"], key="pred_partner")
        dependents = st.selectbox("Has Dependents", ["Yes", "No"], key="pred_dependents")
        tenure = st.slider("Months with Company", 0, 72, 12, key="pred_tenure")
        monthly = st.slider("Monthly Bill ($)", 0.0, 120.0, 65.0, step=0.5, key="pred_monthly")
        total = st.slider("Total Paid ($)", 0.0, 9000.0, 1500.0, step=10.0, key="pred_total")

    with col2:
        st.markdown("### 📡 Services")
        phone = st.selectbox("Phone Service", ["Yes", "No"], key="pred_phone")
        multi = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="pred_multi")
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="pred_internet")
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key="pred_security")
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="pred_backup")
        device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="pred_device")
        tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="pred_tech")
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="pred_tv")
        movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="pred_movies")

    with col3:
        st.markdown("### 💳 Account Info")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="pred_contract")
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key="pred_paperless")
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            key="pred_payment",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='button-primary'>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict Churn Risk", use_container_width=True, type="primary", key="pred_btn")
    st.markdown("</div>", unsafe_allow_html=True)

    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None

    if predict_btn:
        input_data = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multi,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

        try:
            if predict_churn is None or model is None or scaler is None or columns is None:
                raise ValueError("Model files are missing. Please run src/train.py first.")

            with st.spinner("Analyzing customer data..."):
                prediction, probability = predict_churn(input_data)

            st.session_state.prediction_result = {
                "prediction": prediction,
                "probability": probability,
                "input": input_data,
            }
        except Exception as exc:
            st.error("Error: " + str(exc))

    if st.session_state.prediction_result is not None:
        result = st.session_state.prediction_result
        risk_pct = round(result["probability"] * 100, 1)

        if risk_pct >= 70:
            risk_color = "#ff4b6e"
        elif risk_pct >= 40:
            risk_color = "#ffaa00"
        else:
            risk_color = "#4f8ef7"

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        res_col_1, res_col_2, res_col_3 = st.columns(3)

        with res_col_1:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_pct,
                    title={"text": "Churn Risk Score"},
                    number={"suffix": "%", "font": {"color": risk_color}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": risk_color},
                        "bgcolor": "#1e2130",
                    },
                )
            )
            gauge.update_layout(
                paper_bgcolor="#1e2130",
                plot_bgcolor="#1e2130",
                font=dict(color="white"),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(gauge, use_container_width=True)

        with res_col_2:
            if risk_pct >= 70:
                st.error("🚨 HIGH RISK — This customer is likely to churn!")
                st.metric("Churn Probability", f"{risk_pct}%", delta="High Risk")
            elif risk_pct >= 40:
                st.warning("⚠️ MEDIUM RISK — Monitor this customer closely")
                st.metric("Churn Probability", f"{risk_pct}%", delta="Medium Risk")
            else:
                st.success("✅ LOW RISK — Customer likely to stay")
                st.metric("Churn Probability", f"{risk_pct}%", delta="Low Risk")

            st.markdown(
                f"""
                **Contract:** {contract}  
                **Tenure:** {tenure} months  
                **Monthly Bill:** ${monthly}  
                **Internet:** {internet}
                """
            )

        with res_col_3:
            st.markdown("### ⚠️ Top Factors Driving This Prediction")
            try:
                input_df = pd.DataFrame([result["input"]])
                input_encoded = pd.get_dummies(input_df, drop_first=True)
                input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

                contributions = np.abs(model.coef_[0] * input_encoded.values[0])
                top5_idx = contributions.argsort()[-5:][::-1]
                top5_features = [columns[i] for i in top5_idx]
                top5_values = [contributions[i] for i in top5_idx]

                clean_features = [name.replace("_", " ").title() for name in top5_features]

                bar_color = "#ff4b6e" if risk_pct >= 70 else ("#ffaa00" if risk_pct >= 40 else "#4f8ef7")
                fig_top = go.Figure(
                    go.Bar(
                        x=top5_values,
                        y=clean_features,
                        orientation="h",
                        marker_color=bar_color,
                        text=[f"{v:.2f}" for v in top5_values],
                        textposition="auto",
                    )
                )
                fig_top.update_layout(
                    paper_bgcolor="#1e2130",
                    plot_bgcolor="#1e2130",
                    font=dict(color="white", size=12),
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_title="Contribution Score",
                )
                st.plotly_chart(fig_top, use_container_width=True)
            except Exception as exc:
                st.error("Error: " + str(exc))

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 👥 Similar Customers in Database")
        st.markdown("Customers in our database with similar profile and their churn status:")

        try:
            filtered_similar = df[
                (df["tenure"].between(tenure - 6, tenure + 6))
                & (df["MonthlyCharges"].between(monthly - 15, monthly + 15))
                & (df["Contract"] == contract)
                & (df["InternetService"] == internet)
            ].copy()

            if len(filtered_similar) == 0:
                st.info(
                    "No similar customers found with these exact parameters. "
                    "Try adjusting tenure or monthly charges."
                )
            else:
                filtered_similar["distance"] = (filtered_similar["Risk Score"] - risk_pct).abs()
                filtered_similar = filtered_similar.sort_values("distance").head(10)

                sim_table = filtered_similar[[
                    "Customer ID",
                    "Plan",
                    "Tenure Display",
                    "MonthlyCharges",
                    "Risk Score",
                    "Churn",
                ]].copy()
                sim_table.columns = [
                    "Customer ID",
                    "Plan",
                    "Tenure",
                    "Monthly Bill",
                    "Risk Score",
                    "Actual Churn",
                ]

                def style_similar(row):
                    styles = [""] * len(row)
                    return styles

                def style_similar_risk(row):
                    risk_val = float(row["Risk Score"])
                    if risk_val >= 70:
                        color = "#ff4b6e"
                        bg = "rgba(255, 75, 110, 0.2)"
                    elif risk_val >= 40:
                        color = "#ffaa00"
                        bg = "rgba(255, 170, 0, 0.2)"
                    else:
                        color = "#00d68f"
                        bg = "rgba(0, 214, 143, 0.2)"

                    churn_val = row["Actual Churn"]
                    churn_color = "#ff4b6e" if churn_val == "Yes" else "#00d68f"
                    churn_bg = "rgba(255, 75, 110, 0.2)" if churn_val == "Yes" else "rgba(0, 214, 143, 0.2)"

                    return [
                        "",
                        "",
                        "",
                        "",
                        f"background-color: {bg}; color: {color}; font-weight: 600;",
                        f"background-color: {churn_bg}; color: {churn_color}; font-weight: 600;",
                    ]

                styled_sim = (
                    sim_table.style
                    .apply(style_similar, axis=1)
                    .apply(style_similar_risk, axis=1)
                )

                st.dataframe(
                    styled_sim,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Customer ID": st.column_config.TextColumn(width=120),
                        "Plan": st.column_config.TextColumn(width=100),
                        "Tenure": st.column_config.TextColumn(width=90),
                        "Monthly Bill": st.column_config.NumberColumn(width=120),
                        "Risk Score": st.column_config.NumberColumn(width=110),
                        "Actual Churn": st.column_config.TextColumn(width=110),
                    },
                )

                sim_col_1, sim_col_2, sim_col_3 = st.columns(3)
                with sim_col_1:
                    st.metric("Similar Customers Found", len(filtered_similar))
                with sim_col_2:
                    churned_count = int((filtered_similar["Churn"] == "Yes").sum())
                    st.metric("Actually Churned", churned_count)
                with sim_col_3:
                    churn_rate = round(churned_count / max(len(filtered_similar), 1) * 100)
                    st.metric("Churn Rate in Group", f"{churn_rate}%")
        except Exception as exc:
            st.error("Error: " + str(exc))

        with st.expander("📖 How does the model decide? Click to understand each parameter"):
            explain_df = pd.DataFrame(
                [
                    ["Contract Type", "Strongest predictor", "Month-to-month"],
                    ["Tenure", "Loyalty indicator", "Less than 12 months"],
                    ["Monthly Charges", "Cost sensitivity", "Above $65/month"],
                    ["Internet Service", "Service quality", "Fiber optic"],
                    ["Tech Support", "Support satisfaction", "No support"],
                    ["Online Security", "Value perception", "No security"],
                    ["Payment Method", "Engagement level", "Electronic check"],
                    ["Total Charges", "Overall investment", "Low total despite long tenure"],
                    ["Senior Citizen", "Demographics", "Yes"],
                    ["Paperless Billing", "Digital engagement", "Yes"],
                ],
                columns=["Parameter", "Why It Matters", "High Churn Risk When"],
            )
            st.dataframe(explain_df, use_container_width=True, hide_index=True)
            st.info(
                "These insights are derived from the Logistic Regression "
                "model coefficients trained on IBM Telco Customer Churn data."
            )


def apply_plotly_theme(fig, title=None):
    """Apply consistent dark theme styling to plotly charts."""
    fig.update_layout(
        title=title,
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1e2130",
        font=dict(color="white", size=12),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    :root {
        --bg: #0f1117;
        --sidebar: #1a1d2e;
        --card: #1e2130;
        --card-border: #2d3250;
        --accent: #4f8ef7;
        --risk-high: #ff4b6e;
        --risk-med: #ffaa00;
        --risk-low: #00d68f;
        --text: #e5e7eb;
        --muted: #9ca3af;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }

    .stApp {
        background-color: var(--bg);
    }

    section[data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid #20243a;
    }

    section[data-testid="stSidebar"] * {
        color: var(--text);
    }

    .block-container {
        padding: 0.5rem 1.5rem 1.5rem 1.5rem;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0c0f17;
        border: 1px solid #1f2335;
        border-radius: 14px;
        padding: 0.75rem 1.5rem;
        margin-bottom: 1rem;
    }

    .nav-left {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        font-weight: 600;
    }

    .nav-logo {
        font-size: 1.1rem;
        color: var(--accent);
    }

    .nav-right {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .avatar {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: linear-gradient(135deg, #4f8ef7, #7dd3fc);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        color: #0f1117;
    }

    .card {
        background: var(--card);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }

    .chip {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .chip-high { background: rgba(255, 75, 110, 0.2); color: var(--risk-high); }
    .chip-med { background: rgba(255, 170, 0, 0.2); color: var(--risk-med); }
    .chip-low { background: rgba(0, 214, 143, 0.2); color: var(--risk-low); }

    .button-primary button {
        background: linear-gradient(135deg, #4f8ef7, #7aa9ff);
        border-radius: 999px;
        border: none;
        color: white;
        font-weight: 600;
    }

    .metric-large {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text);
    }

    .sidebar-card {
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="navbar">
        <div class="nav-left">
            <div class="nav-logo">AuraAnalytics</div>
        </div>
        <div class="nav-right">
            <span>🔔</span>
            <div class="avatar">SJ</div>
            <span>Sarah J.</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


try:
    with st.spinner("Loading data..."):
        df = load_data()
        model, scaler, columns, model_loaded, model_error = load_artifacts()
        if not model_loaded:
            st.warning(f"Model not loaded: {model_error}. Running in demo mode.")
except Exception as exc:
    st.error("Error: " + str(exc))
    st.stop()


tabs = st.tabs(["📊 Predictions", "📈 Analysis", "🔒 Retention", "📋 Reports"])


with tabs[0]:
    try:
        high_risk_df = df[df["Risk Label"] == "High"]
        medium_risk_df = df[df["Risk Label"] == "Medium"]
        low_risk_df = df[df["Risk Label"] == "Low"]

        total = len(df)
        if total == 0:
            st.error("Error: Dataset is empty after preprocessing.")
            st.stop()

        high_pct = round(len(high_risk_df) / total * 100)
        medium_pct = round(len(medium_risk_df) / total * 100)
        low_pct = round(len(low_risk_df) / total * 100)

        def segment_stats(segment_df):
            avg_tenure = round(segment_df["tenure"].mean(), 1) if len(segment_df) else 0
            avg_monthly = round(segment_df["MonthlyCharges"].mean(), 1) if len(segment_df) else 0
            churn_rate = round(segment_df["Churn_Binary"].mean() * 100) if len(segment_df) else 0
            count = len(segment_df)
            return avg_tenure, avg_monthly, churn_rate, count

        high_stats = segment_stats(high_risk_df)
        medium_stats = segment_stats(medium_risk_df)
        low_stats = segment_stats(low_risk_df)

        with st.sidebar:
            st.markdown("<div class='card'><strong>Segmented Customers</strong></div>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="sidebar-card" style="background: #3d1a1a; border: 1px solid rgba(255, 75, 110, 0.35);">
                    <strong>HIGH RISK ({high_pct}%)</strong><br>
                    • Avg Tenure: {high_stats[0]}mo<br>
                    • Avg Monthly: ${high_stats[1]}<br>
                    • Churn Rate: {high_stats[2]}%<br>
                    • Count: {high_stats[3]} customers
                </div>
                <div class="sidebar-card" style="background: #3d2e0a; border: 1px solid rgba(255, 170, 0, 0.35);">
                    <strong>MEDIUM RISK ({medium_pct}%)</strong><br>
                    • Avg Tenure: {medium_stats[0]}mo<br>
                    • Avg Monthly: ${medium_stats[1]}<br>
                    • Churn Rate: {medium_stats[2]}%<br>
                    • Count: {medium_stats[3]} customers
                </div>
                <div class="sidebar-card" style="background: #0a2e1a; border: 1px solid rgba(0, 214, 143, 0.35);">
                    <strong>LOW RISK ({low_pct}%)</strong><br>
                    • Avg Tenure: {low_stats[0]}mo<br>
                    • Avg Monthly: ${low_stats[1]}<br>
                    • Churn Rate: {low_stats[2]}%<br>
                    • Count: {low_stats[3]} customers
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div class='card'><strong>Key Feature</strong></div>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            col_c, col_d = st.columns(2)
            col_a.button("📊 Usage")
            col_b.button("💬 Engagement")
            col_c.button("💳 Billing")
            col_d.button("👥 Demographics")

        st.markdown(
            """
            <div style='background:#1e2130; border:2px dashed #4f8ef7; 
            border-radius:16px; padding:40px; text-align:center; margin-bottom:30px'>
                <h2 style='color:#4f8ef7'>📂 Upload Customer CSV File</h2>
                <p style='color:#aaaaaa'>Upload any telecom customer CSV file.
                The model will auto-detect columns and predict churn for every customer.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Drop your CSV file here",
            type=["csv"],
            help="CSV must contain customer data. The model will match available columns automatically.",
        )

        if "uploaded_df" not in st.session_state:
            st.session_state.uploaded_df = None
        if "predicted_df" not in st.session_state:
            st.session_state.predicted_df = None
        if "uploaded_name" not in st.session_state:
            st.session_state.uploaded_name = None

        if uploaded_file is not None:
            try:
                if st.session_state.uploaded_name != uploaded_file.name or st.session_state.predicted_df is None:
                    st.session_state.uploaded_df = pd.read_csv(uploaded_file)
                    st.session_state.uploaded_name = uploaded_file.name
                    with st.spinner("🔍 Analyzing your customer data..."):
                        st.session_state.predicted_df = predict_new_csv(
                            st.session_state.uploaded_df.copy()
                        )
            except Exception as exc:
                st.error("Error: " + str(exc))

        if st.session_state.uploaded_df is not None:
            uploaded_df = st.session_state.uploaded_df
            st.success(
                f"✅ File uploaded successfully! Found {len(uploaded_df)} customers and {uploaded_df.shape[1]} columns."
            )

            tags_html = "".join(
                [
                    f"<span style='background:#1a2b4f; color:#4f8ef7; padding:6px 10px; border-radius:999px; font-size:12px;'>{col}</span>"
                    for col in uploaded_df.columns
                ]
            )
            st.markdown(
                f"<div style='display:flex; flex-wrap:wrap; gap:8px; margin-bottom:20px'>{tags_html}</div>",
                unsafe_allow_html=True,
            )

        if st.session_state.uploaded_df is None:
            info_col_1, info_col_2, info_col_3 = st.columns(3)
            with info_col_1:
                st.markdown(
                    """
                    <div style='background:#1e2130; border:1px solid #4f8ef7; border-radius:12px; padding:20px; text-align:center'>
                        <h3>📁 Step 1: Prepare CSV</h3>
                        <p>Your CSV should contain customer data with columns like tenure, MonthlyCharges, Contract, InternetService etc.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with info_col_2:
                st.markdown(
                    """
                    <div style='background:#1e2130; border:1px solid #4f8ef7; border-radius:12px; padding:20px; text-align:center'>
                        <h3>⬆️ Step 2: Upload File</h3>
                        <p>Click the upload area above or drag and drop your CSV file. The model supports any telecom customer dataset.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with info_col_3:
                st.markdown(
                    """
                    <div style='background:#1e2130; border:1px solid #4f8ef7; border-radius:12px; padding:20px; text-align:center'>
                        <h3>📊 Step 3: Get Predictions</h3>
                        <p>Instantly see churn predictions, risk scores, and visual analysis for every customer in your file.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("### 📋 Expected CSV Format (sample columns)")
            sample_df = pd.DataFrame(
                {
                    "customerID": ["0001-A", "0002-B", "0003-C"],
                    "tenure": [5, 24, 36],
                    "MonthlyCharges": [45.2, 78.5, 90.1],
                    "Contract": ["Month-to-month", "One year", "Two year"],
                    "InternetService": ["DSL", "Fiber optic", "DSL"],
                    "TotalCharges": [120.5, 1500.0, 3200.0],
                }
            )
            st.dataframe(sample_df, use_container_width=True)
        else:
            predicted_df = st.session_state.predicted_df
            if predicted_df is None:
                st.error("Error: Predictions could not be generated.")
            else:
                analysis_df = predicted_df.copy()
                if "Risk Score" not in analysis_df.columns:
                    analysis_df["Risk Score"] = 0.0
                if "Prediction" not in analysis_df.columns:
                    analysis_df["Prediction"] = np.where(analysis_df["Risk Score"] >= 50, "Will Churn", "Will Stay")
                if "Risk Level" not in analysis_df.columns:
                    analysis_df["Risk Level"] = analysis_df["Risk Score"].apply(
                        lambda x: "High" if x >= 70 else ("Medium" if x >= 40 else "Low")
                    )

                id_col = None
                for col in ["customerID", "CustomerID", "customer_id", "ID", "Name", "name", "Customer ID"]:
                    if col in analysis_df.columns:
                        id_col = col
                        break
                if id_col is None:
                    id_col = analysis_df.columns[0]

                total_customers = len(analysis_df)
                churn_count = int((analysis_df["Prediction"] == "Will Churn").sum())
                safe_count = total_customers - churn_count
                churn_pct = round((churn_count / max(total_customers, 1)) * 100, 1)
                safe_pct = round((safe_count / max(total_customers, 1)) * 100, 1)
                avg_risk = round(analysis_df["Risk Score"].mean(), 1)

                st.markdown("### 📊 Analysis Dashboard")
                kpi_col_1, kpi_col_2, kpi_col_3, kpi_col_4 = st.columns(4)

                def render_list_card(title, items, bg, border, color):
                    list_html = "".join(items)
                    st.markdown(
                        f"""
                        <div style='background:{bg}; border-radius:8px; padding:12px; max-height:200px; overflow-y:auto; border-left:3px solid {border}'>
                            <div style='font-weight:600; margin-bottom:6px; color:{color}'>{title}</div>
                            {list_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with kpi_col_1:
                    st.markdown(
                        f"""
                        <div style='background:#1e2130; border-radius:12px; padding:24px; border-left:4px solid #4f8ef7; text-align:center'>
                          <h1 style='color:#4f8ef7; margin:0'>👥 {total_customers}</h1>
                          <p style='color:#aaaaaa; margin:0'>Total Customers</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    total_ids = analysis_df[id_col].astype(str).tolist()
                    total_items = [f"<div style='color:#4f8ef7'>• {cid}</div>" for cid in total_ids[:20]]
                    if len(total_ids) > 20:
                        total_items.append(f"<div style='color:#aaaaaa'>...and {len(total_ids) - 20} more</div>")
                    render_list_card("👥 All Customers", total_items, "#1a2438", "#4f8ef7", "#4f8ef7")

                with kpi_col_2:
                    st.markdown(
                        f"""
                        <div style='background:#1e2130; border-radius:12px; padding:24px; border-left:4px solid #ff4b6e; text-align:center'>
                          <h1 style='color:#ff4b6e; margin:0'>🚨 {churn_count}</h1>
                          <p style='color:#aaaaaa; margin:0'>At Risk Customers</p>
                          <p style='color:#ff4b6e; margin:0'>{churn_pct}% of total</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    risk_df = analysis_df[analysis_df["Prediction"] == "Will Churn"]
                    risk_items = [
                        f"<div style='color:#ff4b6e'>• {row[id_col]} — {row['Risk Score']:.0f}% risk</div>"
                        for _, row in risk_df.head(20).iterrows()
                    ]
                    if len(risk_df) > 20:
                        risk_items.append(f"<div style='color:#aaaaaa'>...and {len(risk_df) - 20} more</div>")
                    render_list_card("🚨 At Risk Customers", risk_items, "#2d1a1a", "#ff4b6e", "#ff4b6e")

                with kpi_col_3:
                    st.markdown(
                        f"""
                        <div style='background:#1e2130; border-radius:12px; padding:24px; border-left:4px solid #00d68f; text-align:center'>
                          <h1 style='color:#00d68f; margin:0'>✅ {safe_count}</h1>
                          <p style='color:#aaaaaa; margin:0'>Safe Customers</p>
                          <p style='color:#00d68f; margin:0'>{safe_pct}% of total</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    safe_df = analysis_df[analysis_df["Prediction"] == "Will Stay"]
                    safe_items = [
                        f"<div style='color:#00d68f'>• {row[id_col]} — {row['Risk Score']:.0f}% risk</div>"
                        for _, row in safe_df.head(20).iterrows()
                    ]
                    if len(safe_df) > 20:
                        safe_items.append(f"<div style='color:#aaaaaa'>...and {len(safe_df) - 20} more</div>")
                    render_list_card("✅ Safe Customers", safe_items, "#1a2d1a", "#00d68f", "#00d68f")

                with kpi_col_4:
                    st.markdown(
                        f"""
                        <div style='background:#1e2130; border-radius:12px; padding:24px; border-left:4px solid #ffaa00; text-align:center'>
                          <h1 style='color:#ffaa00; margin:0'>📊 {avg_risk}%</h1>
                          <p style='color:#aaaaaa; margin:0'>Avg Risk Score</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    top_risk = analysis_df.sort_values("Risk Score", ascending=False).head(5)
                    top_items = [
                        f"<div style='color:#ffaa00'>• {row[id_col]} — {row['Risk Score']:.0f}%</div>"
                        for _, row in top_risk.iterrows()
                    ]
                    render_list_card("📊 Top Risk Customers", top_items, "#2d250f", "#ffaa00", "#ffaa00")

                st.markdown("### 🔬 Parameter Analysis — What's Driving Churn?")
                st.markdown(
                    "Analysis of every parameter in your uploaded CSV and how it affects churn prediction"
                )

                analysis_base = analysis_df.copy()
                prediction_cols = ["Risk Score", "Prediction", "Risk Level"]

                numerical_cols = analysis_base.select_dtypes(include=["int64", "float64"]).columns.tolist()
                categorical_cols = analysis_base.select_dtypes(include=["object"]).columns.tolist()

                for col in [id_col, *prediction_cols]:
                    if col in numerical_cols:
                        numerical_cols.remove(col)
                    if col in categorical_cols:
                        categorical_cols.remove(col)

                def valid_column(col):
                    missing_ratio = analysis_base[col].isna().mean()
                    unique_count = analysis_base[col].nunique(dropna=True)
                    return missing_ratio <= 0.5 and unique_count > 1

                numerical_cols = [col for col in numerical_cols if valid_column(col)]
                categorical_cols = [col for col in categorical_cols if valid_column(col)]

                for idx in range(0, len(numerical_cols), 2):
                    cols_pair = st.columns(2)
                    for c_idx, col in enumerate(numerical_cols[idx:idx + 2]):
                        with cols_pair[c_idx]:
                            try:
                                fig_box = px.box(
                                    analysis_base,
                                    x="Prediction",
                                    y=col,
                                    color="Prediction",
                                    color_discrete_map={"Will Churn": "#ff4b6e", "Will Stay": "#00d68f"},
                                    points=False,
                                )
                                fig_box.update_traces(boxmean=True)
                                apply_plotly_theme(fig_box, f"📊 {col} by Churn Prediction")
                                st.plotly_chart(fig_box, use_container_width=True)

                                churned_avg = analysis_base[analysis_base["Prediction"] == "Will Churn"][col].mean()
                                safe_avg = analysis_base[analysis_base["Prediction"] == "Will Stay"][col].mean()
                                diff_pct = 0 if safe_avg == 0 else ((churned_avg - safe_avg) / safe_avg) * 100

                                stat_col_1, stat_col_2, stat_col_3 = st.columns(3)
                                stat_col_1.metric("Churned Avg", f"{churned_avg:.2f}")
                                stat_col_2.metric("Safe Avg", f"{safe_avg:.2f}")
                                stat_col_3.metric("Difference", f"{diff_pct:.1f}%")

                                if diff_pct > 10:
                                    st.warning(f"⚠️ Higher {col} linked to more churn")
                                elif diff_pct < -10:
                                    st.info(f"ℹ️ Lower {col} linked to more churn")
                            except Exception as exc:
                                st.error("Error: " + str(exc))

                for idx in range(0, len(categorical_cols), 2):
                    cols_pair = st.columns(2)
                    for c_idx, col in enumerate(categorical_cols[idx:idx + 2]):
                        with cols_pair[c_idx]:
                            try:
                                counts = (
                                    analysis_base.groupby([col, "Prediction"]).size().reset_index(name="count")
                                )
                                fig_cat = px.bar(
                                    counts,
                                    x=col,
                                    y="count",
                                    color="Prediction",
                                    barmode="group",
                                    color_discrete_map={"Will Churn": "#ff4b6e", "Will Stay": "#00d68f"},
                                )
                                apply_plotly_theme(fig_cat, f"📋 {col} — Churn vs Stay")
                                st.plotly_chart(fig_cat, use_container_width=True)

                                churn_rates = (
                                    analysis_base.groupby(col)["Prediction"]
                                    .apply(lambda s: (s == "Will Churn").mean())
                                    .sort_values(ascending=False)
                                )
                                if not churn_rates.empty:
                                    most_churny = churn_rates.index[0]
                                    churn_rate_pct = churn_rates.iloc[0] * 100
                                    st.error(
                                        f"🔴 '{most_churny}' has highest churn rate in {col}: {churn_rate_pct:.1f}%"
                                    )
                            except Exception as exc:
                                st.error("Error: " + str(exc))

                impact_rows = []
                for col in analysis_base.columns:
                    if col in prediction_cols or col == id_col:
                        continue
                    if not valid_column(col):
                        continue

                    clean_name = col.replace("_", " ").title()
                    if col in numerical_cols:
                        churned_mean = analysis_base[analysis_base["Prediction"] == "Will Churn"][col].mean()
                        safe_mean = analysis_base[analysis_base["Prediction"] == "Will Stay"][col].mean()
                        overall_std = analysis_base[col].std() or 1
                        impact_score = abs(churned_mean - safe_mean) / overall_std * 10
                        risk_direction = "↑ Higher = More Risk" if churned_mean > safe_mean else "↓ Lower = More Risk"
                        impact_rows.append(
                            {
                                "Parameter": clean_name,
                                "Type": "Numerical",
                                "Churned Avg/Mode": f"{churned_mean:.2f}",
                                "Safe Avg/Mode": f"{safe_mean:.2f}",
                                "Impact Score": round(impact_score, 1),
                                "Risk Direction": risk_direction,
                            }
                        )
                    else:
                        churned_mode = (
                            analysis_base[analysis_base["Prediction"] == "Will Churn"][col]
                            .mode()
                            .iloc[0]
                            if not analysis_base[analysis_base["Prediction"] == "Will Churn"][col].mode().empty
                            else "N/A"
                        )
                        safe_mode = (
                            analysis_base[analysis_base["Prediction"] == "Will Stay"][col]
                            .mode()
                            .iloc[0]
                            if not analysis_base[analysis_base["Prediction"] == "Will Stay"][col].mode().empty
                            else "N/A"
                        )
                        churn_rate_by_cat = (
                            analysis_base.groupby(col)["Prediction"]
                            .apply(lambda s: (s == "Will Churn").mean())
                        )
                        impact_score = churn_rate_by_cat.max() * 10 if not churn_rate_by_cat.empty else 0
                        impact_rows.append(
                            {
                                "Parameter": clean_name,
                                "Type": "Categorical",
                                "Churned Avg/Mode": str(churned_mode),
                                "Safe Avg/Mode": str(safe_mode),
                                "Impact Score": round(impact_score, 1),
                                "Risk Direction": "Varies by category",
                            }
                        )

                impact_df = pd.DataFrame(impact_rows)
                if not impact_df.empty:
                    impact_df = impact_df.sort_values("Impact Score", ascending=False)

                    def impact_color(val):
                        if val >= 7:
                            return "color: #ff4b6e; font-weight: 600;"
                        if val >= 4:
                            return "color: #ffaa00; font-weight: 600;"
                        return "color: #00d68f; font-weight: 600;"

                    styled_impact = impact_df.style.applymap(impact_color, subset=["Impact Score"])
                    st.markdown("### 📊 Parameter Impact Summary")
                    st.dataframe(styled_impact, use_container_width=True, hide_index=True)

                chart_col_1, chart_col_2, chart_col_3 = st.columns(3)

                with chart_col_1:
                    churn_values = [churn_count, safe_count]
                    churn_labels = ["Churn", "No Churn"]
                    fig_donut = go.Figure(
                        data=[
                            go.Pie(
                                labels=churn_labels,
                                values=churn_values,
                                hole=0.6,
                                marker=dict(colors=["#ff4b6e", "#00d68f"]),
                                textinfo="percent",
                                textposition="outside",
                            )
                        ]
                    )
                    fig_donut.add_annotation(
                        text="Churn Rate",
                        x=0.5,
                        y=0.5,
                        font=dict(color="white", size=14),
                        showarrow=False,
                    )
                    apply_plotly_theme(fig_donut, "📊 Churn Distribution")
                    st.plotly_chart(fig_donut, use_container_width=True)

                with chart_col_2:
                    fig_hist = px.histogram(
                        analysis_df,
                        x="Risk Score",
                        nbins=20,
                        color="Risk Score",
                        color_continuous_scale=["#00d68f", "#ffaa00", "#ff4b6e"],
                    )
                    fig_hist.add_vline(x=50, line_dash="dash", line_color="#ffffff")
                    apply_plotly_theme(fig_hist, "📈 Risk Score Distribution")
                    fig_hist.update_coloraxes(showscale=False)
                    st.plotly_chart(fig_hist, use_container_width=True)

                with chart_col_3:
                    if model is None or columns is None or not hasattr(model, "coef_"):
                        st.info("Model not available for feature importance.")
                    else:
                        coef_values = model.coef_[0]
                        coef_df = pd.DataFrame(
                            {
                                "feature": columns,
                                "coef": coef_values,
                                "abs_coef": np.abs(coef_values),
                            }
                        )
                        top10 = coef_df.nlargest(10, "abs_coef")
                        top10["feature"] = top10["feature"].str.replace("_", " ").str.title()
                        colors = ["#ff4b6e" if val > 0 else "#4f8ef7" for val in top10["coef"]]
                        fig_feat = go.Figure(
                            go.Bar(
                                x=top10["abs_coef"],
                                y=top10["feature"],
                                orientation="h",
                                marker_color=colors,
                            )
                        )
                        apply_plotly_theme(fig_feat, "🔍 Top Churn Drivers")
                        st.plotly_chart(fig_feat, use_container_width=True)

                chart_col_4, chart_col_5 = st.columns(2)

                with chart_col_4:
                    if "Contract" not in analysis_df.columns:
                        st.info("Contract column not found in uploaded file")
                    else:
                        contract_counts = (
                            analysis_df.groupby(["Contract", "Prediction"]).size().reset_index(name="count")
                        )
                        contracts = contract_counts["Contract"].unique()
                        churn_vals = []
                        stay_vals = []
                        for contract_item in contracts:
                            churn_vals.append(
                                contract_counts[
                                    (contract_counts["Contract"] == contract_item)
                                    & (contract_counts["Prediction"] == "Will Churn")
                                ]["count"].sum()
                            )
                            stay_vals.append(
                                contract_counts[
                                    (contract_counts["Contract"] == contract_item)
                                    & (contract_counts["Prediction"] == "Will Stay")
                                ]["count"].sum()
                            )

                        fig_contract = go.Figure()
                        fig_contract.add_bar(x=contracts, y=churn_vals, name="Churned", marker_color="#ff4b6e")
                        fig_contract.add_bar(x=contracts, y=stay_vals, name="Not Churned", marker_color="#00d68f")
                        fig_contract.update_layout(barmode="group")
                        apply_plotly_theme(fig_contract, "📋 Churn by Contract Type")
                        st.plotly_chart(fig_contract, use_container_width=True)

                with chart_col_5:
                    risk_scores = analysis_df["Risk Score"].fillna(0)
                    if "MonthlyCharges" in analysis_df.columns:
                        x_col = "MonthlyCharges"
                    else:
                        numeric_candidates = [
                            c for c in analysis_df.select_dtypes(include=[np.number]).columns if c != "Risk Score"
                        ]
                        x_col = numeric_candidates[0] if numeric_candidates else None

                    if x_col is None:
                        st.info("No numeric column found for scatter plot.")
                    else:
                        size_col = "TotalCharges" if "TotalCharges" in analysis_df.columns else None
                        colors = ["#ff4b6e" if score >= 70 else "#00d68f" for score in risk_scores]
                        fig_scatter = go.Figure(
                            go.Scatter(
                                x=analysis_df[x_col],
                                y=risk_scores,
                                mode="markers",
                                marker=dict(
                                    color=colors,
                                    size=analysis_df[size_col] if size_col else 10,
                                    sizemode="area",
                                    sizeref=2.0 * max(analysis_df[size_col].max(), 1) / (40 ** 2) if size_col else 1,
                                ),
                                text=analysis_df[id_col].astype(str),
                                hovertemplate="ID: %{text}<br>Risk: %{y:.1f}%<br>Value: %{x:.2f}<extra></extra>",
                            )
                        )
                        apply_plotly_theme(fig_scatter, "💰 Charges vs Churn Risk")
                        fig_scatter.update_yaxes(title="Churn Probability %")
                        fig_scatter.update_xaxes(title=x_col)
                        st.plotly_chart(fig_scatter, use_container_width=True)

                st.markdown("### 🗺️ Customer Risk Heatmap by Tenure")
                if "tenure" in analysis_df.columns:
                    tenure_bins = pd.cut(
                        analysis_df["tenure"],
                        bins=[0, 12, 24, 48, 72],
                        labels=["New", "Growing", "Mature", "Loyal"],
                        include_lowest=True,
                    )
                else:
                    tenure_bins = pd.qcut(
                        analysis_df.index + 1,
                        q=4,
                        labels=["New", "Growing", "Mature", "Loyal"],
                    )

                risk_bins = pd.cut(
                    analysis_df["Risk Score"],
                    bins=[0, 25, 50, 75, 100],
                    labels=["0-25%", "26-50%", "51-75%", "76-100%"],
                    include_lowest=True,
                )

                heat_df = pd.DataFrame({"Tenure Group": tenure_bins, "Risk Bucket": risk_bins})
                heat_map = heat_df.pivot_table(
                    index="Tenure Group",
                    columns="Risk Bucket",
                    aggfunc="size",
                    fill_value=0,
                )
                fig_heat = px.imshow(
                    heat_map,
                    color_continuous_scale=["#ffffff", "#ff4b6e"],
                    aspect="auto",
                )
                apply_plotly_theme(fig_heat, "🗺️ Customer Risk Heatmap by Tenure")
                st.plotly_chart(fig_heat, use_container_width=True)

                st.markdown("### 👥 Customer Prediction Results")
                st.markdown(
                    "Every customer from your uploaded file with their predicted churn risk"
                )

                table_col_1, table_col_2, table_col_3, table_col_4 = st.columns(4)
                with table_col_1:
                    risk_filter = st.selectbox("Risk Level", ["All", "High", "Medium", "Low"])
                with table_col_2:
                    search_term = st.text_input("Search customer...")
                with table_col_3:
                    sort_by = st.selectbox(
                        "Sort by",
                        ["Risk Score ↓", "Risk Score ↑", "Name A-Z", "Churn First"],
                    )
                with table_col_4:
                    csv_data = analysis_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Results CSV",
                        data=csv_data,
                        file_name="churn_predictions.csv",
                        mime="text/csv",
                    )

                selected_id = st.selectbox(
                    "🔍 View detailed profile for:",
                    analysis_df[id_col].astype(str).tolist(),
                )

                with st.expander(f"📋 Full Profile: {selected_id}", expanded=True):
                    selected_row = analysis_df[analysis_df[id_col].astype(str) == str(selected_id)].iloc[0]

                    profile_col_1, profile_col_2, profile_col_3 = st.columns(3)

                    personal_cols = ["gender", "SeniorCitizen", "Partner", "Dependents"]
                    service_cols = [
                        "PhoneService",
                        "MultipleLines",
                        "InternetService",
                        "OnlineSecurity",
                        "OnlineBackup",
                        "DeviceProtection",
                        "TechSupport",
                        "StreamingTV",
                        "StreamingMovies",
                    ]
                    account_cols = [
                        "Contract",
                        "PaperlessBilling",
                        "PaymentMethod",
                        "tenure",
                        "MonthlyCharges",
                        "TotalCharges",
                    ]

                    with profile_col_1:
                        st.markdown("### 👤 Personal Info")
                        for col in personal_cols:
                            if col in analysis_df.columns:
                                st.metric(label=col, value=str(selected_row[col]))

                    with profile_col_2:
                        st.markdown("### 📡 Service Info")
                        for col in service_cols:
                            if col in analysis_df.columns:
                                st.metric(label=col, value=str(selected_row[col]))

                    with profile_col_3:
                        st.markdown("### 💳 Account & Prediction")
                        for col in account_cols:
                            if col in analysis_df.columns:
                                st.metric(label=col, value=str(selected_row[col]))

                        risk = float(selected_row["Risk Score"])
                        if risk >= 70:
                            st.error(f"🚨 WILL CHURN — {risk:.0f}% probability")
                            gauge_color = "#ff4b6e"
                        elif risk >= 40:
                            st.warning(f"⚠️ AT RISK — {risk:.0f}% probability")
                            gauge_color = "#ffaa00"
                        else:
                            st.success(f"✅ WILL STAY — {risk:.0f}% probability")
                            gauge_color = "#00d68f"

                        fig_gauge = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=risk,
                                number={"suffix": "%", "font": {"color": gauge_color}},
                                gauge={
                                    "shape": "angular",
                                    "axis": {"range": [0, 100]},
                                    "bar": {"color": gauge_color},
                                    "bgcolor": "#1e2130",
                                },
                            )
                        )
                        apply_plotly_theme(fig_gauge)
                        st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown("### 🔍 Why is this customer predicted to churn/stay?")
                try:
                    if model is None or columns is None:
                        st.info("Model not available for contribution analysis.")
                    else:
                        selected_df = pd.DataFrame([selected_row])
                        selected_df = selected_df.drop(columns=["Risk Score", "Prediction", "Risk Level"], errors="ignore")
                        selected_encoded = pd.get_dummies(selected_df, drop_first=True)
                        selected_encoded = selected_encoded.reindex(columns=columns, fill_value=0)

                        model_coef = model.coef_[0]
                        contributions = model_coef * selected_encoded.values[0]
                        contrib_df = pd.DataFrame(
                            {
                                "Parameter": columns,
                                "Contribution": contributions,
                                "Direction": [
                                    "Increases Risk" if c > 0 else "Decreases Risk" for c in contributions
                                ],
                            }
                        )
                        contrib_df = contrib_df[contrib_df["Contribution"] != 0]
                        contrib_df["Abs"] = contrib_df["Contribution"].abs()
                        contrib_df = contrib_df.sort_values("Abs", ascending=False).head(10)
                        contrib_df["Parameter"] = contrib_df["Parameter"].str.replace("_", " ").str.title()

                        bar_colors = ["#ff4b6e" if c > 0 else "#4f8ef7" for c in contrib_df["Contribution"]]
                        fig_contrib = go.Figure(
                            go.Bar(
                                x=contrib_df["Contribution"],
                                y=contrib_df["Parameter"],
                                orientation="h",
                                marker_color=bar_colors,
                            )
                        )
                        title = f"Why {selected_id} is predicted to {'Churn' if risk >= 50 else 'Stay'}"
                        apply_plotly_theme(fig_contrib, title)
                        st.plotly_chart(fig_contrib, use_container_width=True)

                        top_risk_factors = contrib_df[contrib_df["Direction"] == "Increases Risk"]
                        top_safe_factors = contrib_df[contrib_df["Direction"] == "Decreases Risk"]
                        top_risk_factor = top_risk_factors.iloc[0]["Parameter"] if not top_risk_factors.empty else "N/A"
                        top_safe_factor = top_safe_factors.iloc[0]["Parameter"] if not top_safe_factors.empty else "N/A"

                        st.info(
                            f"""
                            📌 The biggest reason this customer is at risk is: 
                            **{top_risk_factor}**
                            🛡️ The biggest factor keeping them is: 
                            **{top_safe_factor}**
                            """
                        )
                except Exception as exc:
                    st.error("Error: " + str(exc))

                filtered_df = analysis_df.copy()
                filtered_df["Risk Score Value"] = filtered_df["Risk Score"].astype(float)

                if risk_filter != "All":
                    filtered_df = filtered_df[filtered_df["Risk Level"] == risk_filter]

                if search_term:
                    search_term_lower = search_term.lower()
                    search_cols = [id_col]
                    name_cols = [c for c in analysis_df.columns if "name" in c.lower()]
                    if name_cols:
                        search_cols.append(name_cols[0])
                    mask = filtered_df[search_cols].apply(
                        lambda row: any(search_term_lower in str(val).lower() for val in row),
                        axis=1,
                    )
                    filtered_df = filtered_df[mask]

                if sort_by == "Risk Score ↓":
                    filtered_df = filtered_df.sort_values("Risk Score Value", ascending=False)
                elif sort_by == "Risk Score ↑":
                    filtered_df = filtered_df.sort_values("Risk Score Value", ascending=True)
                elif sort_by == "Name A-Z":
                    name_cols = [c for c in analysis_df.columns if "name" in c.lower()]
                    if name_cols:
                        filtered_df = filtered_df.sort_values(name_cols[0])
                elif sort_by == "Churn First":
                    filtered_df = filtered_df.sort_values("Prediction", ascending=False)

                if "table_page" not in st.session_state:
                    st.session_state.table_page = 1

                total_rows = len(filtered_df)
                rows_per_page = 20
                total_pages = max(1, int(np.ceil(total_rows / rows_per_page)))
                st.session_state.table_page = min(st.session_state.table_page, total_pages)

                start_idx = (st.session_state.table_page - 1) * rows_per_page
                end_idx = start_idx + rows_per_page
                page_df = filtered_df.iloc[start_idx:end_idx].copy()

                def style_risk_badge(val):
                    try:
                        risk_val = float(val)
                    except Exception:
                        risk_val = 0
                    if risk_val >= 70:
                        return "background-color: rgba(255, 75, 110, 0.2); color: #ff4b6e; font-weight: 600;"
                    if risk_val >= 40:
                        return "background-color: rgba(255, 170, 0, 0.2); color: #ffaa00; font-weight: 600;"
                    return "background-color: rgba(0, 214, 143, 0.2); color: #00d68f; font-weight: 600;"

                def style_prediction(val):
                    if "Will Churn" in str(val):
                        return "color: #ff4b6e; font-weight: 600;"
                    return "color: #00d68f; font-weight: 600;"

                display_table = page_df.copy()
                display_table["Risk Score Display"] = display_table["Risk Score"].round(0).astype(int)
                display_table["Risk Score"] = display_table["Risk Score"].round(0).astype(int).astype(str) + "% " + display_table["Risk Level"]
                display_table["Prediction"] = display_table["Prediction"].apply(
                    lambda x: "🚨 Will Churn" if x == "Will Churn" else "✅ Will Stay"
                )

                styled = display_table.style
                styled = styled.applymap(style_prediction, subset=["Prediction"])
                styled = styled.applymap(style_risk_badge, subset=["Risk Score Display"])

                columns_to_show = list(analysis_df.columns) + ["Risk Score", "Prediction", "Risk Level"]
                columns_to_show = [col for col in columns_to_show if col in display_table.columns]

                st.dataframe(
                    styled,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Risk Score Display": None,
                    },
                )

                high_risk_count = int((filtered_df["Risk Level"] == "High").sum())
                st.markdown(
                    f"Showing {min(rows_per_page, total_rows)} of {total_rows} customers | {high_risk_count} at high risk"
                )

                pag_col_1, pag_col_2, pag_col_3 = st.columns([1, 2, 1])
                with pag_col_1:
                    if st.button("Previous"):
                        st.session_state.table_page = max(1, st.session_state.table_page - 1)
                with pag_col_2:
                    st.markdown(
                        f"<div style='text-align:center;'>Page {st.session_state.table_page} of {total_pages}</div>",
                        unsafe_allow_html=True,
                    )
                with pag_col_3:
                    if st.button("Next"):
                        st.session_state.table_page = min(total_pages, st.session_state.table_page + 1)

    except Exception as exc:
        st.error("Error: " + str(exc))

with tabs[1]:
    try:
        render_predictor_section(df, model, scaler, columns)
    except Exception as exc:
        st.error("Error: " + str(exc))

with tabs[2]:
    st.info("Coming soon")

with tabs[3]:
    st.info("Coming soon")

st.markdown(
    "<div style='text-align:center; color:#64748b; padding-top:0.75rem;'>"
    "© 2023 AuraAnalytics Inc. All rights reserved."
    "</div>",
    unsafe_allow_html=True,
)
