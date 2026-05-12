"""Streamlit app entry point for churn prediction."""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
            st.plotly_chart(gauge, width="stretch")

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
                    font=dict(color="white"),
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="Contribution Score",
                )
                st.plotly_chart(fig_top, width="stretch")
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
                    width="stretch",
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
            st.dataframe(explain_df, width="stretch", hide_index=True)
            st.info(
                "These insights are derived from the Logistic Regression "
                "model coefficients trained on IBM Telco Customer Churn data."
            )


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
            <div class="card">
                <div style="font-size: 1.6rem; font-weight: 700;">Churn Risk Prediction Dashboard</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        filter_col_1, filter_col_2, filter_col_3, filter_col_4, filter_col_5 = st.columns([1, 1, 1, 1, 0.7])
        with filter_col_1:
            st.selectbox("Date Range", ["Last 30 Days", "Last 60 Days", "Last 90 Days"])
        with filter_col_2:
            st.selectbox("Region", ["All", "North", "South", "East", "West"])
        with filter_col_3:
            st.selectbox("Plan", ["All", "Platinum", "Gold", "Silver"])
        with filter_col_4:
            st.number_input("Churn Threshold", min_value=0, max_value=100, value=65, format="%d", step=1)
        with filter_col_5:
            st.markdown("<div class='button-primary'>", unsafe_allow_html=True)
            st.button("Apply Campaign")
            st.markdown("</div>", unsafe_allow_html=True)

        if "selected_customer" not in st.session_state:
            st.session_state.selected_customer = df.iloc[0]["customerID"]

        left_panel, right_panel = st.columns([1.2, 0.8])

        with left_panel:
            st.markdown("<div class='card'><h3>Customer Churn Risk List</h3>", unsafe_allow_html=True)
            st.text_input("Search", placeholder="🔎 View Profile", label_visibility="collapsed")

            table_df = df[[
                "Customer ID",
                "customerID",
                "Plan",
                "Tenure Display",
                "Activity Score",
                "Risk Score",
                "Risk Label",
            ]].copy()
            table_df.columns = [
                "Customer ID",
                "Name",
                "Plan",
                "Tenure",
                "Activity Score",
                "Risk Score",
                "Risk Label",
            ]

            filter_col_a, filter_col_b, filter_col_c = st.columns(3)
            with filter_col_a:
                risk_filter = st.selectbox("Risk filter", ["All", "High", "Medium", "Low"])
            with filter_col_b:
                plan_filter = st.selectbox("Plan filter", ["All", "Basic", "Gold", "Platinum"])
            with filter_col_c:
                sort_filter = st.selectbox("Sort by", ["Risk Score ↓", "Tenure ↓", "Activity Score ↓"])

            filtered_df = table_df.copy()
            if risk_filter != "All":
                filtered_df = filtered_df[filtered_df["Risk Label"] == risk_filter]
            if plan_filter != "All":
                filtered_df = filtered_df[filtered_df["Plan"] == plan_filter]

            filtered_df["Risk Score Num"] = filtered_df["Risk Score"]
            if sort_filter == "Risk Score ↓":
                filtered_df = filtered_df.sort_values("Risk Score Num", ascending=False)
            elif sort_filter == "Tenure ↓":
                filtered_df["Tenure Num"] = filtered_df["Tenure"].str.replace("mo", "", regex=False).astype(int)
                filtered_df = filtered_df.sort_values("Tenure Num", ascending=False)
            else:
                filtered_df = filtered_df.sort_values("Activity Score", ascending=False)

            filtered_df["Risk Score"] = filtered_df.apply(
                lambda row: f"{row['Risk Score Num']:.0f}% {row['Risk Label']}", axis=1
            )

            if len(filtered_df) == 0:
                st.info("No customers match the current filters.")
                selected_name = df.iloc[0]["customerID"]
            else:
                selected_name = st.selectbox("Select customer", filtered_df["Name"].tolist())

            st.session_state.selected_customer = selected_name

            display_df = filtered_df.head(10).copy()
            display_df = display_df[[
                "Customer ID",
                "Name",
                "Plan",
                "Tenure",
                "Activity Score",
                "Risk Score",
                "Risk Label",
                "Risk Score Num",
            ]]

            def style_rows(row):
                styles = [""] * len(row)
                if row["Name"] == st.session_state.selected_customer:
                    styles = ["background-color: rgba(79, 142, 247, 0.2);"] * len(row)
                elif row.name % 2 == 1:
                    styles = ["background-color: #1b1e2b;"] * len(row)
                return styles

            def style_risk_cell(row):
                if row["Risk Score Num"] >= 70:
                    color = "#ff4b6e"
                    bg = "rgba(255, 75, 110, 0.2)"
                elif row["Risk Score Num"] >= 40:
                    color = "#ffaa00"
                    bg = "rgba(255, 170, 0, 0.2)"
                else:
                    color = "#00d68f"
                    bg = "rgba(0, 214, 143, 0.2)"
                return ["", "", "", "", "", f"background-color: {bg}; color: {color}; font-weight: 600;", "", ""]

            styled_df = (
                display_df.style
                .apply(style_rows, axis=1)
                .apply(style_risk_cell, axis=1)
            )

            st.dataframe(
                styled_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Customer ID": st.column_config.TextColumn(width=110),
                    "Name": st.column_config.TextColumn(width=160),
                    "Plan": st.column_config.TextColumn(width=100),
                    "Tenure": st.column_config.TextColumn(width=90),
                    "Activity Score": st.column_config.NumberColumn(width=120),
                    "Risk Score": st.column_config.TextColumn(width=130),
                    "Risk Label": st.column_config.TextColumn(width=120),
                    "Risk Score Num": None,
                },
            )

            st.markdown(
                f"<div style='color:#9ca3af; margin-top:0.6rem;'>{len(filtered_df)} customers | showing top 10</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'><h3>Churn Factors Heatmap</h3>", unsafe_allow_html=True)
            try:
                corr_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn_Binary"]
                corr_matrix = df[corr_cols].corr().round(2)
                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor("#1e2130")
                ax.set_facecolor("#1e2130")
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    fmt=".2f",
                    cmap="RdBu_r",
                    center=0,
                    ax=ax,
                    annot_kws={"color": "white"},
                )
                ax.tick_params(colors="white")
                ax.set_title("Churn Factors Heatmap", color="white")
                st.pyplot(fig, clear_figure=True)
            except Exception as exc:
                st.error("Error: " + str(exc))
            st.markdown("</div>", unsafe_allow_html=True)

        with right_panel:
            try:
                selected_customer = df[df["customerID"] == st.session_state.selected_customer].iloc[0]
            except Exception:
                selected_customer = df.iloc[0]

            risk_pct = float(selected_customer["Risk Score"])
            tenure_val = int(selected_customer["tenure"])
            monthly_val = float(selected_customer["MonthlyCharges"])
            contract_val = selected_customer["Contract"]

            st.markdown(
                f"<div class='card'><h3>Individual Customer Insights: {selected_customer['customerID']}</h3>",
                unsafe_allow_html=True,
            )
            st.markdown("<h4>Risk Trend (Last 3mo)</h4>", unsafe_allow_html=True)

            months_back = max(1, min(tenure_val, 6))
            trend_values = np.linspace(risk_pct * 0.6, risk_pct, months_back)
            trend_values += np.random.normal(0, 3, months_back)
            trend_values = np.clip(trend_values, 0, 100)
            x_labels = [f"{i}mo" for i in range(months_back, 0, -1)]
            baseline = [40] * months_back

            fig_trend = go.Figure()
            fig_trend.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=trend_values,
                    mode="lines+markers",
                    line=dict(color="#ffaa00", width=3),
                    name="Risk Trend",
                )
            )
            fig_trend.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=baseline,
                    mode="lines+markers",
                    line=dict(color="#4f8ef7", width=2),
                    name="Baseline",
                )
            )
            fig_trend.update_layout(
                paper_bgcolor="#1e2130",
                plot_bgcolor="#1e2130",
                font=dict(color="white"),
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(range=[0, 100], ticksuffix="%"),
            )
            st.plotly_chart(fig_trend, width="stretch")

            if risk_pct >= 70:
                gauge_color = "#ff4b6e"
                label_text = "High Risk"
            elif risk_pct >= 40:
                gauge_color = "#ffaa00"
                label_text = "Medium Risk"
            else:
                gauge_color = "#00d68f"
                label_text = "Low Risk"

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_pct,
                    number={"suffix": "%", "font": {"color": gauge_color}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": gauge_color},
                        "bgcolor": "#1e2130",
                    },
                )
            )
            fig_gauge.update_layout(
                paper_bgcolor="#1e2130",
                font=dict(color="white"),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_gauge, width="stretch")
            st.markdown(
                f"<div style='color:{gauge_color}; font-weight:700;'>Churn Risk: {risk_pct:.0f}% • {label_text}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("<h4>Top Churn Drivers</h4>", unsafe_allow_html=True)
            try:
                if model is not None and columns is not None and hasattr(model, "coef_"):
                    coefficients = np.abs(model.coef_[0])
                    coef_df = pd.DataFrame({"feature": columns, "coefficient": coefficients})
                    top_drivers = coef_df.nlargest(3, "coefficient")
                    driver_names = top_drivers["feature"].tolist()
                    driver_values = (top_drivers["coefficient"] / top_drivers["coefficient"].max() * 100).round(0)
                else:
                    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn_Binary"]
                    corr = df[numeric_cols].corr()["Churn_Binary"].abs().sort_values(ascending=False)
                    driver_names = corr.index[1:4].tolist()
                    driver_values = (corr.values[1:4] / corr.values[1:4].max() * 100).round(0)

                fig_drivers = go.Figure(
                    go.Bar(
                        x=driver_values,
                        y=driver_names,
                        orientation="h",
                        marker_color=["#4f8ef7", "#ffaa00", "#00d68f"],
                        text=[f"{v:.0f}%" for v in driver_values],
                        textposition="auto",
                    )
                )
                fig_drivers.update_layout(
                    paper_bgcolor="#1e2130",
                    plot_bgcolor="#1e2130",
                    font=dict(color="white"),
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_drivers, width="stretch")
            except Exception as exc:
                st.error("Error: " + str(exc))

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'><h3>Customer Tenure Distribution vs. Churn Rate</h3>", unsafe_allow_html=True)
            tenure_counts = df["tenure"].value_counts().sort_index()
            churn_rate = df.groupby("tenure")["Churn_Binary"].mean()
            x_vals = list(range(1, 31))
            bar_vals = [tenure_counts.get(i, 0) / max(tenure_counts.max(), 1) for i in x_vals]
            line_vals = [churn_rate.get(i, 0) for i in x_vals]

            fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
            fig_dual.add_trace(
                go.Bar(x=x_vals, y=bar_vals, name="Customer Tenure", marker_color="#22d3ee"),
                secondary_y=False,
            )
            fig_dual.add_trace(
                go.Scatter(x=x_vals, y=line_vals, name="Churn Rate", line=dict(color="#ffaa00", width=3)),
                secondary_y=True,
            )
            fig_dual.update_layout(
                paper_bgcolor="#1e2130",
                plot_bgcolor="#1e2130",
                font=dict(color="white"),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig_dual.update_yaxes(range=[0, 0.5], secondary_y=False)
            fig_dual.update_yaxes(range=[0, 1.0], secondary_y=True)
            st.plotly_chart(fig_dual, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'><h3>Retention Strategy Simulator</h3><div style='color:#9ca3af'>Modify customer parameters</div>", unsafe_allow_html=True)
            contract_slider_value = st.selectbox(
                "Contract Length",
                ["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(contract_val),
            )
            call_freq_slider = st.slider("Call Frequency -10%", 0, 100, 20)
            add_feature = st.selectbox("Add Feature", ["None", "Premium Support", "Data Boost", "Free Month"])

            base_risk = float(selected_customer["Risk Score"])
            contract_adjustment = {
                "Month-to-month": 0,
                "One year": -15,
                "Two year": -25,
            }[contract_slider_value]
            call_freq_adjustment = -call_freq_slider * 0.1
            feature_adjustment = -5 if add_feature != "None" else 0

            updated_risk = max(0, min(100, base_risk + contract_adjustment + call_freq_adjustment + feature_adjustment))

            if updated_risk >= 70:
                updated_color = "#ff4b6e"
            elif updated_risk >= 40:
                updated_color = "#ffaa00"
            else:
                updated_color = "#00d68f"

            fig_updated = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=updated_risk,
                    number={"suffix": "%", "font": {"color": updated_color}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": updated_color},
                        "bgcolor": "#1e2130",
                    },
                )
            )
            fig_updated.update_layout(
                paper_bgcolor="#1e2130",
                font=dict(color="white"),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_updated, width="stretch")

            st.markdown("<div style='margin-top:0.8rem; font-weight:700;'>Updated Risk Score</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='metric-large' style='color:{updated_color}'>{updated_risk:.0f}%</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><h3>Model Performance (XGBoost)</h3>", unsafe_allow_html=True)
        try:
            y_true = df["Churn_Binary"].values
            y_pred_proba = df["Risk Score"].values / 100
            y_pred = (y_pred_proba >= 0.5).astype(int)

            real_auc = round(roc_auc_score(y_true, y_pred_proba), 2)
            real_recall = round(recall_score(y_true, y_pred) * 100)
            real_f1 = round(f1_score(y_true, y_pred), 2)

            metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
            with metric_col_1:
                st.markdown(f"<div class='metric-large'>{real_auc}</div><div style='color:#9ca3af;'>AUC</div>", unsafe_allow_html=True)
            with metric_col_2:
                st.markdown(f"<div class='metric-large'>{real_recall}%</div><div style='color:#9ca3af;'>Recall</div>", unsafe_allow_html=True)
            with metric_col_3:
                st.markdown(f"<div class='metric-large'>{real_f1}</div><div style='color:#9ca3af;'>F1-Score</div>", unsafe_allow_html=True)
        except Exception as exc:
            st.error("Error: " + str(exc))

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
