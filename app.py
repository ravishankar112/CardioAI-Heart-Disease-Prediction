import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CardioAI | Heart Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PROFESSIONAL CSS
# =========================================================
st.markdown("""
<style>

/* =========================
   GLOBAL READABILITY
   ========================= */
html, body, [class*="css"], .stApp {
    font-family: "Inter", "Segoe UI", Arial, sans-serif;
}

.stApp {
    background: #F4F8FB;
    color: #111827;
}

/* Main text is dark/black for easy reading */
.stApp p,
.stApp label,
.stApp span,
.stApp div[data-testid="stMarkdownContainer"] p,
.stApp div[data-testid="stMarkdownContainer"] li,
.stApp div[data-testid="stMarkdownContainer"] strong,
.stApp div[data-testid="stMarkdownContainer"] b {
    color: #111827;
}

/* Streamlit headings */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: #071A2B !important;
    font-weight: 800 !important;
}

#MainMenu, footer { visibility: hidden; }

/* =========================
   SIDEBAR — COLOURED DESIGN
   ========================= */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #062B46 0%, #075C6D 55%, #087F73 100%);
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.22) !important;
}

/* =========================
   HERO — COLOURED DESIGN
   ========================= */
.hero {
    background: linear-gradient(135deg, #062B46 0%, #0A6475 58%, #0D8B78 100%);
    padding: 38px 42px;
    border-radius: 24px;
    color: #FFFFFF !important;
    margin-bottom: 28px;
    box-shadow: 0 14px 38px rgba(6,43,70,0.20);
}

.hero * { color: #FFFFFF !important; }
.hero-title {
    font-size: 40px;
    font-weight: 850;
    margin-bottom: 8px;
}
.hero-subtitle {
    font-size: 18px;
    line-height: 1.55;
    opacity: 0.95;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.20);
    padding: 8px 15px;
    border-radius: 20px;
    margin-top: 18px;
    font-size: 14px;
}

/* =========================
   COLOURED CARDS
   ========================= */
.card {
    background: #FFFFFF;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #D8E5EC;
    box-shadow: 0 7px 24px rgba(16,45,65,0.07);
    margin-bottom: 20px;
}
.card-title {
    color: #071A2B !important;
    font-size: 21px;
    font-weight: 800;
    margin-bottom: 7px;
}
.card-subtitle {
    color: #263746 !important;
    font-size: 14px;
    line-height: 1.65;
}

/* =========================
   STATS
   ========================= */
.stat-card {
    background: linear-gradient(145deg, #FFFFFF 0%, #F2FAFA 100%);
    border: 1px solid #CFE3E8;
    border-top: 4px solid #0B8178;
    border-radius: 17px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 6px 20px rgba(16,45,65,0.06);
}
.stat-value {
    color: #087B74 !important;
    font-size: 26px;
    font-weight: 850;
}
.stat-label {
    color: #1F2937 !important;
    font-size: 13px;
    margin-top: 5px;
    font-weight: 600;
}

/* =========================
   RESULT
   ========================= */
.result-heading {
    color: #071A2B !important;
    font-size: 30px;
    font-weight: 850;
    margin-bottom: 5px;
}
.result-subheading {
    color: #34495A !important;
    font-size: 15px;
    margin-bottom: 18px;
}
.risk-high, .risk-low {
    padding: 24px 26px;
    border-radius: 16px;
    margin: 12px 0 20px 0;
    box-shadow: 0 7px 20px rgba(16,45,65,0.07);
}
.risk-high {
    background: linear-gradient(135deg, #FFF0F0 0%, #FFF9F9 100%);
    border: 1px solid #F0B7B7;
    border-left: 7px solid #D23C3C;
}
.risk-low {
    background: linear-gradient(135deg, #E9FAF3 0%, #F9FFFC 100%);
    border: 1px solid #B6E3D2;
    border-left: 7px solid #118B69;
}
.risk-title {
    color: #071A2B !important;
    font-size: 24px;
    font-weight: 850;
    margin-bottom: 8px;
}
.risk-text {
    color: #17212B !important;
    font-size: 15px;
    line-height: 1.7;
}

/* =========================
   INPUTS — BLACK TEXT
   ========================= */
.stSlider label,
.stSelectbox label,
.stNumberInput label,
.stTextInput label,
.stRadio label {
    color: #111827 !important;
    font-weight: 700 !important;
}

div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] > div,
div[data-testid="stTextInput"] > div {
    background: #FFFFFF !important;
    border: 1px solid #B8C9D3 !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] *,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    color: #111827 !important;
}

/* =========================
   BUTTON — COLOUR
   ========================= */
.stButton > button {
    background: linear-gradient(90deg, #075B70, #0A8A78) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    height: 54px;
    font-size: 17px;
    font-weight: 800;
    box-shadow: 0 7px 20px rgba(7,91,112,0.22);
}
.stButton > button * { color: #FFFFFF !important; }
.stButton > button:hover {
    background: linear-gradient(90deg, #064A5B, #087565) !important;
    transform: translateY(-1px);
}

/* =========================
   TABLE + METRICS
   ========================= */
div[data-testid="stDataFrame"] {
    border: 1px solid #D5E2E9;
    border-radius: 12px;
    overflow: hidden;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #D4E2E8;
    border-radius: 15px;
    padding: 18px;
    box-shadow: 0 5px 18px rgba(16,45,65,0.05);
}

div[data-testid="stMetricLabel"] p {
    color: #243746 !important;
    font-weight: 700 !important;
}

div[data-testid="stMetricValue"] {
    color: #071A2B !important;
    font-weight: 850 !important;
}

/* =========================
   DIVIDER + FOOTER
   ========================= */
.section-line {
    height: 2px;
    background: linear-gradient(90deg, #0A8A78, #DCE8ED);
    margin: 28px 0;
}
.footer {
    text-align: center;
    color: #263746 !important;
    padding: 28px;
    font-size: 13px;
    line-height: 1.6;
}
.footer b { color: #075B70 !important; }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================
try:
    model = XGBClassifier()
    model.load_model("XGBoost_heart.json")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")

except Exception as e:
    st.error("Model files could not be loaded.")
    st.code(str(e))
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("""
    <div style="text-align:center;">
        <div style="font-size:45px;">❤️</div>
        <h2>CardioAI</h2>
        <p style="opacity:0.8;">Intelligent Heart Risk Awareness</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### SYSTEM")

    st.markdown("""
    **Model**  
    XGBoost

    **Input Parameters**  
    11 cardiovascular features

    **Purpose**  
    Early risk awareness

    **Platform**  
    Streamlit
    """)

    st.markdown("---")

    st.markdown("### SDG ALIGNMENT")

    st.markdown("""
    **SDG 3**  
    Good Health and Well-Being
    """)

    st.markdown("---")

    st.caption("CardioAI Project")
    st.caption("CardioAI")


# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">

<div class="hero-title">
❤️ CardioAI
</div>

<div class="hero-subtitle">
AI-Based Heart Disease Risk Prediction & Early Awareness System
</div>

<div class="hero-badge">
Machine Learning • XGBoost • SDG 3
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# STATS
# =========================================================
s1, s2, s3, s4 = st.columns(4)

with s1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">AI</div>
        <div class="stat-label">Risk Assessment</div>
    </div>
    """, unsafe_allow_html=True)

with s2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">XGB</div>
        <div class="stat-label">ML Algorithm</div>
    </div>
    """, unsafe_allow_html=True)

with s3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">11</div>
        <div class="stat-label">Health Parameters</div>
    </div>
    """, unsafe_allow_html=True)

with s4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">SDG 3</div>
        <div class="stat-label">Health & Well-Being</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")


# =========================================================
# INTRODUCTION
# =========================================================
st.markdown("""
<div class="card">

<div class="card-title">
About the Assessment
</div>

<div class="card-subtitle">
This system uses a trained machine learning model to estimate
heart disease risk from selected cardiovascular health parameters.
The result is intended for early awareness and educational
decision-support, not clinical diagnosis.
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# PATIENT INFORMATION
# =========================================================
st.markdown("""
<div class="card">
<div class="card-title">01 — Patient Information</div>
<div class="card-subtitle">
Basic demographic information used by the prediction model.
</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.slider(
        "Age",
        min_value=18,
        max_value=100,
        value=40
    )

with col2:
    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )


# =========================================================
# CARDIOVASCULAR PARAMETERS
# =========================================================
st.markdown("""
<div class="card">
<div class="card-title">02 — Cardiovascular Parameters</div>
<div class="card-subtitle">
Enter the patient's available cardiovascular measurements.
</div>
</div>
""", unsafe_allow_html=True)


col1, col2 = st.columns(2)

with col1:
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

with col2:
    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )


col1, col2 = st.columns(2)

with col1:
    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=200,
        value=120
    )

with col2:
    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=100,
        max_value=600,
        value=200
    )


col1, col2 = st.columns(2)

with col1:
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1]
    )

with col2:
    max_hr = st.slider(
        "Maximum Heart Rate",
        min_value=60,
        max_value=220,
        value=150
    )


col1, col2 = st.columns(2)

with col1:
    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Y", "N"]
    )

with col2:
    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )


oldpeak = st.slider(
    "Oldpeak (ST Depression)",
    min_value=0.0,
    max_value=6.0,
    value=1.0,
    step=0.1
)


st.write("")


# =========================================================
# PREDICTION
# =========================================================
st.markdown("""
<div class="card">

<div class="card-title">03 — AI Risk Assessment</div>

<div class="card-subtitle">
The trained XGBoost model will process the entered parameters
and generate an estimated risk classification.
</div>

</div>
""", unsafe_allow_html=True)


if st.button(
    "Run AI Risk Assessment",
    use_container_width=True
):

    # -----------------------------------------------------
    # RAW INPUT
    # -----------------------------------------------------

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,

        "Sex_" + sex: 1,

        "ChestPainType_" + chest_pain: 1,

        "RestingECG_" + resting_ecg: 1,

        "ExerciseAngina_" + exercise_angina: 1,

        "ST_Slope_" + st_slope: 1
    }


    input_df = pd.DataFrame([raw_input])


    # -----------------------------------------------------
    # ADD MISSING COLUMNS
    # -----------------------------------------------------

    for col in expected_columns:

        if col not in input_df.columns:

            input_df[col] = 0


    # Correct order

    input_df = input_df[expected_columns]


    try:

        # -------------------------------------------------
        # SCALE
        # -------------------------------------------------

        scaled_input = scaler.transform(input_df)


        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        prediction = int(model.predict(scaled_input)[0])
        probability = float(model.predict_proba(scaled_input)[0][1]) * 100


        # =================================================
        # RESULT
        # =================================================

        st.markdown("""
        <div class="section-line"></div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="result-heading">Prediction Result</div>
        <div class="result-subheading">AI-based assessment of the health information provided</div>
        """, unsafe_allow_html=True)

        if prediction == 1:
            st.markdown("""
            <div class="risk-high">
                <div class="risk-title">🔴 Higher Estimated Risk</div>
                <div class="risk-text">
                    The machine learning model estimates a higher risk of heart disease
                    based on the health information provided.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-low">
                <div class="risk-title">🟢 Lower Estimated Risk</div>
                <div class="risk-text">
                    The machine learning model estimates a lower risk of heart disease
                    based on the health information provided.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # =================================================
        # RISK PROBABILITY
        # =================================================

        st.markdown("### Risk Probability")

        probability_col, gauge_col = st.columns([1, 2])

        with probability_col:
            st.metric("Estimated Risk", f"{probability:.1f}%")

        with gauge_col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability,
                number={"suffix": "%"},
                title={"text": "Estimated Heart Disease Risk"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#128C7E"},
                    "steps": [
                        {"range": [0, 40], "color": "#E8F7F1"},
                        {"range": [40, 70], "color": "#FFF6D8"},
                        {"range": [70, 100], "color": "#FFE8E8"}
                    ]
                }
            ))
            fig.update_layout(
                height=280,
                margin=dict(l=20, r=20, t=45, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)


        # =================================================
        # INPUT SUMMARY
        # =================================================

        st.write("")

        st.markdown("### Assessment Input Summary")


        summary = pd.DataFrame({

            "Parameter": [
                "Age",
                "Sex",
                "Chest Pain Type",
                "Resting Blood Pressure",
                "Cholesterol",
                "Fasting Blood Sugar",
                "Resting ECG",
                "Maximum Heart Rate",
                "Exercise Angina",
                "Oldpeak",
                "ST Slope"
            ],

            "Value": [
                age,
                sex,
                chest_pain,
                f"{resting_bp} mm Hg",
                f"{cholesterol} mg/dL",
                fasting_bs,
                resting_ecg,
                max_hr,
                exercise_angina,
                oldpeak,
                st_slope
            ]
        })


        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


    except Exception as e:

        st.error(
            "Prediction could not be completed."
        )

        st.code(str(e))


# =========================================================
# RESPONSIBLE AI
# =========================================================
st.markdown("""
<div class="section-line"></div>

<div class="card">

<div class="card-title">
Responsible AI & Safety
</div>

<div class="card-subtitle">

<b>Transparency:</b>
The prediction is generated using a trained XGBoost
machine learning model.

<br><br>

<b>Privacy:</b>
Users should avoid entering unnecessary personally identifiable
information into this educational prototype.

<br><br>

<b>Medical Safety:</b>
This system does not diagnose disease and must not replace
professional medical evaluation.

<br><br>

<b>Fairness:</b>
Model performance may vary across populations depending on the
training data. Results should therefore be interpreted carefully.

</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# PROJECT INFORMATION
# =========================================================
st.markdown("""
<div class="card">

<div class="card-title">
Project Information
</div>

<div class="card-subtitle">

<b>Project:</b>
AI-Based Heart Disease Risk Prediction

<br><br>

<b>Primary SDG:</b>
SDG 3 — Good Health and Well-Being

<br><br>

<b>AI Technique:</b>
XGBoost

<br><br>

<b>Application:</b>
Early health-risk awareness and decision-support

<br><br>

<b>Developer:</b>
Ravi Shankar Kumar

</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# USER FEEDBACK
# =========================================================
st.markdown("""
<div class="card">
<div class="card-title">User Feedback</div>
<div class="card-subtitle">Help improve the CardioAI experience.</div>
</div>
""", unsafe_allow_html=True)

feedback_rating = st.radio(
    "How would you rate this application?",
    ["Excellent", "Good", "Average", "Needs Improvement"],
    horizontal=True
)
feedback_text = st.text_area(
    "Your feedback",
    placeholder="Write your suggestion or experience here...",
    height=100
)

if st.button("Submit Feedback", use_container_width=True):
    st.success("Thank you for your feedback!")


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">

<b>CardioAI</b> • AI-Based Heart Disease Risk Prediction

<br><br>

Educational & Awareness Prototype — Not a Medical Diagnosis

</div>
""", unsafe_allow_html=True)