import os
import streamlit as st

# ---------------- USER SESSION ----------------
if "user" not in st.session_state:
    st.session_state.user = None
# ---------------------------------------------

import pandas as pd
from groq import Groq
import joblib
import shap
import random
import time
from datetime import datetime
import numpy as np
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

try:
    model = joblib.load(os.path.join(MODEL_DIR, "diabetes_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoder = joblib.load(os.path.join(MODEL_DIR, "encoder.pkl"))
    MODELS_AVAILABLE = True
except Exception as e:
    model = None
    scaler = None
    encoder = None
    MODELS_AVAILABLE = False
    load_error = str(e)

st.set_page_config(page_title="Health Predictor", layout="wide")

# ---------------------------
# GROQ CLIENT
# ---------------------------
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">

<style>

/* GLOBAL FONT */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
}

/* PAGE LAYOUT */
.main {
    padding: 20px 30px !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    backdrop-filter: blur(6px);
}

/* HEADINGS  */
h1, h2, h3 {
    margin-top: 6px !important;
    margin-bottom: 12px !important;
    font-weight: 700 !important;
}

/* TEXT */
p, li {
    line-height: 1.55 !important;
    font-size: 15px !important;
}

/* INPUTS */
.stNumberInput, .stSelectbox {
    margin-bottom: 12px !important;
}

/* CARD */
.card {
    padding: 22px;
    border-radius: 12px;
    margin-bottom: 18px;
    border: 1px solid rgba(0,0,0,0.15);
    backdrop-filter: blur(4px);
    transition: 0.22s ease;
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 22px rgba(0,0,0,0.25);
}

/* HERO SECTION */
.hero {
    text-align:center;
    padding:24px;
    border-radius:12px;
    margin-bottom:18px;
    backdrop-filter: blur(4px);
}

/* FORM PANEL */
.form-card {
    padding: 20px;
    border-radius: 12px;
    backdrop-filter: blur(4px);
    border: 1px solid rgba(0,0,0,0.12);
}

/* DIVIDER */
.divider {
    height: 1px;
    margin: 18px 0;
}

/* Hide theme switcher completely */
[data-testid="stThemeSwitcher"] {
    display: none !important;
}

/* BUTTONS */
.stButton>button {
    background-color: #4c8bf5 !important;
    color: white !important;
    padding: 8px 16px !important;
    border-radius: 6px !important;
    border: none !important;
    font-weight: 500 !important;
    transition: 0.18s ease !important;
}
.stButton>button:hover {
    background-color: #3a7be0 !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(76,139,245,0.32);
}

/* METER ANIMATION */
circle {
    transition: stroke-dashoffset 1.2s ease;
}

/* ANIMATIONS */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0px); }
}
</style>
""", unsafe_allow_html=True)

def play_sound():
    sound_html = """
    <audio autoplay>
        <source src="https://actions.google.com/sounds/v1/cartoon/wood_plank_flicks.ogg" type="audio/ogg">
    </audio>
    """
    st.markdown(sound_html, unsafe_allow_html=True)


health_tips = [
    "Drink 8 glasses of water every day.",
    "Walk 20–30 minutes daily.",
    "Avoid sugary drinks and junk food.",
    "Eat more fruits & vegetables.",
    "Sleep 7–8 hours every night.",
    "Reduce salt to maintain BP.",
    "Stretch every hour.",
    "Don't skip breakfast.",
    "Reduce screen time before bed.",
    "Practice deep breathing daily."
]

def get_daily_tip():
    return random.choice(health_tips)


def calculate_bmi(weight_kg, height_ft, height_in):
    height_m = (height_ft * 0.3048) + (height_in * 0.0254)
    if height_m <= 0:
        return None
    return weight_kg / (height_m ** 2)


def bmi_category(bmi):
    if bmi is None:
        return "Invalid"
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def predict_diabetes(age, gender_str, bp, glucose, bmi_value):
    if not MODELS_AVAILABLE:
        return "Model Missing", 0.0
    
    try:
        gender_enc = encoder.transform([gender_str])[0]
    except:
        gender_enc = 0

    X = pd.DataFrame([[age, gender_enc, bp, glucose, bmi_value]],
                     columns=["Age","Gender","BloodPressure","Glucose","BMI"])

    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    conf = model.predict_proba(X_scaled)[0].max()
    return pred, float(conf)

def generate_diabetes_explanation(age, glucose, bp, bmi):
    reasons = []
    notes = []

    if glucose >= 140:
        reasons.append("high glucose level")
        notes.append("Glucose is a major indicator of diabetes.")

    if bmi >= 25:
        reasons.append("increased BMI")
        notes.append("Higher BMI is linked to insulin resistance.")

    if bp >= 130:
        reasons.append("elevated blood pressure")
        notes.append("High blood pressure is associated with metabolic disorders.")

    if age >= 45:
        reasons.append("age-related risk")
        notes.append("Risk increases with age.")

    if reasons:
        return "This result is mainly due to " + ", ".join(reasons) + ". " + " ".join(notes)
    else:
        return "Your health parameters are within a healthy range, indicating lower diabetes risk."

def explain_diabetes_shap(age, gender_str, bp, glucose, bmi_value, pred):
    try:
        gender_enc = encoder.transform([gender_str])[0]
    except:
        gender_enc = 0

    X = pd.DataFrame([[age, gender_enc, bp, glucose, bmi_value]],
                     columns=["Age","Gender","BloodPressure","Glucose","BMI"])

    X_scaled = scaler.transform(X)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled)

    values = shap_values.values

    # Flatten properly
    values = np.array(values).flatten()

    feature_names = ["Age", "Gender", "BloodPressure", "Glucose", "BMI"]

    impact = dict(zip(feature_names, values))

    sorted_impact = sorted(impact.items(), key=lambda x: abs(x[1]), reverse=True)

    explanation = []



    for feature, val in sorted_impact:

        # 👉 For HIGH RISK → only positive contributors
        if pred == "High Risk" and val <= 0:
            continue

        # 👉 For LOW RISK → only negative contributors
        if pred != "High Risk" and val > 0:
            continue

        if feature == "BMI":
            if pred == "High Risk":
                explanation.append("Your BMI is contributing to increased diabetes risk.")
            else:
                explanation.append("Your BMI is in a healthy range, helping reduce diabetes risk.")

        elif feature == "Glucose":
            if pred == "High Risk":
                explanation.append("Your glucose level is high and is a major risk factor.")
            else:
                explanation.append("Your glucose level is normal, which is a positive sign.")

        elif feature == "Age":
            if pred == "High Risk":
                explanation.append("Your age increases your susceptibility to diabetes.")
            else:
                explanation.append("Your age is relatively low, reducing diabetes risk.")

        elif feature == "BloodPressure":
            if pred == "High Risk":
                explanation.append("Your blood pressure is contributing to higher risk.")
            else:
                explanation.append("Your blood pressure is within a healthy range.")

        elif feature == "Gender":
            # 👇 ONLY show if relevant after filtering
            if pred == "High Risk":
                explanation.append("Your gender has a minor influence on increasing risk.")
            else:
                explanation.append("Your gender is not contributing to diabetes risk.")

        # Stop after top 3 relevant reasons
        if len(explanation) == 3:
            break

    return explanation

def calculate_health_score(bp, glucose, bmi):
    score = 100
    if bmi < 18.5 or bmi > 30:
        score -= 20
    if bp > 140 or bp < 90:
        score -= 20
    if glucose > 140 or glucose < 70:
        score -= 30
    return max(score, 0)

PAGES = [
    "Home",
    "Check BMI",
    "Diabetes Prediction",
    "Stress Level Checker",
    "Symptom Chat",
    "Skin Cancer Detection",
    "About"
]

def change_page():
    st.session_state["page"] = st.session_state["sidebar_page"]

if "page" not in st.session_state:
    st.session_state["page"] = "Home"



page = st.session_state["page"]

# ---------------- LOGIN PAGE ----------------
if st.session_state.user is None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.title("🩺 Health Partner")
    st.subheader("👤 Enter Your Details")

    name = st.text_input("Full Name")
    age = st.number_input("Age", min_value=1, max_value=120, value=None, placeholder="Enter your age")
    gender = st.selectbox("Gender", ["Select", "Male", "Female"])

    col1, col2 = st.columns(2)

    with col1:
        height_ft = st.number_input("Height (feet)", min_value=0, max_value=8, value=None, placeholder="Feet")

    with col2:
        height_in = st.number_input("Height (inches)", min_value=0, max_value=11, value=None, placeholder="Inches")

    weight = st.number_input("Weight (kg)", min_value=10, max_value=200, value=None, placeholder="Enter your weight")

    if st.button("Continue"):
        if not name:
            st.warning("Please enter your name")

        elif age is None:
            st.warning("Please enter your age")

        elif gender == "Select":
            st.warning("Please select your gender")

        elif height_ft is None or height_in is None:
            st.warning("Please enter your height")

        elif weight is None:
            st.warning("Please enter your weight")

        else:
            st.session_state.user = {
                "name": name,
                "age": age,
                "gender": gender,
                "height_ft": height_ft,
                "height_in": height_in,
                "weight": weight
            }
            st.success("Profile created successfully!")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

# -------- SIDEBAR AFTER LOGIN ONLY --------

st.sidebar.radio(
    "Choose a page:",
    PAGES,
    key="sidebar_page",
    index=PAGES.index(st.session_state["page"]),
    on_change=change_page
)

st.sidebar.markdown("---")

# -------- USER PROFILE --------
# -------- USER PROFILE (IMPROVED UI) --------
user = st.session_state.user

height_m = (user["height_ft"] * 0.3048) + (user["height_in"] * 0.0254)
bmi = user["weight"] / (height_m ** 2)

st.sidebar.markdown("### 👤 **User Profile**")

st.sidebar.markdown(f"""
**Name:** {user['name']}  
**Age:** {user['age']}  
**Gender:** {user['gender']}  
**Height:** {user['height_ft']} ft {user['height_in']} in  
**Weight:** {user['weight']} kg  
""")
# ------------------------------------------

st.sidebar.markdown("---")


if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()
# -----------------------------------------
# ------------------------------------------------

if page == "Home":
    user = st.session_state.user
    st.write(f"👋 Welcome, {user['name']}")
    st.markdown("""
    <div class="hero">
        <h1>🏥 Health Predictor</h1>
        <p>Your personal AI-based health companion</p>
    </div>
    """, unsafe_allow_html=True)


    st.write("")
    st.markdown("### Explore Features")

    features = [
        ("Check BMI", "Check BMI"),
        ("Diabetes Prediction", "Diabetes Prediction"),
        ("Stress Level Checker", "Stress Level Checker"),
        ("Symptom Chat", "Symptom Chat"),
        ("Skin Cancer Detection", "Skin Cancer Detection"),
        ("About", "About"),
    ]

    cols = st.columns(3)

    for idx, (label, page_name) in enumerate(features):
        with cols[idx % 3]:
            if st.button(label):
                st.session_state["page"] = page_name
                st.rerun()


    st.markdown("""
    <div class="card">
        <h3>✨ What this app offers</h3>
        <ul>
            <li>Instant BMI calculation</li>
            <li>ML-based Diabetes Prediction</li>
            <li>Personalized Health Score</li>
            <li>Nearby Doctor & Hospital Finder</li>
            <li>Daily Health Tips</li>
        </ul>
    </div>

    <div class="card">
        <h3>💙 Daily Health Tip</h3>
    </div>
    """, unsafe_allow_html=True)

    st.info(get_daily_tip())

elif page == "Check BMI":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("BMI Calculator")
    st.write("")
    st.markdown("<div class='form-card'>", unsafe_allow_html=True)

    with st.form("bmi_form"):
        user = st.session_state.user
        age = st.number_input("Age", 1, 120, user["age"])
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if user["gender"]=="Male" else 1)

        c1, c2 = st.columns(2)
        with c1:
            height_ft = st.number_input("Height (feet)", 0, 8, user["height_ft"])
        with c2:
            height_in = st.number_input("Height (inches)", 0, 11, user["height_in"])
        weight = st.number_input("Weight (kg)", 10.0, 300.0, float(user["weight"]))

        submitted = st.form_submit_button("Calculate BMI")

    if submitted:
        bmi = calculate_bmi(weight, height_ft, height_in)
        if bmi is None:
            st.error("Invalid height.")
        else:
            cat = bmi_category(bmi)
            st.success(f"Your BMI = {bmi:.2f} ({cat})")
            play_sound()
            st.toast("BMI calculated successfully!", icon="🟦")

    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Diabetes Prediction":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Diabetes Prediction")

    if not MODELS_AVAILABLE:
        st.error("Model files missing.")
        st.write(load_error)

    else:
        st.write("")
        st.markdown("<div class='form-card'>", unsafe_allow_html=True)

        with st.form("dia_form"):
            user = st.session_state.user
            age = st.number_input("Age", 1, 120, user["age"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            bp = st.number_input("Blood Pressure (mmHg)", 60, 250, 120)
            st.caption("Normal BP ≈ 120 mmHg")

            glucose = st.number_input("Glucose (mg/dL)", 30, 300, 100)
            st.caption("Normal fasting glucose ≈ 70–100 mg/dL")

            height_m = (user["height_ft"] * 0.3048) + (user["height_in"] * 0.0254)
            auto_bmi = user["weight"] / (height_m ** 2)
            bmi = st.number_input("BMI", 10.0, 70.0, float(auto_bmi))

            submitted = st.form_submit_button("Predict Risk")

        if submitted:
            pred, conf = predict_diabetes(age, gender, bp, glucose, bmi)
            score = calculate_health_score(bp, glucose, bmi)

            st.write(f"### 🧡 Health Score: **{score}/100**")
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div style='text-align:center; margin-top:10px;'>
                <svg width="180" height="180">
                    <circle cx="90" cy="90" r="70" stroke="#ddd" stroke-width="8" fill="none"/>
                    <circle cx="90" cy="90" r="70" stroke="#4c8bf5" stroke-width="8" fill="none"
                        stroke-dasharray="440"
                        stroke-dashoffset="{440 - (score * 4.4)}"
                        stroke-linecap="round"
                        transform="rotate(-90 90 90)"
                    />
                    <text x="90" y="100" text-anchor="middle" fill="#4c8bf5" font-size="26">{score}</text>
                </svg>
            </div>
            """, unsafe_allow_html=True)

            st.write(f"### Prediction: **{pred}**")
            play_sound()
            st.toast("Diabetes prediction completed.", icon="💙")
            st.write(f"### Confidence: **{conf:.2f}**")
        
            explanation = generate_diabetes_explanation(age, glucose, bp, bmi)
            shap_explanation = explain_diabetes_shap(age, gender, bp, glucose, bmi, pred)
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.subheader("🧠 Why this result?")
            st.info(explanation)

            st.markdown("### 🔍 Detailed AI Explanation")

            for line in shap_explanation:
                st.write(f"• {line}")

            if pred == "High Risk":
                st.error("⚠ HIGH RISK DETECTED")
                st.write("""
                ### What you should do NOW:
                • Reduce sugar & salt  
                • Drink more water  
                • Avoid rice, sweets, junk foods  
                • Walk daily 20–30 minutes  
                """)

                st.write("### Helplines:")
                st.write("• 1800-180-1104 (National Health Helpline)")
                st.write("• 1075 (Health Ministry)")

                colA, colB = st.columns(2)

                with colA:
                    st.markdown(
                        '<a href="https://www.google.com/maps/search/diabetes+doctor+near+me/" target="_blank">'
                        '<button style="background-color:#ef4444; color:white; padding:10px; border-radius:8px;">'
                        'Find Doctor Near Me</button></a>',
                        unsafe_allow_html=True)

                with colB:
                    st.markdown(
                        '<a href="https://www.google.com/maps/search/diabetes+hospital+near+me/" target="_blank">'
                        '<button style="background-color:#1f78b4; color:white; padding:10px; border-radius:8px;">'
                        'Find Hospital Near Me</button></a>',
                        unsafe_allow_html=True)

            else:
                st.success("🎉 You're completely fit, champ!")
                st.write("""
                ### Keep this up:
                • Eat healthy  
                • Exercise daily  
                • Drink water  
                • Do regular checkups  
                """)

            report = f"Age:{age}\nBP:{bp}\nGlucose:{glucose}\nBMI:{bmi}\nScore:{score}\nRisk:{pred}"
            downloaded = st.download_button("Download Report", report, "health_report.txt")

            if downloaded:
                play_sound()
                st.toast("Report downloaded!", icon="📄")

    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Stress Level Checker":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("🧠 Mental Health & Stress Level Check")
    st.write("Answer honestly to understand your emotional well-being.")
    st.markdown("<div class='form-card'>", unsafe_allow_html=True)

    def option_scale(question, key):
        st.markdown(
        f"""
        <div style='margin-top:12px; margin-bottom:6px;'>
            <span style='font-size:18px; font-weight:700; color:#4c8bf5;'>{question}</span>
        </div>
        """,
        unsafe_allow_html=True
        )

        return st.radio(
        "Choose an option",
        ["😄 Never", "🙂 Rarely", "😐 Sometimes", "😟 Often", "😢 Always"],
        key=key,
        horizontal=True,
        label_visibility="collapsed"   
        )



    score_map = {
        "😄 Never": 1,
        "🙂 Rarely": 2,
        "😐 Sometimes": 3,
        "😟 Often": 4,
        "😢 Always": 5
    }

    with st.form("mental_form"):
        st.write("### Your daily mental state:")
        st.markdown(
        """
        <div style='margin-top:12px; margin-bottom:8px;'>
            <span style='font-size:20px; font-weight:700; color:#4c8bf5;'>
                💤 How many hours do you sleep per day?
            </span>
        </div>
        """,
        unsafe_allow_html=True
        )

        sleep_hours = st.number_input(
            "Enter hours of sleep (per day)", 
            min_value=0, 
            max_value=24, 
            value=7, 
            step=1
        )

        st.markdown("<hr style='margin:18px 0;'>", unsafe_allow_html=True)

        q1 = option_scale("1)How often do you feel stressed?", "q1")
        q2 = option_scale("2)How often do you overthink?", "q2")
        q3 = option_scale("3)How often do you feel irritated?", "q3")
        q4 = option_scale("4)Difficulty sleeping?", "q4")
        q5 = option_scale("5)Feeling low energy?", "q5")
        q6 = option_scale("6)Feeling anxious?", "q6")
        q7 = option_scale("7)Loss of appetite or overeating?", "q7")
        q8 = option_scale("8)Feeling lonely?", "q8")
        q9 = option_scale("9)Trouble focusing?", "q9")
        q10 = option_scale("10)Physical stress (headache/back pain)?", "q10")

        submitted = st.form_submit_button("Check Stress Level")

    if submitted:
        score = sum([
            score_map[q1], score_map[q2], score_map[q3], score_map[q4], score_map[q5],
            score_map[q6], score_map[q7], score_map[q8], score_map[q9], score_map[q10]
        ])

        play_sound()
        st.toast("Stress analysis completed!", icon="🧠")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        st.write(f"### 🧡 Your Stress Score: **{score}/50**")

        if score <= 15:
            meter_color = "#3CCF4E"  
        elif score <= 30:
            meter_color = "#F4D03F"  
        elif score <= 45:
            meter_color = "#F39C12"  
        else:
            meter_color = "#E74C3C"  

        st.markdown(f"""
        <div style='text-align:center; margin-top:10px;'>
            <svg width="180" height="180">
                <circle cx="90" cy="90" r="70" stroke="#ddd" stroke-width="8" fill="none"/>
                <circle cx="90" cy="90" r="70" stroke="{meter_color}" stroke-width="8" fill="none"
                    stroke-dasharray="440"
                    stroke-dashoffset="{440 - (score * 8.8)}"
                    stroke-linecap="round"
                    transform="rotate(-90 90 90)"
                />
                <text x="90" y="100" text-anchor="middle" fill="{meter_color}" font-size="26">{score}</text>
            </svg>
        </div>
        """, unsafe_allow_html=True)

        if score <= 15:
            st.success("🌟 **Your Mental Health Is Excellent — Mental Health Champ!**")
            st.write("""
            ### Keep it up Champ!
            - Your mind is stable and healthy  
            - Continue mindfulness habits  
            - Maintain sleep schedule  
            - Keep social connections active  
            """)

        elif score <= 30:
            st.warning("🙂 **Moderate Stress — Manageable but needs attention.**")
            st.write("""
            ### Tips to improve:
            - Try 5-minute deep breathing  
            - Reduce screen time before bed  
            - Light evening walk  
            - Stay hydrated  
            """)

        elif score <= 45:
            st.error("😥 **High Stress Detected — You need rest and recovery.**")
            st.write("""
            ### What you should do now:
            - Take a short break from work/study  
            - Try guided meditation (YouTube apps available)  
            - Talk to someone you trust  
            - Organize your tasks to reduce pressure  
            """)

        else:
            st.error("⚠️ **Severe Stress — Please take this seriously.**")
            st.write("""
            ### Immediate steps:
            - Talk to a close friend/family  
            - Do not isolate yourself  
            - Try breathing + grounding techniques  
            - Reduce workload for 1–2 days  
            """)

            st.write("### Helplines:")
            st.write("• **iCall (India): 9152987821**")
            st.write("• **Aasra: 9820466726**")
            st.write("• **Health Ministry: 14416**")

            colA, colB = st.columns(2)

            colA.markdown(
                '<a href="https://www.google.com/maps/search/mental+health+doctor+near+me/" target="_blank">'
                '<button style="background-color:#E74C3C;color:white;padding:10px;border-radius:8px;">'
                'Find Mental Health Doctor Near Me</button></a>',
                unsafe_allow_html=True
            )

            colB.markdown(
                '<a href="https://www.google.com/maps/search/mental+health+hospital+near+me/" target="_blank">'
                '<button style="background-color:#2980B9;color:white;padding:10px;border-radius:8px;">'
                'Find Mental Health Hospital Near Me</button></a>',
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)


elif page == "Symptom Chat":

    import time

    # ================= CUSTOM CSS =================

    st.markdown("""
    <style>

    .ai-header{
        text-align:center;
        padding:28px 20px;
        border-radius:24px;
        background:linear-gradient(135deg,#0F172A,#1E3A8A);
        margin-bottom:20px;
        box-shadow:0 8px 28px rgba(0,0,0,0.18);
    }

    .ai-title{
        font-size:38px;
        font-weight:700;
        color:white;
        margin-top:10px;
    }

    .ai-sub{
        color:#CBD5E1;
        font-size:15px;
        margin-top:6px;
    }

    .robot{
        font-size:60px;
    }

    .clear-wrap{
        display:flex;
        justify-content:flex-end;
        margin-bottom:12px;
    }

    </style>
    """, unsafe_allow_html=True)

    # ================= HEADER =================

    st.markdown(
    """
    <div class="ai-header">
        <div class="robot">🤖</div>
        <div class="ai-title">Healthcare AI Assistant</div>
        <div class="ai-sub">
            AI-powered healthcare guidance for symptoms,
            wellness and health awareness.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

    # ================= CLEAR CHAT =================

    c1, c2 = st.columns([8,2])

    with c2:

        if st.button("🗑 Clear Chat"):

            st.session_state.chat_messages = []

            st.rerun()

    # ================= SESSION =================

    if "chat_messages" not in st.session_state:

        st.session_state.chat_messages = []

    # ================= DISPLAY OLD CHATS =================

    for msg in st.session_state.chat_messages:

        avatar = None

        if msg["role"] == "user":

            if st.session_state.user["gender"] == "Male":

                avatar = "🧑🏻"

            else:

                avatar = "👩🏻"

        else:

            avatar = "🤖"

        with st.chat_message(msg["role"], avatar=avatar):

            st.markdown(msg["content"])

    # ================= USER INPUT =================

    user_prompt = st.chat_input(
        "Describe your symptoms or ask a health question..."
    )

    # ================= RESPONSE =================

    if user_prompt:

        # ---------- USER MESSAGE ----------

        user_avatar = (
            "🧑🏻"
            if st.session_state.user["gender"] == "Male"
            else "👩🏻"
        )

        st.session_state.chat_messages.append({

            "role": "user",

            "content": user_prompt
        })

        # ================= CHAT LIMIT =================

        if len(st.session_state.chat_messages) > 20:

            st.session_state.chat_messages = (
                st.session_state.chat_messages[-20:]
            )

        with st.chat_message("user", avatar=user_avatar):

            st.markdown(user_prompt)

        # ---------- ASSISTANT ----------

        with st.chat_message("assistant", avatar="🤖"):

            thinking = st.empty()

            thinking.markdown("🤖 Typing...")

            time.sleep(1)

            try:

                # ================= EMERGENCY DETECTION =================

                emergency_keywords = [
                    "chest pain",
                    "heart attack",
                    "can't breathe",
                    "cannot breathe",
                    "difficulty breathing",
                    "stroke",
                    "unconscious",
                    "severe bleeding"
                ]

                if any(
                    k in user_prompt.lower()
                    for k in emergency_keywords
                ):

                    final_response = (
                        "⚠️ Your symptoms may require urgent medical attention. "
                        "Please contact a healthcare professional or visit the nearest emergency facility immediately."
                    )

                else:

                    # ================= CHAT MEMORY =================

                    chat_context = [

                        {
                            "role": "system",
                            "content": (

                                "You are Health Partner AI, a professional healthcare assistant integrated inside a healthcare web application. "

                                "You must remember previous conversation context carefully. "

                                "Never forget previous symptoms mentioned by the user. "

                                "If the user clarifies something, update your understanding instead of contradicting previous messages. "

                                "Your responses should feel like a calm intelligent healthcare assistant. "

                                "Keep responses SHORT, natural, professional and conversational. "

                                "Avoid robotic replies and avoid unnecessary disclaimers. "

                                "Do not repeatedly ask questions forever. "

                                "After asking 1 or 2 relevant follow-up questions, provide practical wellness guidance or possible causes. "

                                "If symptoms appear mild, provide simple lifestyle guidance. "

                                "If symptoms appear serious, recommend consulting a healthcare professional. "

                                "Never suddenly change topic. "

                                "Never act emotionally or dramatically. "

                                "Never say things like 'you're not feeling like talking'. "

                                "Stay medically focused and context-aware. "

                                "Examples:\n\n"

                                "User: headache\n"
                                "Assistant: Headaches are often linked to stress, dehydration or lack of sleep. Is the pain mild or severe?\n\n"

                                "User: severe\n"
                                "Assistant: Severe headaches can sometimes occur due to stress, migraine or exhaustion. Try resting, staying hydrated and reducing screen exposure for some time. If symptoms continue frequently, consult a healthcare professional.\n\n"

                                "User: stress\n"
                                "Assistant: Stress can sometimes affect both sleep and energy levels. Try maintaining proper sleep, hydration and relaxation for a few days. If stress becomes persistent, consider consulting a mental health professional."
                            )
                        }
                    ]

                    # ================= PREVIOUS MEMORY =================

                    for old_msg in st.session_state.chat_messages[-8:]:

                        if old_msg["role"] == "user":

                            chat_context.append({

                                "role": "user",

                                "content": old_msg["content"]
                            })

                        else:

                            chat_context.append({

                                "role": "assistant",

                                "content": old_msg["content"]
                            })

                    # ================= CURRENT MESSAGE =================

                    chat_context.append({

                        "role": "user",

                        "content": user_prompt
                    })

                    # ================= GROQ RESPONSE =================

                    response = client.chat.completions.create(

                        model="llama-3.3-70b-versatile",

                        messages=chat_context,

                        temperature=0.2,

                        max_tokens=90
                    )

                    final_response = (

                        response.choices[0]
                        .message.content
                        .strip()
                    )

            except Exception as e:

                final_response = f"Error: {e}"

            thinking.markdown(final_response)

        # ================= SAVE ASSISTANT RESPONSE =================

        st.session_state.chat_messages.append({

            "role": "assistant",

            "content": final_response
        })

        # ================= CHAT LIMIT =================

        if len(st.session_state.chat_messages) > 20:

            st.session_state.chat_messages = (
                st.session_state.chat_messages[-20:]
            )

elif page == "Skin Cancer Detection":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("🩺 Skin Cancer Detection (Experimental)")
    st.write("Upload a skin lesion image to get a model prediction. "
             "This is NOT a medical diagnosis. Always consult a dermatologist.")

    st.markdown("<div class='form-card'>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload PNG/JPG image", type=['png', 'jpg', 'jpeg'])

    import os
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "skin_model.h5")

    if not os.path.isfile(model_path):
        st.error("Skin cancer model not found! Train the model first.")
    else:
        if uploaded:
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp_file.write(uploaded.read())
            temp_file.flush()
            img_path = temp_file.name

            st.image(img_path, caption="Uploaded Image", width='stretch')

            st.write("Running prediction...")

            from src.skin_predictor import predict_skin
            top_class, confidence, top3, overlay_img, explanation = predict_skin(img_path)

            st.subheader("Prediction Result")
            st.write(f"### 🔍 Most Likely Class: **{top_class}**")
            st.write(f"### 🎯 Confidence: **{confidence*100:.2f}%**")

            st.write("---")

            st.write("### Top 3 Predictions:")
            for cls, prob in top3:
                st.write(f"- **{cls}** — {prob*100:.2f}%")

            st.write("---")
            st.subheader("🔍 Model Focus Area")
            st.image(overlay_img, caption="Explainable AI (Grad-CAM)", width='stretch')
            
            st.subheader("🧠 Why this result?")
            st.info(explanation)

            st.write("---")

            if top_class in ["MEL", "SCC", "BCC"]:
                st.error("⚠ Possible high-risk lesion. Please consult a dermatologist urgently.")
            else:
                st.warning("⚠ Model suggests low-risk class, but still consult a doctor if unsure.")

            st.info(
                "This feature is experimental and for educational purposes only. "
                "Do NOT use it as medical advice."
            )

    st.markdown("</div>", unsafe_allow_html=True)

elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("About")
    st.write("""
    **Early Disease Prediction System**  
    Features:  
    - BMI Calculator  
    - Diabetes Prediction using ML  
    - Mental Health Check  
    - Health Score  
    - Daily Health Tips  
    - Doctor/Hospital Finder  
    """)
    st.markdown("</div>", unsafe_allow_html=True)   