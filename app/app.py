import os
import streamlit as st

# ---------------- USER SESSION ----------------
if "user" not in st.session_state:
    st.session_state.user = None
# ---------------------------------------------

import pandas as pd
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

    # SHAP explainer
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

    from datetime import datetime
    import time
    import re
    import html

    st.markdown(
        """
        <style>
        /* Chat bubbles */
        .chat-row { display:flex; gap:10px; margin-bottom:10px; align-items:flex-end; }
        .user-bubble { background:#DCF8C6; padding:12px; border-radius:14px; border-bottom-right-radius:2px; max-width:78%; animation: fadeIn 0.18s ease; }
        .assistant-bubble { background: #FFFFFF; padding:12px; border-radius:14px; border:1px solid rgba(0,0,0,0.06); max-width:78%; animation: fadeIn 0.18s ease; }
        .meta-time { font-size:11px; margin-top:6px; color:#6b7280; display:block; }

        /* Avatar styles */
        .avatar { width:44px; height:44px; border-radius:8px; display:flex; align-items:center; justify-content:center; }
        .user-avatar { background: linear-gradient(135deg,#32CD32,#22a552); color:white; }
        .doc-avatar { background: linear-gradient(135deg,#2b7edb,#1f6fb3); color:white; }

        /* Animations */
        @keyframes fadeIn { from { opacity:0; transform: translateY(6px);} to {opacity:1; transform: translateY(0);} }

        /* Make chat input bigger + wider */
        [data-testid="stChatInputArea"] textarea {
            min-height: 52px !important;
            font-size: 16px !important;
            padding: 12px !important;
        }
        [data-testid="stChatInput"] {
            width: 90% !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("💬 Symptom Chat — Talk to Your Virtual Clinician")
    st.write("Describe what you're feeling. I'll ask a few friendly follow-up questions and guide you safely.")
    st.markdown("<div class='form-card'>", unsafe_allow_html=True)

    clear_col, _ = st.columns([0.18, 0.82])
    if clear_col.button("🗑 Clear Chat"):
        st.session_state.pop("chat_history", None)
        st.session_state.pop("chat_state", None)
        st.session_state.pop("last_user_msgs", None)
        st.rerun()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_state" not in st.session_state:
        st.session_state.chat_state = {
            "domain": None,
            "q_index": 0,
            "answers": {},
            "finished": False,
            "emergency": False,
            "user_topic": None,
            "show_finders": False,
            "severity_score": 0,
        }

    state = st.session_state.chat_state

    def now_ts():
        return datetime.now().strftime("%I:%M %p")

    def push_user(text):
        st.session_state.chat_history.append({"role": "user", "text": text, "time": now_ts()})

    def push_assistant(text):
        st.session_state.chat_history.append({"role": "assistant", "text": text, "time": now_ts()})

    def classify_initial(text):
        t = text.lower()
        if any(k in t for k in ["chest pain", "can't breathe", "cannot breathe", "passing out", "faint"]):
            return "emergency", "emergency"
        if any(k in t for k in ["anxious", "panic", "overthinking", "depressed", "stress"]):
            return "mental", "mental health"
        if any(k in t for k in ["fever", "temperature", "shivering"]):
            return "fever", "fever"
        if any(k in t for k in ["vomit", "nausea", "diarrhea", "stomach"]):
            return "stomach", "stomach"
        if any(k in t for k in ["headache", "migraine", "dizzy"]):
            return "head", "headache"
        if any(k in t for k in ["rash", "itch", "skin", "spots"]):
            return "skin", "skin"
        if any(k in t for k in ["breath", "shortness", "wheeze", "cough"]):
            return "breathing", "breathing"
        return "general", "general wellness"

    FOLLOWUPS = {
        "fever": [("temp", "Do you have a measured temperature?"), ("cough", "Any cough or throat pain?"), ("aches", "Any muscle/body pain?")],
        "stomach": [("vomit", "Are you vomiting?"), ("diarr", "Any loose motions?"), ("pain_loc", "Where exactly is the pain?")],
        "head": [("light", "Does light or noise worsen your headache?"), ("neck", "Any neck stiffness or fever?")],
        "skin": [("itch", "Is it itchy?"), ("spread", "Has the rash spread?"), ("prod", "Used any new product or soap?")],
        "breathing": [("breath", "Breathless at rest or activity?"), ("wheeze", "Any wheezing?"), ("chest", "Any chest tightness?")],
        "mental": [("sleep", "How many hours do you sleep daily?"), ("mood", "Feeling low or overwhelmed?"), ("energy", "Any low energy or focus issues?")],
        "general": [("sleep", "How many hours do you sleep daily?"), ("appetite", "Any change in appetite?"), ("energy", "Feeling tired often?")],
    }

    def is_emergency(text):
        t = text.lower()
        flags = ["chest pain", "cannot breathe", "faint", "severe bleeding", "slurred speech", "sudden weakness"]
        return any(k in t for k in flags)

    def update_severity(qkey, ans):
        s = state["severity_score"]
        ans = ans.lower()

        if "severe" in ans:
            s += 30
        if "faint" in ans:
            s += 40
        if "worse" in ans:
            s += 20

        nums = re.findall(r"\d+", ans)
        if nums:
            n = int(nums[0])
            if "temp" in qkey:
                if n >= 39:
                    s += 30
                elif n >= 38:
                    s += 15

        state["severity_score"] = min(100, s)

    def natural_summary(domain):
        sev = state["severity_score"]

        if sev >= 70:
            tone = "Thanks for sharing that — some of your symptoms feel a bit more intense, so let’s be cautious."
        elif sev >= 40:
            tone = "Alright — there are a few things here worth paying attention to."
        else:
            tone = "Thanks — this sounds manageable right now."

        if domain == "fever":
            return f"{tone} This pattern fits a feverish illness. Stay hydrated, rest well, and monitor temperature. If breathing worsens or fever persists → see a doctor."
        if domain == "stomach":
            return f"{tone} These signs match indigestion or stomach infection. Sip ORS, avoid oily food, and seek care if vomiting persists or dehydration starts."
        if domain == "head":
            return f'{tone} This feels like a tension or migraine-type headache. Reduce screen time, stay hydrated, and rest. If sudden worst headache of your life → get urgent care.'
        if domain == "skin":
            return f"{tone} This looks like a mild rash or allergy. Avoid irritants, keep the area clean, and monitor spread. If swelling or breathing issues → get urgent help."
        if domain == "breathing":
            return f"{tone} Breathing-related symptoms deserve careful monitoring. If breathlessness worsens or happens at rest, please seek immediate medical attention."
        if domain == "mental":
            return f"{tone} Your responses suggest mental fatigue or stress buildup. Consistent sleep, routine breaks, and talking to someone you trust can help a lot."
        return f"{tone} Nothing alarming right now — keep a balanced routine and observe symptoms over the next 1–2 days."

    def assistant_typing(final):
        st.session_state.chat_history.append(
            {"role": "assistant", "text": "⏳ Dr is typing…", "time": now_ts(), "typing": True}
        )
        time.sleep(0.8)

        if st.session_state.chat_history[-1].get("typing"):
            st.session_state.chat_history.pop()

        push_assistant(final)

    user_msg = st.chat_input("Tell me how you're feeling…")

    if user_msg:
        msg = user_msg.strip()
        push_user(msg)

        if is_emergency(msg):
            assistant_typing("I’m detecting serious warning signs. Please go to the nearest emergency room immediately.")
            state["finished"] = True
            state["show_finders"] = True

        else:
            if state["domain"] is None or state["finished"]:
                dom, topic = classify_initial(msg)
                state["domain"] = dom
                state["q_index"] = 0
                state["answers"] = {}
                state["finished"] = False
                state["show_finders"] = False
                state["severity_score"] = 0
                assistant_typing(f"Okay — let me ask a couple of things about your {topic} so I can understand better.")

            follow = FOLLOWUPS.get(state["domain"], FOLLOWUPS["general"])
            if state["q_index"] > 0 and (state["q_index"] - 1) < len(follow):
                prev_key = follow[state["q_index"] - 1][0]
                state["answers"][prev_key] = msg
                update_severity(prev_key, msg)

            if state["q_index"] < len(follow):
                key, q = follow[state["q_index"]]
                assistant_typing(q)
                state["q_index"] += 1
            else:
                summary = natural_summary(state["domain"])
                assistant_typing(summary)
                state["finished"] = True
                state["show_finders"] = True

    _, chat_col = st.columns([0.06, 0.94])
    with chat_col:
        for msg in st.session_state.chat_history:
            safe_text = html.escape(msg["text"]).replace("\n", "<br>")
            t = msg.get("time", "")

            if msg["role"] == "user":
                user_svg = """
                <svg width="28" height="28" viewBox="0 0 24 24" fill="white">
                    <circle cx="12" cy="7" r="4"/>
                    <path d="M4 20c0-4 3-6 8-6s8 2 8 6"/>
                </svg>"""
                st.markdown(
                    f"""
                    <div class="chat-row" style="justify-content:flex-end;">
                        <div style="display:flex;flex-direction:column;align-items:flex-end;">
                            <div class="user-bubble">{safe_text}</div>
                            <span class="meta-time">{t}</span>
                        </div>
                        <div class="avatar user-avatar">{user_svg}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                doc_svg = """
                <svg width="28" height="28" viewBox="0 0 24 24" fill="#1f6fb3">
                    <circle cx="12" cy="7" r="4" fill="white"/>
                    <path d="M4 20c0-4 3-6 8-6s8 2 8 6" fill="white"/>
                </svg>"""
                st.markdown(
                    f"""
                    <div class="chat-row" style="justify-content:flex-start;">
                        <div class="avatar doc-avatar">{doc_svg}</div>
                        <div>
                            <div class="assistant-bubble">{safe_text}</div>
                            <span class="meta-time">{t}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if state.get("show_finders"):
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown(
            '<a href="https://www.google.com/maps/search/doctor+near+me/" target="_blank">'
            '<button style="background:#1f6fb3;padding:10px;color:white;border-radius:8px;margin-right:8px;">Find Doctor</button></a>'
            '<a href="https://www.google.com/maps/search/hospital+near+me/" target="_blank">'
            '<button style="background:#2b7edb;padding:10px;color:white;border-radius:8px;">Find Hospital</button></a>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

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