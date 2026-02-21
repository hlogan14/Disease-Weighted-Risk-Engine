# =============================================================================
# sentinel/app.py
# Streamlit UI for the Sentinel Weighted Risk Engine.
#
# Run with:  streamlit run app.py
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from engine import score_all_diseases

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title  = "Sentinel – Chronic Disease Risk Screener",
    page_icon   = "🛡️",
    layout      = "centered",
)

# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------
RISK_COLORS = {"High": "#E74C3C", "Medium": "#F39C12", "Low": "#2ECC71"}
RISK_EMOJIS = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

DISEASE_DESCRIPTIONS = {
    "Heart Disease": (
        "Heart disease encompasses conditions that affect the heart's structure "
        "and function, including coronary artery disease and heart failure. "
        "Key risk factors include high blood pressure, high cholesterol, smoking, "
        "and a sedentary lifestyle."
    ),
    "Chronic Kidney Disease": (
        "Chronic Kidney Disease (CKD) is the gradual loss of kidney function over time. "
        "Diabetes and high blood pressure are the two leading causes. "
        "Early detection through lifestyle screening can significantly slow progression."
    ),
    "Lung Cancer": (
        "Lung cancer is one of the most common and serious types of cancer. "
        "Smoking remains the #1 risk factor, but genetic predisposition and "
        "occupational chemical exposure also play a significant role."
    ),
    "Diabetes Type 2": (
        "Type 2 diabetes is a condition affecting how your body processes blood sugar. "
        "It is strongly linked to BMI, diet, and family history. "
        "Lifestyle interventions are highly effective at reducing risk."
    ),
}

DISCLAIMER = (
    "⚠️ **Disclaimer:** Sentinel is a **screening tool only** and does **not** "
    "constitute medical advice, diagnosis, or treatment. "
    "All results are estimates based on self-reported data and statistical models. "
    "Please consult a qualified healthcare professional for any health concerns."
)


def risk_badge(label: str) -> str:
    color = RISK_COLORS.get(label, "#999")
    emoji = RISK_EMOJIS.get(label, "⚪")
    return (
        f'<span style="background-color:{color};color:white;'
        f'padding:4px 14px;border-radius:20px;font-weight:bold;'
        f'font-size:1rem;">{emoji} {label} Risk</span>'
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🛡️ Sentinel")
st.subheader("Chronic Disease Exposure Rating Engine")
st.markdown(
    "Answer the questions below honestly. Sentinel will estimate your "
    "**exposure rating** (High / Medium / Low) for four chronic diseases "
    "using a data-driven weighted risk model."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Input Form
# ---------------------------------------------------------------------------
with st.form("sentinel_form"):

    st.markdown("### 👤 Demographics")
    col1, col2 = st.columns(2)
    with col1:
        age    = st.number_input("What is your age?", min_value=19, max_value=80,
                                 value=35, step=1)
        gender = st.selectbox("What is your gender?",
                              ["Male", "Female", "Other"])
        race   = st.selectbox("What is your race/ethnicity?",
                              ["Caucasian", "African American", "Asian", "Other"])
    with col2:
        height_in  = st.number_input("What is your height? (inches)",
                                     min_value=48, max_value=96, value=68, step=1)
        weight_lbs = st.number_input("What is your weight? (lbs)",
                                     min_value=80, max_value=500, value=160, step=1)

    st.markdown("---")
    st.markdown("### 🩺 Medical History")
    col3, col4 = st.columns(2)
    with col3:
        blood_pressure = st.selectbox(
            "Has a doctor ever told you that you have high blood pressure?",
            ["Good", "High", "Not Sure"]
        )
        cholesterol = st.selectbox(
            "Has a doctor ever told you that you have high or low cholesterol?",
            ["Good", "Low", "High", "Not Sure"]
        )
        stroke = st.selectbox(
            "Have you ever had a stroke or mini-stroke (TIA)?",
            ["No", "Yes", "Not Sure"]
        )
        anemia = st.selectbox(
            "Have you ever been told you are anemic (low iron/blood count)?",
            ["No", "Yes", "Not Sure"]
        )
    with col4:
        chest_pain = st.selectbox(
            "Do you experience chest pain or tightness during physical activity?",
            ["No", "Sometimes", "Yes"]
        )
        diabetes_diagnosed = st.selectbox(
            "Have you ever been diagnosed with diabetes?",
            ["No", "Yes", "Not Sure"]
        )
        hba1c_high = st.selectbox(
            "Has a doctor ever told you your blood sugar or A1C is high or borderline?",
            ["No", "Yes", "Not Sure"]
        )
        family_history_kidney = st.selectbox(
            "Does anyone in your immediate family have kidney disease?",
            ["No", "Yes", "Not Sure"]
        )

    st.markdown("---")
    st.markdown("### 🧬 Family & Genetic History")
    col5, col6 = st.columns(2)
    with col5:
        genetic_risk_lung = st.selectbox(
            "Does anyone in your immediate family have a history of lung cancer?",
            ["No", "Yes", "Not Sure"]
        )
    with col6:
        occupational_hazards = st.selectbox(
            "Does your job expose you to dust, chemicals, fumes, or asbestos?",
            ["No", "Yes", "Not Sure"]
        )

    st.markdown("---")
    st.markdown("### 🚬 Lifestyle")
    col7, col8 = st.columns(2)
    with col7:
        smoking          = st.selectbox("Do you currently smoke?", ["No", "Yes"])
        alcohol          = st.number_input("How many alcoholic drinks do you have per week?",
                                           min_value=0, max_value=20, value=2, step=1)
        sugar_consumption = st.selectbox(
            "How would you describe your daily sugar / processed food intake?",
            ["Low", "Medium", "High"]
        )
        physical_activity = st.number_input(
            "How many hours per week do you exercise?",
            min_value=0, max_value=10, value=3, step=1
        )
    with col8:
        stress_level = st.selectbox(
            "How would you rate your overall stress level?",
            ["Low", "Medium", "High"]
        )
        hours_of_sleep = st.number_input(
            "How many hours of sleep do you get on average per night?",
            min_value=4, max_value=10, value=7, step=1
        )
        sleep_quality = st.slider(
            "How well-rested do you feel most mornings? (1 = exhausted, 10 = fully rested)",
            min_value=1, max_value=10, value=6
        )

    st.markdown("---")
    submitted = st.form_submit_button("🔍 Calculate My Exposure Rating",
                                     use_container_width=True)

# ---------------------------------------------------------------------------
# Scoring & Results
# ---------------------------------------------------------------------------
if submitted:
    user_inputs = {
        "age":                   age,
        "gender":                gender,
        "race":                  race,
        "height_in":             height_in,
        "weight_lbs":            weight_lbs,
        "blood_pressure":        blood_pressure,
        "cholesterol":           cholesterol,
        "stroke":                stroke,
        "anemia":                anemia,
        "chest_pain":            chest_pain,
        "diabetes_diagnosed":    diabetes_diagnosed,
        "hba1c_high":            hba1c_high,
        "family_history_kidney": family_history_kidney,
        "genetic_risk_lung":     genetic_risk_lung,
        "occupational_hazards":  occupational_hazards,
        "smoking":               smoking,
        "alcohol":               alcohol,
        "sugar_consumption":     sugar_consumption,
        "physical_activity":     physical_activity,
        "stress_level":          stress_level,
        "hours_of_sleep":        hours_of_sleep,
        "sleep_quality":         sleep_quality,
    }

    with st.spinner("Calculating your exposure ratings ..."):
        try:
            results = score_all_diseases(user_inputs)
        except FileNotFoundError as e:
            st.error(
                f"**Models not found.** Please run `python src/train.py` first.\n\n{e}"
            )
            st.stop()

    st.markdown("---")
    st.markdown("## 📊 Your Exposure Ratings")

    for disease_name, data in results.items():
        label = data["risk_label"]
        prob  = data["probability"]
        color = RISK_COLORS[label]

        with st.container():
            st.markdown(f"### {disease_name}")

            # Risk badge + probability bar side-by-side
            badge_col, bar_col = st.columns([1, 3])
            with badge_col:
                st.markdown(risk_badge(label), unsafe_allow_html=True)
                st.markdown(
                    f'<p style="color:{color};font-size:0.9rem;margin-top:6px;">'
                    f'Score: {prob*100:.1f}%</p>',
                    unsafe_allow_html=True,
                )
            with bar_col:
                st.progress(prob)

            # Top contributing factors
            st.markdown("**🔑 Top Contributing Factors:**")
            for i, (feat_name, contribution) in enumerate(data["top_factors"], 1):
                direction = "↑ Increases" if contribution > 0 else "↓ Decreases"
                st.markdown(f"&nbsp;&nbsp;**{i}.** {feat_name} — *{direction} risk*")

            # Sleep score callout
            slp = data["sleep_score"]
            slp_label = "Good" if slp >= 0.6 else ("Fair" if slp >= 0.35 else "Poor")
            st.caption(
                f"😴 Sleep Quality Score used in this model: "
                f"**{slp*100:.0f}/100** ({slp_label})"
            )

            # Disease description
            st.info(DISEASE_DESCRIPTIONS.get(disease_name, ""))
            st.markdown("---")

    # Disclaimer
    st.markdown(DISCLAIMER)