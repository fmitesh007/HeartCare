
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="HeartCare | Heart Health",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# 2. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "heartcare_model.joblib"
INFO_PATH = BASE_DIR / "models" / "model_info.json"
COMPARISON_PATH = BASE_DIR / "models" / "model_comparison.csv"


# =========================================================
# 3. CUSTOM WEBSITE DESIGN
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f9fc;
    }

    [data-testid="stHeader"] {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        background: linear-gradient(120deg, #12304a, #176b76);
        color: white;
        padding: 35px 32px;
        border-radius: 20px;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        font-size: 38px;
        margin-bottom: 10px;
    }

    .hero p {
        color: #e5f3f5;
        font-size: 16px;
        line-height: 1.7;
    }

    .section-card {
        background: white;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #e6ebf1;
        margin-bottom: 18px;
    }

    .metric-card {
        background: white;
        border: 1px solid #e6ebf1;
        padding: 20px;
        border-radius: 14px;
        min-height: 120px;
    }

    .small-muted {
        color: #64748b;
        font-size: 14px;
    }

    div.stButton > button[kind="primary"] {
        background: #176b76;
        border: none;
        border-radius: 9px;
        min-height: 45px;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #125761;
        border: none;
    }

    h1, h2, h3 {
        color: #17324d;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model was not found. "
            "Run python train_model.py first."
        )

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_model_info():
    if not INFO_PATH.exists():
        raise FileNotFoundError(
            "model_info.json is missing. "
            "Please train the model again."
        )

    with open(INFO_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_comparison():
    if not COMPARISON_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(COMPARISON_PATH)


try:
    model = load_model()
    model_info = load_model_info()
    comparison_df = load_comparison()

except Exception as error:
    st.error(
        "HeartCare could not load its trained model. "
        "Please check that training completed successfully."
    )
    st.code(str(error))
    st.stop()


# =========================================================
# 5. NAVIGATION
# =========================================================

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px;">
        <div style="font-size:36px;">❤️</div>
        <div>
            <div style="font-size:27px; font-weight:800; color:#17324d;">
                HeartCare
            </div>
            <div style="font-size:13px; color:#64748b;">
                Heart Health Assessment & Lifestyle Insights
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

pages = ["Home", "About", "Health Test", "Heart Health Tips"]
requested_page = st.session_state.pop("heartcare_page_request", "Home")
default_index = pages.index(requested_page) if requested_page in pages else 0

page = st.radio(
    "Navigation",
    pages,
    index=default_index,
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()


# =========================================================
# 6. HOME PAGE
# =========================================================

if page == "Home":

    st.markdown(
        """
        <div class="hero">
            <h1>Understand Your Heart Health ❤️</h1>
            <p>
                Take a guided heart health assessment, explore
                important health factors, and view an educational
                machine-learning classification based on clinical
                measurements.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <h3>🩺 Health Assessment</h3>
                <p>Enter your health measurements in one form.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <h3>🤖 ML Analysis</h3>
                <p>Explore a classification from a trained ML model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <h3>🌿 Lifestyle Insights</h3>
                <p>Review habits and learn about heart-health prevention.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("How HeartCare works")

    st.markdown(
        """
        1. **Complete the assessment:** Enter the required clinical
           measurements and optional lifestyle details.
        2. **View the result:** The trained model classifies the
           clinical inputs using patterns from its training dataset.
        3. **Explore your insights:** Review the measurements you
           reported and general heart-health guidance.
        """
    )

    st.info(
        "The ML model requires clinical measurements. "
        "If you do not know these values, do not guess them. "
        "You can still explore the information and tips on this website."
    )

    if st.button("Start Health Assessment", type="primary"):
        st.session_state["heartcare_page_request"] = "Health Test"
        st.rerun()


# =========================================================
# 7. ABOUT PAGE
# =========================================================

elif page == "About":

    st.title("About HeartCare")

    st.write(
        """
        HeartCare is a student machine-learning project designed
        to demonstrate how a trained classification model can
        analyze clinical measurements from a public heart-disease
        dataset.
        """
    )

    st.subheader("Project objectives")

    st.markdown(
        """
        - Develop a simple healthcare assessment website.
        - Train and compare supervised ML algorithms.
        - Present model results in an understandable dashboard.
        - Explain relevant health measurements and lifestyle factors.
        - Encourage informed conversations with healthcare professionals.
        """
    )

    st.subheader("Technology used")

    tech1, tech2, tech3 = st.columns(3)

    tech1.markdown("**Python**\n\nProgramming language")
    tech2.markdown("**Scikit-learn**\n\nMachine learning")
    tech3.markdown("**Streamlit**\n\nInteractive website")

    st.subheader("Dataset")

    st.write(
        "The model uses the Cleveland subset of the UCI Heart Disease "
        "dataset, containing 303 records. The binary target groups "
        "dataset records into target 0 and targets 1–4."
    )

    st.markdown(
        "[Open the official UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)"
    )

    st.warning(
        "This educational model has not been clinically validated. "
        "It does not establish an individual's actual heart health "
        "or predict their future cardiovascular risk."
    )


# =========================================================
# 8. HEALTH TEST PAGE
# =========================================================

elif page == "Health Test":

    st.title("Heart Health Assessment")

    st.write(
        "Complete the form using your actual clinical measurements "
        "where available. The clinical inputs are required because "
        "they are the features used to train this model."
    )

    st.info(
        "Do not guess cholesterol, ECG, blood pressure or other "
        "clinical values. Use a reliable report or measurements "
        "provided by a healthcare professional."
    )

    with st.form("heartcare_assessment"):

        st.subheader("1. Personal Information")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age (years)",
                min_value=18,
                max_value=120,
                value=None,
                placeholder="Enter age",
                help="The model was trained on adult clinical records.",
            )

        with col2:
            sex_label = st.selectbox(
                "Sex recorded in the dataset",
                ["Select", "Female", "Male"],
                help="This is the sex encoding used in the source dataset.",
            )

        st.divider()
        st.subheader("2. Clinical Measurements")

        col1, col2 = st.columns(2)

        with col1:
            cp_label = st.selectbox(
                "Chest pain category",
                [
                    "Select",
                    "Typical angina",
                    "Atypical angina",
                    "Non-anginal pain",
                    "Asymptomatic",
                ],
                help=(
                    "Select the category recorded by a healthcare "
                    "professional. Symptoms alone cannot establish this category."
                ),
            )

            trestbps = st.number_input(
                "Resting blood pressure (mm Hg)",
                min_value=50,
                max_value=260,
                value=None,
                placeholder="Enter measured value",
            )

            chol = st.number_input(
                "Serum cholesterol (mg/dL)",
                min_value=50,
                max_value=700,
                value=None,
                placeholder="Enter lab result",
            )

            fbs_label = st.selectbox(
                "Fasting blood sugar above 120 mg/dL?",
                ["Select", "No", "Yes"],
            )

            restecg_label = st.selectbox(
                "Resting ECG result",
                [
                    "Select",
                    "Normal",
                    "ST-T wave abnormality",
                    "Left ventricular hypertrophy",
                ],
                help="Use the result from an ECG report.",
            )

            ca_label = st.selectbox(
                "Major vessels observed by fluoroscopy",
                ["Select", "0", "1", "2", "3"],
                help=(
                    "This is a clinical field from the original dataset. "
                    "Use a recorded result; do not estimate it."
                ),
            )

        with col2:
            thalach = st.number_input(
                "Maximum heart rate recorded during exercise",
                min_value=50,
                max_value=250,
                value=None,
                placeholder="Enter recorded value",
                help="Use a clinical exercise-test result.",
            )

            exang_label = st.selectbox(
                "Exercise-induced angina recorded?",
                ["Select", "No", "Yes"],
                help="Use the recorded clinical test result.",
            )

            oldpeak = st.number_input(
                "ST depression (oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=None,
                step=0.1,
                placeholder="Enter recorded value",
                help="Use the value reported by the clinical exercise test.",
            )

            slope_label = st.selectbox(
                "Slope of peak-exercise ST segment",
                [
                    "Select",
                    "Upsloping",
                    "Flat",
                    "Downsloping",
                ],
                help="Use the exercise-test report.",
            )

            thal_label = st.selectbox(
                "Thal test result",
                [
                    "Select",
                    "Normal",
                    "Fixed defect",
                    "Reversible defect",
                ],
                help="Use the recorded clinical result.",
            )

        st.divider()
        st.subheader("3. Additional Health & Lifestyle Details")

        st.caption(
            "These details are displayed separately. They are not "
            "used by the trained model because they were not included "
            "as features in the training dataset."
        )

        col1, col2 = st.columns(2)

        with col1:
            smoking = st.selectbox(
                "Smoking status",
                ["Not provided", "Never", "Former", "Current"],
            )

            activity = st.selectbox(
                "Physical activity",
                [
                    "Not provided",
                    "Low",
                    "Moderate",
                    "High",
                ],
            )

        with col2:
            family_history = st.selectbox(
                "Family history of heart disease",
                ["Not provided", "No", "Yes", "Unsure"],
            )

            known_bp_history = st.selectbox(
                "History of high blood pressure",
                ["Not provided", "No", "Yes", "Unsure"],
            )

        submitted = st.form_submit_button(
            "Analyze My Assessment",
            type="primary",
            use_container_width=True,
        )

    # -----------------------------------------------------
    # 9. VALIDATE AND PREDICT
    # -----------------------------------------------------

    if submitted:

        missing_fields = []

        required_values = {
            "Age": age,
            "Resting blood pressure": trestbps,
            "Cholesterol": chol,
            "Maximum exercise heart rate": thalach,
            "ST depression": oldpeak,
        }

        for label, value in required_values.items():
            if value is None:
                missing_fields.append(label)

        selections = {
            "Sex": sex_label,
            "Chest pain category": cp_label,
            "Fasting blood sugar": fbs_label,
            "Resting ECG": restecg_label,
            "Exercise-induced angina": exang_label,
            "ST segment slope": slope_label,
            "Major vessels": ca_label,
            "Thal test result": thal_label,
        }

        for label, value in selections.items():
            if value == "Select":
                missing_fields.append(label)

        if missing_fields:
            st.error(
                "Please complete these required fields: "
                + ", ".join(missing_fields)
            )

        else:

            # Map human-readable choices to the dataset codes.
            sex_map = {
                "Female": 0,
                "Male": 1,
            }

            cp_map = {
                "Typical angina": 1,
                "Atypical angina": 2,
                "Non-anginal pain": 3,
                "Asymptomatic": 4,
            }

            fbs_map = {
                "No": 0,
                "Yes": 1,
            }

            restecg_map = {
                "Normal": 0,
                "ST-T wave abnormality": 1,
                "Left ventricular hypertrophy": 2,
            }

            exang_map = {
                "No": 0,
                "Yes": 1,
            }

            slope_map = {
                "Upsloping": 1,
                "Flat": 2,
                "Downsloping": 3,
            }

            thal_map = {
                "Normal": 3,
                "Fixed defect": 6,
                "Reversible defect": 7,
            }

            # Validate clinical values.
            # This prevents obviously implausible values but
            # does not validate clinical accuracy.
            problems = []

            if not 18 <= age <= 100:
                problems.append("Age must be between 18 and 100.")

            if not 80 <= trestbps <= 250:
                problems.append(
                    "Check the resting blood pressure value."
                )

            if not 80 <= chol <= 600:
                problems.append(
                    "Check the cholesterol value."
                )

            if not 60 <= thalach <= 220:
                problems.append(
                    "Check the maximum heart rate value."
                )

            if problems:
                for problem in problems:
                    st.error(problem)

            else:

                # Keep the feature names and ordering consistent
                # with train_model.py.
                input_data = pd.DataFrame(
                    [
                        {
                            "age": float(age),
                            "sex": float(sex_map[sex_label]),
                            "cp": float(cp_map[cp_label]),
                            "trestbps": float(trestbps),
                            "chol": float(chol),
                            "fbs": float(fbs_map[fbs_label]),
                            "restecg": float(
                                restecg_map[restecg_label]
                            ),
                            "thalach": float(thalach),
                            "exang": float(exang_map[exang_label]),
                            "oldpeak": float(oldpeak),
                            "slope": float(slope_map[slope_label]),
                            "ca": float(ca_label),
                            "thal": float(thal_map[thal_label]),
                        }
                    ],
                    columns=model_info["features"],
                )

                try:
                    # Get the predicted class and its model-estimated probability.
                    prediction = int(model.predict(input_data)[0])

                    if not hasattr(model, "predict_proba"):
                        raise AttributeError(
                            "The saved model does not support probability output. "
                            "Please retrain using the Logistic Regression pipeline."
                        )

                    probabilities = model.predict_proba(input_data)[0]
                    model_classes = list(model.classes_)
                    positive_class_index = model_classes.index(1)
                    positive_probability = float(
                        probabilities[positive_class_index]
                    )

                    st.session_state["heartcare_result"] = {
                        "prediction": prediction,
                        "positive_probability": positive_probability,
                        "input": input_data.to_dict(
                            orient="records"
                        )[0],
                        "lifestyle": {
                            "smoking": smoking,
                            "activity": activity,
                            "family_history": family_history,
                            "known_bp_history": known_bp_history,
                        },
                    }

                    st.success(
                        "Assessment completed. View the result below."
                    )

                except Exception as error:
                    st.error(
                        "The assessment could not be calculated. "
                        "Please check the saved model and its input features."
                    )
                    st.code(str(error))

    # -----------------------------------------------------
    # 10. DISPLAY RESULT
    # -----------------------------------------------------

    result = st.session_state.get("heartcare_result")

    if result is not None:

        st.divider()
        st.title("Your Assessment Result")

        prediction = result["prediction"]
        positive_probability = result["positive_probability"]
        score_percent = round(positive_probability * 100)

        st.subheader("Your HeartCare Assessment")

        score_col, result_col = st.columns([1, 1])

        with score_col:
            st.markdown("#### Model-estimated probability")
            st.metric(
                label="Probability of the dataset's positive class",
                value=f"{score_percent}%",
                help=(
                    "This is the saved model's estimated probability for "
                    "the positive class, not a clinically validated personal "
                    "risk score or a prediction of future disease."
                ),
            )
            st.progress(positive_probability)
            st.caption(
                "The bar visualizes the model probability. It is not a "
                "medical risk scale and its thresholds are not clinically validated."
            )

        with result_col:
            st.markdown("#### Model classification")
            if prediction == 1:
                st.warning(
                    "**Positive class**\n\n"
                    "The input was classified with the dataset's positive label. "
                    "This does not confirm that you have heart disease."
                )
            else:
                st.info(
                    "**Negative class**\n\n"
                    "The input was classified with the dataset's negative label. "
                    "This does not rule out heart disease or establish that your "
                    "heart is healthy."
                )

        st.caption(
            "Important: this model was trained on a small public dataset and "
            "has not been clinically validated. Treat the percentage as an "
            "educational model output only, not as your actual chance of having "
            "or developing heart disease."
        )

        st.subheader("Your submitted measurements")

        input_df = pd.DataFrame(
            [
                {
                    "Measurement": "Age",
                    "Value": f"{result['input']['age']:.0f} years",
                },
                {
                    "Measurement": "Resting blood pressure",
                    "Value": (
                        f"{result['input']['trestbps']:.0f} mm Hg"
                    ),
                },
                {
                    "Measurement": "Cholesterol",
                    "Value": f"{result['input']['chol']:.0f} mg/dL",
                },
                {
                    "Measurement": "Maximum exercise heart rate",
                    "Value": f"{result['input']['thalach']:.0f}",
                },
                {
                    "Measurement": "ST depression",
                    "Value": f"{result['input']['oldpeak']:.1f}",
                },
            ]
        )

        st.dataframe(
            input_df,
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("Lifestyle information")

        lifestyle = result["lifestyle"]

        lcol1, lcol2 = st.columns(2)

        with lcol1:
            st.markdown("**Smoking status**")
            st.write(lifestyle["smoking"])

            st.markdown("**Physical activity**")
            st.write(lifestyle["activity"])

        with lcol2:
            st.markdown("**Family history**")
            st.write(lifestyle["family_history"])

            st.markdown("**History of high blood pressure**")
            st.write(lifestyle["known_bp_history"])

        st.subheader("General heart-health guidance")

        st.markdown(
            """
            - Avoid smoking and tobacco products.
            - Stay physically active within your abilities.
            - Follow a balanced eating pattern.
            - Have blood pressure and cholesterol checked as recommended.
            - Discuss family history and concerning measurements with a clinician.
            """
        )

        st.warning(
            "HeartCare is an educational project, not a medical "
            "device. The dataset is small and the model has not been "
            "clinically validated. Do not use this result to make "
            "treatment decisions or delay medical care."
        )

        if st.button("Clear Assessment"):
            st.session_state.pop("heartcare_result", None)
            st.rerun()


# =========================================================
# 11. HEART HEALTH TIPS PAGE
# =========================================================

elif page == "Heart Health Tips":

    st.title("Everyday Heart Health Tips")

    tips = [
        (
            "🥗 Balanced eating",
            "Eat a variety of vegetables, fruits, whole grains, "
            "and suitable protein sources. Limit excess salt and "
            "highly processed foods."
        ),
        (
            "🚶 Regular activity",
            "Choose regular physical activity appropriate for your "
            "health and abilities."
        ),
        (
            "🚭 Avoid tobacco",
            "Avoid smoking and other tobacco products. Seek support "
            "if you want help stopping."
        ),
        (
            "🩺 Routine check-ups",
            "Monitor blood pressure and cholesterol as recommended "
            "by a healthcare professional."
        ),
        (
            "😴 Healthy sleep",
            "Keep a consistent sleep schedule and discuss persistent "
            "sleep problems with a healthcare professional."
        ),
        (
            "🧘 Manage stress",
            "Consider relaxing activities, social support and "
            "healthy routines."
        ),
    ]

    for title, description in tips:
        with st.container(border=True):
            st.subheader(title)
            st.write(description)

    st.error(
        "If you experience chest pain, severe breathlessness, "
        "fainting, or other potentially serious symptoms, seek "
        "urgent medical care. Do not rely on this website to assess "
        "an emergency."
    )


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:13px;">
        <b>HeartCare</b> · Heart Health Assessment & Lifestyle Insights
        <br><br>
        Educational machine-learning project · Not a medical diagnosis
    </div>
    """,
    unsafe_allow_html=True,
)
