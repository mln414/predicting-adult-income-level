"""Interactive Streamlit demonstration for Adult Income classifiers."""

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction_demo import (  # noqa: E402
    AVAILABLE_MODELS,
    UNAVAILABLE_MODELS,
    load_demo_bundle,
    transform_user_input,
)


st.set_page_config(
    page_title="Adult Income Prediction Demo",
    page_icon="📊",
    layout="wide",
)
st.title("Adult Income Prediction")
st.write(
    "Fill in the profile, choose a model, and select **Predict income** to see "
    "the model's estimate."
)


@st.cache_resource(show_spinner=False)
def get_demo_bundle():
    return load_demo_bundle()


try:
    with st.spinner("Preparing the dataset and loading trained models..."):
        bundle = get_demo_bundle()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

all_model_names = list(AVAILABLE_MODELS) + list(UNAVAILABLE_MODELS)


def model_status(name: str) -> str:
    if name in AVAILABLE_MODELS:
        return f"{name} ({AVAILABLE_MODELS[name]})"
    return f"{name} (not yet available)"


selected_model = st.selectbox(
    "Prediction model",
    all_model_names,
    format_func=model_status,
)

if selected_model in AVAILABLE_MODELS:
    member_id = AVAILABLE_MODELS[selected_model]
    model_json_path = PROJECT_ROOT / "results" / "model_results" / f"{member_id}.json"
    info_parts = [f"<strong>Student ID:</strong> <code>{member_id}</code>"]
    if model_json_path.exists():
        with open(model_json_path, "r", encoding="utf-8-sig") as f:
            res_data = json.load(f)
        metrics = res_data.get("metrics") or res_data.get("tuned") or {}
        for acc_key in ("Accuracy", "accuracy"):
            if acc_key in metrics:
                info_parts.append(f"<strong>Test Accuracy:</strong> {float(metrics[acc_key]):.2%}")
                break
        for f1_key in ("F1_Score", "f1_score"):
            if f1_key in metrics:
                info_parts.append(f"<strong>Test F1:</strong> {float(metrics[f1_key]):.4f}")
                break
        for auc_key in ("ROC_AUC", "roc_auc"):
            if auc_key in metrics:
                info_parts.append(f"<strong>ROC-AUC:</strong> {float(metrics[auc_key]):.4f}")
                break
    st.markdown(
        '<p style="color: gray; font-size: 0.85em;">' + " &nbsp;|&nbsp; ".join(info_parts) + "</p>",
        unsafe_allow_html=True,
    )

else:
    st.warning(
        f"{selected_model} is listed as a planned group model "
        f"({UNAVAILABLE_MODELS.get(selected_model, '')}), but its implementation "
        "is not available yet. Please select another model to continue."
    )

numeric_labels = {
    "age": "Age (years)",
    "fnlwgt": "Census record weight",
    "education.num": "Education level (1–16)",
    "capital.gain": "Capital gains ($)",
    "capital.loss": "Capital losses ($)",
    "hours.per.week": "Hours worked per week",
}
categorical_labels = {
    "workclass": "Workclass",
    "marital.status": "Marital status",
    "occupation": "Occupation",
    "relationship": "Relationship",
    "race": "Race",
    "sex": "Sex",
    "native.country": "Native country",
}

numeric_help = {
    "age": "Age in years.",
    "fnlwgt": "Census sampling weight for this record; it is not income.",
    "education.num": "Dataset education code, from 1 (lowest) to 16 (highest).",
    "capital.gain": "Reported capital gains in dollars.",
    "capital.loss": "Reported capital losses in dollars.",
    "hours.per.week": "Usual hours worked per week.",
}


def display_category(value: str) -> str:
    return value.replace("-", " ").replace("_", " ").title()


with st.form("adult_income_prediction", border=True):
    st.subheader("Profile")
    st.caption("Values and category options are based on the cleaned project dataset.")

    st.markdown("#### Work and finances")
    user_input = {}
    numeric_columns = st.columns(3)
    numeric_features = [
        "age",
        "fnlwgt",
        "education.num",
        "capital.gain",
        "capital.loss",
        "hours.per.week",
    ]
    for index, feature in enumerate(numeric_features):
        defaults = bundle["numeric_defaults"][feature]
        with numeric_columns[index % len(numeric_columns)]:
            user_input[feature] = st.number_input(
                numeric_labels[feature],
                min_value=defaults["min"],
                max_value=defaults["max"],
                value=defaults["value"],
                step=1.0,
                format="%.0f",
                help=(
                    f"{numeric_help[feature]} Observed range: "
                    f"{defaults['min']:,.0f}–{defaults['max']:,.0f}."
                ),
            )

    st.markdown("#### Work and education")
    work_columns = st.columns(2)
    for index, feature in enumerate(
        ["workclass", "occupation", "marital.status", "relationship"]
    ):
        choices = bundle["category_levels"][feature]
        default = bundle["categorical_defaults"][feature]
        with work_columns[index % len(work_columns)]:
            user_input[feature] = st.selectbox(
                categorical_labels[feature],
                choices,
                index=choices.index(default),
                format_func=display_category,
                key=f"input_{feature}",
            )

    st.markdown("#### Personal background")
    background_columns = st.columns(2)
    for index, feature in enumerate(["sex", "race", "native.country"]):
        choices = bundle["category_levels"][feature]
        default = bundle["categorical_defaults"][feature]
        with background_columns[index % len(background_columns)]:
            user_input[feature] = st.selectbox(
                categorical_labels[feature],
                choices,
                index=choices.index(default),
                format_func=display_category,
                key=f"input_{feature}",
            )

    submitted = st.form_submit_button(
        "Predict income",
        type="primary",
        use_container_width=True,
        disabled=selected_model not in AVAILABLE_MODELS,
    )

if submitted:
    model_input = transform_user_input(user_input, bundle)
    model = bundle["models"][selected_model]
    predicted_class = int(model.predict(model_input)[0])
    classes = list(getattr(model, "classes_", [0, 1]))

    if hasattr(model, "predict_proba"):
        pos_idx = classes.index(1) if 1 in classes else -1
        positive_probability = float(model.predict_proba(model_input)[0][pos_idx])
    elif hasattr(model, "decision_function"):
        decision = float(model.decision_function(model_input)[0])
        positive_probability = float(1.0 / (1.0 + np.exp(-np.clip(decision, -500, 500))))
    else:
        positive_probability = float(predicted_class)

    st.subheader("Prediction")
    cols = st.columns(2)
    with cols[0]:
        if predicted_class == 1:
            st.success("Predicted income group: **>50K**")
        else:
            st.info("Predicted income group: **<=50K**")
    with cols[1]:
        st.metric("Estimated probability of >50K", f"{positive_probability:.1%}")
