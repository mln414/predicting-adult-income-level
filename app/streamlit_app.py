"""Interactive Streamlit demonstration for Adult Income classifiers."""

import sys
from pathlib import Path

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
    with st.spinner("Preparing the dataset and training available models..."):
        bundle = get_demo_bundle()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

all_model_names = list(AVAILABLE_MODELS) + list(UNAVAILABLE_MODELS)


def model_status(name: str) -> str:
    if name in AVAILABLE_MODELS:
        return name
    return f"{name} (not yet available)"


selected_model = st.selectbox(
    "Prediction model",
    all_model_names,
    format_func=model_status,
)

if selected_model not in AVAILABLE_MODELS:
    st.warning(
        f"{selected_model} is listed as a planned group model "
        f"({UNAVAILABLE_MODELS[selected_model]}), but its implementation notebook "
        "is not available in this project yet. Choose KNN, Logistic Regression, "
        "or Decision Tree to continue."
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
    classes = list(model.classes_)
    positive_probability = float(
        model.predict_proba(model_input)[0][classes.index(1)]
    )
    st.subheader("Prediction")
    if predicted_class == 1:
        st.success("Predicted income group: **>50K**")
    else:
        st.info("Predicted income group: **<=50K**")
    st.metric("Estimated probability of >50K", f"{positive_probability:.1%}")
