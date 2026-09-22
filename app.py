import streamlit as st
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model




st.set_page_config(
    page_title="Diabetes Prediction Using ANN",
    page_icon="🩺",
    layout="centered"
)




model = load_model("diabetes_ann_model.keras")

imputer = joblib.load("imputer.pkl")

scaler = joblib.load("scaler.pkl")



st.title("🩺 Diabetes Prediction Using ANN")

st.write(
    "Artificial Neural Network Based "
    "Diabetes Prediction System"
)

st.markdown("---")




st.subheader("Enter Patient Details")


pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)

glucose = st.number_input(
    "Glucose",
    min_value=0.0,
    max_value=300.0,
    value=120.0
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0.0,
    max_value=100.0,
    value=20.0
)

insulin = st.number_input(
    "Insulin",
    min_value=0.0,
    max_value=900.0,
    value=80.0
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=80.0,
    value=30.0
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.50,
    step=0.01
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)




if st.button(
    "🔍 Predict Diabetes",
    use_container_width=True
):

    input_data = pd.DataFrame({

        "Pregnancies": [pregnancies],

        "Glucose": [glucose],

        "BloodPressure": [blood_pressure],

        "SkinThickness": [skin_thickness],

        "Insulin": [insulin],

        "BMI": [bmi],

        "DiabetesPedigreeFunction": [
            diabetes_pedigree
        ],

        "Age": [age]
    })



    zero_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    input_data[zero_columns] = (
        input_data[zero_columns]
        .replace(0, np.nan)
    )


   

    input_data = imputer.transform(
        input_data
    )



    input_data = scaler.transform(
        input_data
    )



    probability = model.predict(
        input_data,
        verbose=0
    )[0][0]

    prediction = int(
        probability >= 0.5
    )

    st.markdown("---")

    st.subheader("Prediction Result")

    st.metric(
        "Diabetes Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:

        st.error(
            "⚠️ Higher predicted likelihood "
            "of diabetes"
        )

    else:

        st.success(
            "✅ Lower predicted likelihood "
            "of diabetes"
        )

    st.progress(
        float(probability)
    )

    st.info(
        "This is an academic machine-learning "
        "project and is not a medical diagnosis."
    )