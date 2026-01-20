import streamlit as st
import pandas as pd
import numpy as np
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# ===================== CONSTANTS =====================

MODEL_PARAMS = {
    "n_estimators": 1000,
    "max_features": 3,
    "max_depth": 6
}

DATA_COLUMNS = [
    "Gender", "Age", "BMI",
    "Duration", "Heart_Rate",
    "Body_Temp", "Calories"
]

NAVIGATION_PAGES = [
    "🏠 Welcome",
    "📝 User Input & Prediction",
    "📊 Analysis & Recommendations"
]


# ===================== DATA FUNCTIONS =====================

@st.cache_data
def load_and_prepare_data() -> pd.DataFrame:
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")

    df = exercise.merge(calories, on="User_ID")
    df.drop(columns=["User_ID"], inplace=True)

    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    df["BMI"] = df["BMI"].round(2)

    df = df[DATA_COLUMNS]
    df = pd.get_dummies(df, drop_first=True)

    return df


def split_data(df: pd.DataFrame):
    train, test = train_test_split(df, test_size=0.2, random_state=1)

    X_train = train.drop("Calories", axis=1)
    y_train = train["Calories"]

    X_test = test.drop("Calories", axis=1)
    y_test = test["Calories"]

    return X_train, X_test, y_train, y_test


@st.cache_resource
def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X, y)
    return model


# ===================== HELPER FUNCTIONS =====================

def calculate_bmi(weight: float, height: float) -> float:
    return round(weight / ((height / 100) ** 2), 2)


def percentile_message(df: pd.DataFrame, column: str, value: float) -> float:
    return round((df[column] < value).mean() * 100, 2)


def generate_recommendations(bmi, heart_rate, duration, body_temp):

    recommendations = []

    if bmi < 18.5:
        recommendations.append("🔹 Your BMI is low. Consider increasing protein and healthy calories.")
    elif bmi > 25:
        recommendations.append("⚠️ Your BMI is high. More cardio and balanced nutrition are recommended.")

    if heart_rate > 100:
        recommendations.append("🔴 Heart rate is high. Reduce intensity or consult a professional.")
    elif heart_rate < 70:
        recommendations.append("🟢 Heart rate is low. Consider increasing workout intensity.")

    if duration < 10:
        recommendations.append("🟠 Try exercising at least 30 minutes per session.")

    if body_temp > 39:
        recommendations.append("⚠️ High body temperature detected. Stay hydrated and rest.")

    return recommendations


# ===================== APP INITIALIZATION =====================

df = load_and_prepare_data()
X_train, X_test, y_train, y_test = split_data(df)
model = train_model(X_train, y_train)

st.sidebar.title("Navigation")
screen = st.sidebar.radio("Go to:", NAVIGATION_PAGES)


# ===================== PAGE 1 =====================

if screen == "🏠 Welcome":
    st.title("Welcome to the Personal Fitness Tracker 🏋️‍♂️")
    st.write("""
    This application predicts the **calories burned** based on your workout details.
    
    👉 Go to **User Input & Prediction** to start.
    """)


# ===================== PAGE 2 =====================

elif screen == "📝 User Input & Prediction":
    st.title("User Input & Prediction 🎯")

    st.sidebar.header("Enter Your Details")

    age = st.sidebar.slider("Age", 10, 100, 30)
    weight = st.sidebar.slider("Weight (kg)", 30, 150, 70)
    height = st.sidebar.slider("Height (cm)", 120, 220, 170)
    duration = st.sidebar.slider("Duration (min)", 0, 60, 15)
    heart_rate = st.sidebar.slider("Heart Rate", 50, 150, 80)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 35, 42, 37)
    gender = st.sidebar.radio("Gender", ["Male", "Female"])

    bmi = calculate_bmi(weight, height)
    gender_encoded = 1 if gender == "Male" else 0

    input_data = pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp],
        "Gender_male": [gender_encoded]
    })

    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

    st.write("### Your Input Details")
    st.dataframe(input_data)

    if st.button("Predict Calories Burned 🔥"):

        with st.spinner("Calculating prediction..."):
            time.sleep(1)

            prediction = model.predict(input_data)[0]

            st.session_state.prediction = prediction
            st.session_state.user_input = input_data

            st.success(
                f"🔥 Estimated Calories Burned: **{round(prediction, 2)} kcal**"
            )


# ===================== PAGE 3 =====================

elif screen == "📊 Analysis & Recommendations":

    st.title("Analysis & Personalized Recommendations 📈")

    if "prediction" not in st.session_state:
        st.warning("Please make a prediction first!")
        st.stop()

    prediction = st.session_state.prediction
    user_data = st.session_state.user_input.copy()

    st.write("### 🔍 Similar Exercise Records")

    similar = df[
        (df["Calories"] >= prediction - 10) &
        (df["Calories"] <= prediction + 10)
    ]

    display = similar.copy()
    display["Gender"] = display["Gender_male"].apply(
        lambda x: "Male" if x == 1 else "Female"
    )
    display.drop(columns=["Gender_male"], inplace=True)

    st.dataframe(display.sample(min(5, len(display))))

    st.write("---")
    st.write("### 📊 Comparison With Other Users")

    age = user_data["Age"].values[0]
    duration = user_data["Duration"].values[0]
    heart_rate = user_data["Heart_Rate"].values[0]
    body_temp = user_data["Body_Temp"].values[0]

    st.write(f"You are older than **{percentile_message(df, 'Age', age)}%** of users.")
    st.write(f"Your workout duration is higher than **{percentile_message(df, 'Duration', duration)}%** of users.")
    st.write(f"Your heart rate is higher than **{percentile_message(df, 'Heart_Rate', heart_rate)}%** of users.")
    st.write(f"Your body temperature is higher than **{percentile_message(df, 'Body_Temp', body_temp)}%** of users.")

    st.write("---")
    st.write("### 💡 Personalized Recommendations")

    recs = generate_recommendations(
        bmi=user_data["BMI"].values[0],
        heart_rate=heart_rate,
        duration=duration,
        body_temp=body_temp
    )

    if recs:
        for r in recs:
            st.write(r)
    else:
        st.success("✅ Everything looks great. Keep up the good work!")
