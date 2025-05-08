import streamlit as st
import pickle
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
import numpy as np
import os
import math
import gdown

# ------------------------- Utility Functions -------------------------

# Truncate float f to n decimal places without rounding
def truncate(f, n):
    if n == 0:
        return int(f)
    return math.floor(f * 10 ** n) / 10 ** n

# Convert grams per kilometer to tonnes per mile
def gpkm_to_tpm(gpkm):
    grams_per_tonne = 1_000_000
    kilometers_per_mile = 1.60934
    return (gpkm / grams_per_tonne) * kilometers_per_mile

# Label encode a column using a specified class order
def LabelEncode(input_df, column, order):
    le = preprocessing.LabelEncoder()
    le.fit(order)
    input_df[column] = le.transform(input_df[column])
    return input_df

# Download and load the trained model
def load_model():
    model_path = "Fuel Efficency estimator/RFR_model.pkl"
    if not os.path.exists(model_path):
        url = "https://drive.google.com/drive/folders/1efAbSVlSc8YIFFZ_9mAfZFyJsSRdD-jv"
        gdown.download_folder(url)
    with open(model_path, "rb") as f:
        return pickle.load(f)

# UI for collecting user inputs
def get_input_data():
    st.title("Fuel Consumption Estimator")
    st.subheader("Input car details and click Run")

    st.sidebar.title("Vehicle Inputs")
    year = st.sidebar.slider("Year", 2000, 2023)
    model = st.sidebar.radio("Drive Type", ["2WD", "4WD", "AWD"])
    size = st.sidebar.selectbox("Size", ['mini', 'sub', 'small', 'mid-size', 'standard', 'full-size', 'passenger', 'cargo', 'massive'])
    vehicle_class = st.sidebar.selectbox("Vehicle Class", [
        'compact', 'mid-size', 'station wagon', 'two-seater', 'full-size',
        'suv', 'van', 'pickup truck', 'minivan', 'special purpose vehicle'
    ])
    engine_size = st.sidebar.slider("Engine Size (L)", 0.9, 8.5, step=0.1)
    cylinders = st.sidebar.slider("Cylinder Count", 1, 16)
    fuel_type = st.sidebar.selectbox("Fuel Type", ['diesel', 'premium gasoline', 'gasoline', 'ethanol', 'natural gas'])
    make = st.sidebar.selectbox("Brand", [
        'acura', 'audi', 'bmw', 'buick', 'cadillac', 'chevrolet', 'chrysler', 'daewoo', 'dodge', 'ferrari',
        'ford', 'gmc', 'honda', 'hyundai', 'infiniti', 'isuzu', 'jaguar', 'jeep', 'kia', 'land rover',
        'lexus', 'lincoln', 'mazda', 'mercedes-benz', 'nissan', 'oldsmobile', 'plymouth', 'pontiac',
        'porsche', 'saab', 'saturn', 'subaru', 'suzuki', 'toyota', 'volkswagen', 'volvo', 'bentley',
        'rolls-royce', 'maserati', 'mini', 'mitsubishi', 'smart', 'hummer', 'aston martin', 'lamborghini',
        'bugatti', 'scion', 'fiat', 'ram', 'srt', 'alfa romeo', 'genesis'
    ])
    transmission = st.sidebar.radio("Transmission Type", ["A", "AM", "AS", "AV", "M"])
    gears_disabled = (transmission == "AV")
    gears = st.sidebar.slider("Number of Gears", 1, 10, disabled=gears_disabled)

    units = st.selectbox("Display Units", ['miles per gallon', 'liters per 100 kilometers'])
    decimal_places = st.slider("Decimal Places", 0, 10, value=4)

    data = {
        "year": year, "model": model, "size": size, "vehicle class": vehicle_class,
        "engine size": engine_size, "cylinders": cylinders, "fuel": fuel_type,
        "make": make, "transmission": transmission, "gears": gears
    }

    return pd.DataFrame(data, index=[0]), units, decimal_places

# Encode categorical variables using label encoding and one-hot encoding
def preprocess_input(input_df):
    input_df = LabelEncode(input_df, 'model', ['2WD', 'AWD', '4WD'])
    input_df = LabelEncode(input_df, "size", ['mini', 'sub', 'small', 'mid-size', 'standard', 'full-size', 'passenger', 'cargo', 'massive'])
    input_df = LabelEncode(input_df, "fuel", ['diesel', 'premium gasoline', 'gasoline', 'ethanol', 'natural gas'], value="gasoline")

    transmission_cats = [f"transmission type_{t}" for t in ["A", "AM", "AV", "M", "AS"]]
    make_cats = [f"make_{m}" for m in [
        'acura', 'audi', 'bmw', 'buick', 'cadillac', 'chevrolet', 'chrysler', 'daewoo', 'dodge', 'ferrari',
        'ford', 'gmc', 'honda', 'hyundai', 'infiniti', 'isuzu', 'jaguar', 'jeep', 'kia', 'land rover',
        'lexus', 'lincoln', 'mazda', 'mercedes-benz', 'nissan', 'oldsmobile', 'plymouth', 'pontiac',
        'porsche', 'saab', 'saturn', 'subaru', 'suzuki', 'toyota', 'volkswagen', 'volvo', 'bentley',
        'rolls-royce', 'maserati', 'mini', 'mitsubishi', 'smart', 'hummer', 'aston martin', 'lamborghini',
        'bugatti', 'scion', 'fiat', 'ram', 'srt', 'alfa romeo', 'genesis'
    ]]
    vehicle_cats = [f"vehicle_{v}" for v in [
        'compact', 'mid-size', 'station wagon', 'two-seater', 'full-size',
        'suv', 'van', 'pickup truck', 'minivan', 'special purpose vehicle'
    ]]

    onehot_T = pd.get_dummies(input_df["transmission"].astype(pd.CategoricalDtype(categories=transmission_cats)))
    onehot_M = pd.get_dummies(input_df["make"].astype(pd.CategoricalDtype(categories=make_cats)))
    onehot_V = pd.get_dummies(input_df["vehicle class"].astype(pd.CategoricalDtype(categories=vehicle_cats)))

    input_df.drop(["transmission", "make", "vehicle class"], axis=1, inplace=True)
    input_df = pd.concat([input_df, onehot_T, onehot_M, onehot_V], axis=1)

    final_columns = ['year', 'model', 'size', 'engine size', 'cylinders', 'gears', 'fuel'] + make_cats + vehicle_cats + transmission_cats
    return input_df[final_columns]

# ------------------------- App Logic -------------------------

model = load_model()
input_df, units, decimal_places = get_input_data()
run = st.button("Run")

# Initialize session state variables
st.session_state.setdefault('prediction', None)
st.session_state.setdefault('ran', False)

if run:
    st.session_state.ran = True
    with st.spinner("Running prediction..."):
        processed = preprocess_input(input_df)
        input_values = [processed.loc[0].values.tolist()]
        prediction = model.predict(input_values)
        if units == 'miles per gallon':
            prediction[0][0] = 235.215 / prediction[0][0]
            prediction[0][1] = 235.215 / prediction[0][1]
        st.session_state.prediction = prediction

# ------------------------- Output Display -------------------------

if st.session_state.ran:
    titles = ["City Fuel Consumption", "Highway Fuel Consumption", "Emissions (g/km)"]
    preds = [truncate(x, decimal_places) for x in st.session_state.prediction[0]]
    st.table(pd.DataFrame([preds], columns=titles))

    mileage = st.number_input("Trip Mileage Estimate", value=None)
    gas_price = st.number_input("Estimated Gas Price", value=None)
    tank_size = st.number_input("Gas Tank Size (gallons)", value=None)
    highway_ratio = st.slider("Highway Driving %", 0, 100, value=75) / 100

    if mileage:
        h_miles = highway_ratio * mileage
        r_miles = (1 - highway_ratio) * mileage
    if mileage and gas_price:
        total_cost = truncate(((h_miles / st.session_state.prediction[0][1]) + (r_miles / st.session_state.prediction[0][0])) * gas_price, 2)
    if mileage and tank_size:
        tank_refills = int(((h_miles / st.session_state.prediction[0][1]) + (r_miles / st.session_state.prediction[0][0])) / tank_size)

    if mileage:
        st.table(pd.DataFrame([[h_miles, r_miles]], columns=["Highway Miles", "City Miles"]))
    if mileage and gas_price and tank_size:
        st.table(pd.DataFrame([[total_cost, tank_refills]], columns=["Trip Cost", "Tank Refills"]))

    emissions_tpm = gpkm_to_tpm(st.session_state.prediction[0][2])
    st.write(f"### Totatl Emissions (tonnes): {truncate(emissions_tpm * milage, 10)}")
