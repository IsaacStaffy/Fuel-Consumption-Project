import streamlit as st
import pickle
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn import preprocessing
import numpy as np
import os
import math

def truncate(f, n):
    if n == 0:
      return int(f)
    return math.floor(f * 10 ** n) / 10 ** n
#import models for streamlit

import gdown

# a file
if not os.path.exists("Fuel Efficency estimator/RFR_model.pkl"):
  url = "https://drive.google.com/drive/folders/1efAbSVlSc8YIFFZ_9mAfZFyJsSRdD-jv"
  gdown.download_folder(url)

with open("Fuel Efficency estimator/RFR_model.pkl", "rb") as f:
  clf = pickle.load(f)

#label encoding function
def LabelEncode(column, order):
  label_encoder.fit(input[column])
  label_encoder.classes_ = np.array(order)
  input[column] = label_encoder.transform(input[column])
  print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

#ui

st.title("Fuel consumption estimator")
st.subheader("Input stats for your car and click run to estimate fuel consumption")
st.sidebar.title("Inputs")

def ui():
  year = st.sidebar.slider("Year", 2000, 2023)
  model = st.sidebar.pills("Drive type", ["2WD", "4WD", "AWD"], default="2WD")
  size = st.sidebar.selectbox("Size", ['mini', 'sub', 'small', 'mid-size', 'standard', 'full-size', 'passenger', 'cargo', 'massive'])
  vehicle_class = st.sidebar.selectbox("Vehicle class", ['compact', 'mid-size', 'station wagon', 'two-seater', 'full-size',
        'suv', 'van', 'pickup truck', 'minivan', 'special purpose vehicle'])
  engine_size = st.sidebar.slider("Engine size", 0.9, 8.5, step=0.1)
  cylinders = st.sidebar.slider("Cylinder count", 1, 16)
  fuel_type = st.sidebar.selectbox("Fuel type", ['diesel', 'premium gasoline', 'gasoline', 'ethanol', 'natural gas'])
  make = st.sidebar.selectbox("Brand of vehicle", ['acura', 'audi', 'bmw', 'buick', 'cadillac', 'chevrolet',
        'chrysler', 'daewoo', 'dodge', 'ferrari', 'ford', 'gmc', 'honda',
        'hyundai', 'infiniti', 'isuzu', 'jaguar', 'jeep', 'kia',
        'land rover', 'lexus', 'lincoln', 'mazda', 'mercedes-benz',
        'nissan', 'oldsmobile', 'plymouth', 'pontiac', 'porsche', 'saab',
        'saturn', 'subaru', 'suzuki', 'toyota', 'volkswagen', 'volvo',
        'bentley', 'rolls-royce', 'maserati', 'mini', 'mitsubishi',
        'smart', 'hummer', 'aston martin', 'lamborghini', 'bugatti',
        'scion', 'fiat', 'ram', 'srt', 'alfa romeo', 'genesis'])
  transmission = st.sidebar.pills("Transmission type", ["A", "AM", "AS", "AV", "M"], default="A")
  if transmission == "AV":
    gears_bool = True
    gears = 20
  else:
    gears_bool = False
  gears = st.sidebar.slider("Gear count", 1, 10, disabled=gears_bool)
  data =  {"year" : year, "model" : model, "size" : size, "vehicle class" : vehicle_class, "engine size" : engine_size, "cylinders" : cylinders, "fuel" : fuel_type,
           "make" : make, "transmission" : transmission, "gears" : gears}
  units = st.selectbox('Units ' , ['miles per gallon' , 'liters per 100 kilometers'])
  decimal_places = st.slider("Decimal places", 0, 10, value=1)
  #model_type = st.selectbox("Select a model to use", ["Random forest regressor", "Neural network"])


  return pd.DataFrame(data, index=[0]), units, decimal_places



input, units, decimal_places = ui()

run = st.button("Run")



if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None

if 'ran' not in st.session_state:
    st.session_state['ran'] = False

if run:
  st.session_state.ran = True
  with st.spinner("running..."):
    # encoding
    # label encoding
    label_encoder = preprocessing.LabelEncoder()



    LabelEncode('model', ['2WD', 'AWD', '4WD'])
    LabelEncode("size", ['mini', 'sub', 'small', 'mid-size', 'standard', 'full-size', 'passenger', 'cargo', 'massive'])
    LabelEncode("fuel", ['diesel', 'premium gasoline', 'gasoline', 'ethanol', 'natural gas'])

    transmission_catagories = ["A", "AM", "AV", "M", "AS"]
    make_catagories = ['acura', 'audi', 'bmw', 'buick', 'cadillac', 'chevrolet',
          'chrysler', 'daewoo', 'dodge', 'ferrari', 'ford', 'gmc', 'honda',
          'hyundai', 'infiniti', 'isuzu', 'jaguar', 'jeep', 'kia',
          'land rover', 'lexus', 'lincoln', 'mazda', 'mercedes-benz',
          'nissan', 'oldsmobile', 'plymouth', 'pontiac', 'porsche', 'saab',
          'saturn', 'subaru', 'suzuki', 'toyota', 'volkswagen', 'volvo',
          'bentley', 'rolls-royce', 'maserati', 'mini', 'mitsubishi',
          'smart', 'hummer', 'aston martin', 'lamborghini', 'bugatti',
          'scion', 'fiat', 'ram', 'srt', 'alfa romeo', 'genesis']
    vehicle_class_catagories = ['compact', 'mid-size', 'station wagon', 'two-seater', 'full-size',
          'suv', 'van', 'pickup truck', 'minivan', 'special purpose vehicle']

    #add keywords to colum names to pass into model
    i = 0
    for value in transmission_catagories:
      transmission_catagories[i] = "transmission type_" + value
      i += 1

    i = 0
    for value in make_catagories:
      make_catagories[i] = "make_" + value
      i += 1

    i = 0
    for value in vehicle_class_catagories:
      vehicle_class_catagories[i] = "vehicle_" + value
      i += 1

    # onehot encoding

    onehot_T = pd.get_dummies(input["transmission"].astype(pd.CategoricalDtype(categories=transmission_catagories)))
    onehot_M = pd.get_dummies(input["make"].astype(pd.CategoricalDtype(categories=make_catagories)))
    onehot_V = pd.get_dummies(input["vehicle class"].astype(pd.CategoricalDtype(categories=vehicle_class_catagories)))

    _ = input.drop("transmission", axis=1, inplace=True)
    _ = input.drop("make", axis=1, inplace=True)
    _ = input.drop("vehicle class", axis=1, inplace=True)
    input = pd.concat([input, onehot_T], axis=1)
    input = pd.concat([input, onehot_M], axis=1)
    input = pd.concat([input, onehot_V], axis=1)
    input = input[['year',
  'model',
  'size',
  'engine size',
  'cylinders',
  'gears',
  'fuel',
  'make_acura',
  'make_alfa romeo',
  'make_aston martin',
  'make_audi',
  'make_bentley',
  'make_bmw',
  'make_bugatti',
  'make_buick',
  'make_cadillac',
  'make_chevrolet',
  'make_chrysler',
  'make_daewoo',
  'make_dodge',
  'make_ferrari',
  'make_fiat',
  'make_ford',
  'make_genesis',
  'make_gmc',
  'make_honda',
  'make_hummer',
  'make_hyundai',
  'make_infiniti',
  'make_isuzu',
  'make_jaguar',
  'make_jeep',
  'make_kia',
  'make_lamborghini',
  'make_land rover',
  'make_lexus',
  'make_lincoln',
  'make_maserati',
  'make_mazda',
  'make_mercedes-benz',
  'make_mini',
  'make_mitsubishi',
  'make_nissan',
  'make_oldsmobile',
  'make_plymouth',
  'make_pontiac',
  'make_porsche',
  'make_ram',
  'make_rolls-royce',
  'make_saab',
  'make_saturn',
  'make_scion',
  'make_smart',
  'make_srt',
  'make_subaru',
  'make_suzuki',
  'make_toyota',
  'make_volkswagen',
  'make_volvo',
  'vehicle_compact',
  'vehicle_full-size',
  'vehicle_mid-size',
  'vehicle_minivan',
  'vehicle_pickup truck',
  'vehicle_special purpose vehicle',
  'vehicle_station wagon',
  'vehicle_suv',
  'vehicle_two-seater',
  'vehicle_van',
  'transmission type_A',
  'transmission type_AM',
  'transmission type_AS',
  'transmission type_AV',
  'transmission type_M']]

    input = input.loc[0].values.tolist()
    input = [input]

    st.session_state.prediction = clf.predict(input)
    #else:
    #  st.session_state.predictions = []
    #  st.write(input)
    #  st.session_state.predictions.append(model1.predict(np.array(input[0])))
    #  st.session_state.predictions.append(model2.predict(np.array(input[0])))
    #  st.session_state.predictions.append(model3.predict(np.array(input[0])))

  if units == 'miles per gallon':
    st.session_state.prediction[0][0] = 235.215 / st.session_state.prediction[0][0]
    st.session_state.prediction[0][1] = 235.215 / st.session_state.prediction[0][1]

if st.session_state.ran == True:

  titles = ["City road fuel consumption", "Highway fuel consumption", "Emmisions (grams per km)"]

  predictions = [truncate(st.session_state.prediction[0][0], decimal_places), truncate(st.session_state.prediction[0][1], decimal_places), truncate(st.session_state.prediction[0][2], decimal_places)]
  
  output = pd.DataFrame([predictions], columns=titles)

  st.table(output)

  #st.write("City road fuel consumption")
  #st.write(truncate(st.session_state.prediction[0][0], decimal_places))
  #st.write("Highway fuel consumption")
  #st.write(truncate(st.session_state.prediction[0][1], decimal_places))
  #st.write("emmisions (grams per km)")
  #st.write(truncate(st.session_state.prediction[0][2], decimal_places))

  milage = st.number_input("Insert an estimated milage for a trip", value=None, placeholder="Insert estimated milage")
  gas_price = st.number_input("Insert estimated price for gas in your area", value=None, placeholder="Insert estimated gas price")
  tank_size = st.number_input("Insert the size in gallons of your gas tank", value=None, placeholder="Insert size here (gallons)")
  road_type_slider = st.slider("Precentage of distance driven on the highway", 0, 100, value=75)

  if milage and gas_price and tank_size:
    highway_milage = (road_type_slider * (0.01)) * milage
    road_milage = (1 - (road_type_slider * (0.01))) * milage

    total_cost = truncate(((highway_milage / st.session_state.prediction[0][1]) + (road_milage / st.session_state.prediction[0][0])) * gas_price, 2)

    tank_refills = int(((highway_milage / st.session_state.prediction[0][1]) + (road_milage / st.session_state.prediction[0][0])) / tank_size)

    titles = ["Milage on highway", "Milage on roads"]
    numbers = [highway_milage, road_milage]

    output = pd.DataFrame([numbers], columns=titles)

    st.table(output)

    titles = ["Total trip cost", "Minimum tank refills"]
    numbers = [total_cost, tank_refills]

    output = pd.DataFrame([numbers], columns=titles)

    st.table(output)
