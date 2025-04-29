import streamlit as st
import pickle
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn import preprocessing
import numpy as np
#import models for streamlit

import gdown

# a file
url = "https://drive.google.com/file/d/18CjNTBC7qt3NQvvHmarXtycDKdp1sK-V/view?usp=drive_link"
output = "RFR_model.pkl"
gdown.download(url, output)


with open("RFR_model.pkl", "rb") as f:
  clf = pickle.load(f)
#try:
#  model1 = keras.models.load_model('drive/MyDrive/fuel_project/model1.keras')
#  model2 = keras.models.load_model('drive/MyDrive/fuel_project/model2.keras')
#  model3 = keras.models.load_model('drive/MyDrive/fuel_project/model3.keras')
#except:
#  with open("NN_model1.pkl", "rb") as f:
#    pickle.load(f)
#
#  with open("NN_model2.pkl", "rb") as f:
#    pickle.load(f)
#
#  with open("NN_model3.pkl", "rb") as f:
#    pickle.load(f)

#label encoding function
def LabelEncode(column, order):
  label_encoder.fit(input[column])
  label_encoder.classes_ = np.array(order)
  input[column] = label_encoder.transform(input[column])
  print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

#ui

st.title("Fuel consumption estimator")
st.subheader("An app to meet all your fuel estimation needs")
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
  return pd.DataFrame(data, index=[0])



input = ui()

run = st.button("Run")



if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None

if run:
  with st.status("running..."):
    st.write("Encoding...")
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
    st.write("Reordering...")
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

    st.write("Converting...")
    input = input.loc[0].values.tolist()
    input = [input]

    st.write("Predicting...")
    st.session_state.prediction = clf.predict(input)

  st.write(st.session_state.prediction)
