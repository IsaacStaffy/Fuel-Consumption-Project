
import streamlit as st
import pickle
import pandas as pd
from sklearn import preprocessing
import numpy as np
import os
import math
import gdown
import googlemaps
import openrouteservice
import requests
import json
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

API_KEY = st.secrets["API_KEY"]
ORS_API_KEY = st.secrets["ORS_API_KEY"]
ORS_client = openrouteservice.Client(key=ORS_API_KEY)
gmaps = googlemaps.Client(key=API_KEY)


# ------------------------- Functions -------------------------

# -- Utility
# Truncate float f to n decimal places without rounding
def truncate(f, n):
    if n == 0:
        return int(f)
    return math.floor(f * 10 ** n) / 10 ** n

# Converts meters_to_feet
def meters_to_feet(meters):
  feet = meters * 3.28084
  return feet

# Convert meters to miles
def meters_to_miles(meters):
    return meters / 1609.344


# Convert grams per kilometer to tonnes per mile
def gpkm_to_tpm(gpkm):
    grams_per_tonne = 1_000_000
    kilometers_per_mile = 1.60934
    return (gpkm / grams_per_tonne) * kilometers_per_mile

# --- Google maps
# Address to Coordinates (Geocoding)
def get_lat_lng(gmaps, address):
    try:
        geocode_result = gmaps.geocode(address)
        if geocode_result:
            lat = geocode_result[0]['geometry']['location']['lat']
            lng = geocode_result[0]['geometry']['location']['lng']
            return lat, lng
        else:
            print("Geocoding failed. Address not found.")
            return None
    except Exception as e:
        print(f"An error occurred during geocoding: {e}")
        return None

# Gets elevation of a coordinate in meters
def get_elevation(gmaps, lat, lng):
    try:
        elevation_result = gmaps.elevation((lat, lng))
        if elevation_result:
            elevation = elevation_result[0]['elevation']
            return elevation
        else:
            print("Elevation lookup failed.")
            return None
    except Exception as e:
        print("An error occurred during elevation lookup: {e}")
        return None

# -- Open route service maps api (after ors_client has been defined)
# Snaps coordinates to nearest to road
def snap_to_road(lat, lon):
    result = ORS_client.pelias_reverse((lon, lat))
    snapped = result['features'][0]['geometry']['coordinates']  # [lon, lat]
    return [snapped[1], snapped[0]]  # Return as [lat, lon]

# Snap coords to nearest city/town
def snap_to_city(latitude, longitude):
    # Initialize geolocator with user-agent
    geolocator = Nominatim(user_agent="city_snapper")

    # Get location from coordinates
    location = geolocator.reverse((latitude, longitude), language='en', exactly_one=True, timeout=10)

    if location:
        # Extract address details
        address = location.raw.get('address', {})

        # Try to get the most accurate city/town/village info
        city = address.get('city', None)
        town = address.get('town', None)
        village = address.get('village', None)
        suburb = address.get('suburb', None)

        # Use a hierarchy of fallback options
        if city:
            return location.latitude, location.longitude, city
        elif town:
            return location.latitude, location.longitude, town
        elif village:
            return location.latitude, location.longitude, village
        elif suburb:
            return location.latitude, location.longitude, suburb
        else:
            # Return general location if no city/town info available
            state = address.get('state', '')
            country = address.get('country', '')
            return location.latitude, location.longitude, f"{state}, {country}", 'spacer'
    else:
        return None

# Locate gas stations: returns [[[lon1, lat1], [lon2, lat2]], [{'name': 'Chevron'}, {'name': 'Shell'}]] up to 3 stations
def locate_gas(lat, lon, search_radius=2000, amount=10, ORS_API_KEY=ORS_API_KEY):
  body = {"request":"pois","geometry":{"geojson":{"type":"Point","coordinates":[lon, lat]},"buffer":search_radius},"filters":{"category_ids":[596]},"limit":amount}
  headers = {
      'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
      'Authorization': ORS_API_KEY,
      'Content-Type': 'application/json; charset=utf-8'
      }

  call = requests.post('https://api.openrouteservice.org/pois', json=body, headers=headers)
  response = json.loads(call.text)

  try:
    if response["features"]:
      return [[feature['geometry']['coordinates'] for feature in response["features"]],
              [feature['properties'].get('osm_tags', {}) for feature in response["features"]]]
    else:
      print("no features found")
  except:
    print(f"issue finding features: {response}")

# subdevides a route returns pandas df of points,
def subdivide_route(df, input, units="miles"):
  if units == "meters":
    interval = input
  elif units == "miles":
    interval = input * 1609.34

  # Initialize cumulative distance
  cumulative_distance = [0]
  for i in range(1, len(df)):
    dist = geodesic(
      (df.iloc[i-1]['lat'], df.iloc[i-1]['lon']),
      (df.iloc[i]['lat'], df.iloc[i]['lon'])
    ).meters
    cumulative_distance.append(cumulative_distance[-1] + dist)

  df['cumulative_dist'] = cumulative_distance

  # Get points at every interval meters
  output_points = []
  next_dist = 0

  for i in range(1, len(df)):
    if df.iloc[i]['cumulative_dist'] >= next_dist:
      output_points.append(df.iloc[i])
      next_dist += interval

  return pd.DataFrame(output_points).drop(columns=['cumulative_dist'])

# -- Model functions
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

# -- Main body dunctions
# UI for collecting user inputs
def get_input_data():
    st.title("Fuel Consumption and Road trip calculator")
    st.subheader("Input vehical specifications and click run")

    st.sidebar.title("Vehicle Spcifications")
    st.sidebar.write("Use Automobile-catalog to get unknown vehicle details")
    st.sidebar.link_button("Go to Automobile-catalog", "https://www.automobile-catalog.com/")

    year = st.sidebar.slider("Year", 2000, 2023)
    model = st.sidebar.radio("Drive Type", ["2WD", "4WD", "AWD"])
    size = st.sidebar.selectbox("Size", ['mini', 'sub', 'small', 'mid-size', 'standard', 'full-size', 'passenger', 'cargo', 'massive'])
    vehicle_class = st.sidebar.selectbox("Vehicle Class", [
        'compact', 'mid-size', 'station wagon', 'two-seater', 'full-size',
        'suv', 'van', 'pickup truck', 'minivan', 'special purpose vehicle'
    ])
    engine_size = st.sidebar.slider("Engine Size (L)", 0.9, 8.5, step=0.1)
    cylinders = st.sidebar.slider("Cylinder Count", 1, 16)
    fuel_type = st.sidebar.selectbox("Fuel Type", ['diesel', 'premium gasoline', 'gasoline', 'ethanol', 'natural gas'], index=2)
    make = st.sidebar.selectbox("Brand", [
        'acura', 'audi', 'bmw', 'buick', 'cadillac', 'chevrolet', 'chrysler', 'daewoo', 'dodge', 'ferrari',
        'ford', 'gmc', 'honda', 'hyundai', 'infiniti', 'isuzu', 'jaguar', 'jeep', 'kia', 'land rover',
        'lexus', 'lincoln', 'mazda', 'mercedes-benz', 'nissan', 'oldsmobile', 'plymouth', 'pontiac',
        'porsche', 'saab', 'saturn', 'subaru', 'suzuki', 'toyota', 'volkswagen', 'volvo', 'bentley',
        'rolls-royce', 'maserati', 'mini', 'mitsubishi', 'smart', 'hummer', 'aston martin', 'lamborghini',
        'bugatti', 'scion', 'fiat', 'ram', 'srt', 'alfa romeo', 'genesis'
    ])
    transmission = st.sidebar.radio("Transmission Type", ["Automatic", "Automatic Manual", "Automatic with select shift", "Continuously variable transmission", "Manual"])

    gears_disabled = (transmission == "AV")
    gears = st.sidebar.slider("Number of Gears", 1, 10, disabled=gears_disabled)

    # Remap Transission types from readible values to the ones used in preproccesing and prediction
    if transmission == "Automatic":
      transmission = "A"
    elif transmission == "Automatic Manual":
      transmission = "AM"
    elif transmission == "Automatic with select shift":
      transmission = "AS"
    elif transmission == "Continuously variable transmission":
      transmission = "AV"
    elif transmission == "Manual":
      transmission = "M"

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
    input_df = LabelEncode(input_df, "fuel", ['diesel', 'premium gasoline', 'gasoline', 'ethanol', 'natural gas'])

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
run = st.button("Run", use_container_width=True)

# Initialize session state variables
st.session_state.setdefault('prediction', None)
st.session_state.setdefault('ran', False)

bypass = st.segmented_control("Maps API", ["Use", "Bypass"], selection_mode="single")

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

    gas_price = st.number_input("Estimated Gas Price", value=None)
    tank_size = st.number_input("Gas Tank Size (gallons)", value=None)

    mileage = None

    # Use maps api: disply features
    if bypass == "Use":
      highway_ratio = 0.75

      # Start and end inputs
      start = st.text_input("Start point")
      end = st.text_input("End point")

      # Route display subdivision
      subdivision = st.slider("tenths of a mile between route markers", 1, 250)
      subdivision /= 10

      if start and end:
        start_coords = get_lat_lng(gmaps, start)
        end_coords = get_lat_lng(gmaps, end)

        # If google maps returns coords
        if start_coords and end_coords:
          start_lat, start_lon = start_coords
          end_lat, end_lon = end_coords

          # Button for finding gas stations
          if tank_size:
            gas_subdivision = st.slider("percent of tank between shown gas stops", 0, 100, value=25) * 0.01
            gas_locator = st.button("Locate gas stations along route")
          else:
            gas_locator = False
            gas_subdivision = 0

          # Get route
          try:
            route = ORS_client.directions(
                coordinates=[[start_lon, start_lat], [end_lon, end_lat]],
                profile='driving-car',
                format='geojson'
            )
          # Check if route is valid
          except:
            st.title(":warning: No route found :warning:")
            route = None

          if route:
            # Decode the polyline from geometry (or use GeoJSON coordinates directly)
            geometry = route['features'][0]['geometry']['coordinates']

            # Convert geometry to a DataFrame
            route_points = pd.DataFrame(geometry, columns=['lon', 'lat'])

            # Locate gas stations along route
            if gas_locator:
              with st.status("Locating gas stations...."):
                # subdivide route so that you have 80% of gas tank left between waypoints if filling up at gas stations
                gas_waypoints = subdivide_route(route_points, (st.session_state.prediction[0][1] * tank_size) * gas_subdivision)

                gas_stations = []
                # iterate through gas waypoints and locate gas stations
                for index, row in gas_waypoints.iterrows():
                  stations = locate_gas(row['lat'], row['lon'])
                  # if it found stations
                  if stations:
                    print(f"fully formatted: {stations}")
                    gas_stations.append(stations)
                  else:
                    # attempt to snap it to a city
                    snapped_coords = snap_to_city(row['lat'], row['lon'])
                    # If it succesfully snapped coords to a city
                    if len(snapped_coords) == 3:
                      print(f"snapped to city: {snapped_coords}")
                      stations = locate_gas(snapped_coords[0], snapped_coords[1])
                    # didn't find city and snapped to area instead
                    elif len(snapped_coords) == 4:
                      print("no city near")
                    # something wrong happened
                    else:
                      print("Error snapping to town")
                    if stations:
                      print(f"fully formatted after snapped: {stations}")
                      gas_stations.append(stations)

                print(f"searching for gas stations along {len(gas_stations)} points")

                gas_station_points = []

                for station_set in gas_stations:
                  for station_coords in station_set[0]:
                    gas_station_points.append([station_coords[1], station_coords[0]])

                  for station_info in station_set[1]:
                    st.write(station_info)


                # Make pandas dataframe out of points and add color and size
                gas_station_points = pd.DataFrame(gas_station_points, columns=['lat', 'lon'])
                gas_station_points['color'] = '#AAFF0080'
                gas_station_points['size'] = 50

            # Make dataframe of route display points and add color and size
            route_display_points = subdivide_route(route_points, subdivision)
            route_display_points['color'] = '#FF0000'
            route_display_points['size'] = 0.1

            # Make end and start points for route and add color and size
            end_points = pd.DataFrame({"lat": [start_lat, end_lat], "lon": [start_lon, end_lon]})
            end_points['color'] = '#0000FF80'
            end_points['size'] = 75

            # Combine route display and end points into one dataframe
            if gas_locator:
              final_points = pd.concat([route_display_points, end_points, gas_station_points], ignore_index=True)
            else:
              final_points = pd.concat([route_display_points, end_points], ignore_index=True)

            # Display map
            st.map(final_points, color='color', size='size')

            # Get elevation from start and end coords
            start_elevation = get_elevation(gmaps, start_lat, start_lon)
            end_elevation = get_elevation(gmaps, end_lat, end_lon)
            avg_elevation = (meters_to_feet(start_elevation)  + meters_to_feet(end_elevation)) / 2

            # Elevation checks and warnings
            if avg_elevation >= 1750:
              st.write(f":warning: *Warning: The elevation you are driving at is {int(avg_elevation)} and at this elevation engine power and effciency will be reduced by about {int((avg_elevation / 1000) * 3)}%* :warning:")
            if (end_elevation - start_elevation) >= 500:
              st.write(f":warning: *Warning: Net elevation change ascends {int(meters_to_feet(end_elevation - start_elevation))} feet which may reduce fuel efficiency* :warning:")
            if (end_elevation - start_elevation) <= -500:
              st.write(f"*Net elevation change descends {int(meters_to_feet(end_elevation - start_elevation))} feet which may increase fuel efficiency*")

            # Link to google maps route
            st.link_button("Go to Google maps route", f"https://www.google.com/maps/dir/?api=1&origin={start_lat},{start_lon}&destination={end_lat},{end_lon}&travelmode=driving")

            # Auto populate milaege from route
            mileage = meters_to_miles(route['features'][0]['properties']['summary']['distance'])
            st.write(f"Mileage: {truncate(mileage, 1)}")

            # Display elevation stats
            st.write("Elevation Change (m): " + str(truncate(end_elevation - start_elevation, 2)))
            st.write("Average Elevation (m): " + str(truncate(avg_elevation, 2)))
    else:
      # Request mileage from the user
      mileage = st.number_input("Trip Mileage Estimate", value=None)
      highway_ratio = st.slider("Highway Driving %", 0, 100, value=75) / 100

    # Calculate stats from mileage etc
    if mileage:
        h_miles = highway_ratio * mileage
        r_miles = (1 - highway_ratio) * mileage
    if mileage and gas_price:
        total_cost = truncate(((h_miles / st.session_state.prediction[0][1]) + (r_miles / st.session_state.prediction[0][0])) * gas_price, 2)
    if mileage and tank_size:
        tank_refills = int(((h_miles / st.session_state.prediction[0][1]) + (r_miles / st.session_state.prediction[0][0])) / (tank_size * 0.75))

    if mileage and gas_price and tank_size:
        st.table(pd.DataFrame([[total_cost, tank_refills]], columns=["Trip Cost", "Tank Refills"]))

    # Emmisions estimate
    if mileage:
      emissions_tpm = gpkm_to_tpm(st.session_state.prediction[0][2])
      st.subheader(f"Total Emissions (Tonnes): {truncate(emissions_tpm * mileage, 10)}")

      # Display donation stuff
      st.write(f"Donate ${truncate((emissions_tpm * mileage) * 18.68, 2)}, to offset carbon emmisions")

      st.link_button("Donate Here", "https://www.cooleffect.org/store/donate")

      st.subheader("What is this?")
      st.write("Cooleffect.org takes carbon offset donations and uses them to fund emmision reducing projects around the world!")
      st.write("Learn more here:")
      st.page_link("https://www.cooleffect.org/about-us", label="CoolEffect.org", icon="🌎")
