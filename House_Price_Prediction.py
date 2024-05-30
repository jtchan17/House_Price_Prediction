import json
import pickle
import numpy as np
import streamlit as st
import geopy.geocoders
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
import random
from shapely.geometry import Point, box
import base64


def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         body {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover;
             background-attachment: fixed; /* Ensures the background image stays fixed */
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


set_bg_hack('bg.png')

def get_random_points(state, city, num_points=1):
    geolocator = Nominatim(user_agent="streamlit_app")
    location = geolocator.geocode(f"{city}, {state}")
    if location is None:
        return None
    
    lat, lon = location.latitude, location.longitude
    offset = 0.05
    bbox = box(lon - offset, lat - offset, lon + offset, lat + offset)
    
    minx, miny, maxx, maxy = bbox.bounds
    random_points = []
    for _ in range(num_points):
        random_point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        random_points.append((random_point.y, random_point.x))
    return random_points

def get_city_location(state, city):
    geolocator = Nominatim(user_agent="streamlit_app")
    location = geolocator.geocode(f"{city}, {state}")
    if location is None:
        return None
    return location.latitude, location.longitude

def find_city_value(json_data, city_name):
    for state, state_data in json_data.items():
        if "cities" in state_data:
            if city_name in state_data["cities"]:
                return state_data["cities"][city_name]
        elif isinstance(state_data, dict):
            result = find_city_value(state_data, city_name)
            if result is not None:
                return result
    return None

def find_state_value(json_data, state_name):
    for state in json_data.keys():
        cleaned_state = state.replace(" ", "")
        cleaned_state_name = state_name.replace(" ", "")
        if cleaned_state_name == cleaned_state:
            return json_data[state].get("le_state", None)
    return None

def generate_deviation(average_price, deviation_percentage=0.1):
    deviation = average_price * deviation_percentage * random.uniform(-1, 1)
    return average_price + deviation

# Streamlit app code
st.title("USA Real Estate Price Prediction")

st.subheader("USA Map")

# Load the JSON data
with open("state&city.json", "r") as json_file:
    state_city_data = json.load(json_file)



# Input form
st.sidebar.title("Input Features")
bedrooms = st.sidebar.number_input("Number of bedroom", value=3, min_value=2, max_value=5)
bathrooms = st.sidebar.number_input("Number of bathroom", value=2, min_value=1, max_value=4)
acrelot = st.sidebar.number_input("Acre lot (in acres)", value=1.0, min_value=0.0, max_value=1.2)
house_size = st.sidebar.number_input("House Size (in square feet)", value=600, min_value=100, max_value=4363)
selected_state = st.sidebar.selectbox("State", list(state_city_data.keys()))

# Ensure selected_cities is not None
selected_cities = state_city_data.get(selected_state, {"cities": []})

if "cities" in selected_cities:
    selected_city = st.sidebar.selectbox("City", selected_cities["cities"])
else:
    selected_city = st.sidebar.selectbox("City", [])

# Getting city numerical value from mapping
city_value = find_city_value(state_city_data, selected_city) if selected_city else None

# Getting state numerical value from mapping
state_value = find_state_value(state_city_data, selected_state)


# Initialize session state
if "predictions" not in st.session_state:
    st.session_state["predictions"] = None
if "random_points" not in st.session_state:
    st.session_state["random_points"] = None
if "point_predictions" not in st.session_state:
    st.session_state["point_predictions"] = None
if "average_prediction" not in st.session_state:
    st.session_state["average_prediction"] = None
if "exactLocation" not in st.session_state:
    st.session_state["exactLocation"] = None
if "percentageDifference" not in st.session_state:
    st.session_state["percentageDifference"] = None


if city_value is not None and state_value is not None:
    # Prepare input data for prediction
    input_data = np.array([[bedrooms, bathrooms, acrelot, house_size, state_value, city_value]])

    # Load the model and encoders from the pickle file
    with open('predict_model.pkl', 'rb') as file:
        data = pickle.load(file)


    rf_model = data["model"]
    # dummyData = np.array([[3, 3,0.7,3,10,3852]]) #hard coded input
    # #bed bath acre size le_state city
    
    # priceForDummy = rf_model.predict(dummyData)
    # st.write(priceForDummy)
    averageStatePrices = [];
    for state, state_data in state_city_data.items():
    # Get state and city values
        state_value = state_data.get("le_state")
        first_city = next(iter(state_data.get("cities", {})), None)
        city_value = state_data.get("cities", {}).get(first_city)

    # Fixed values for bed, bath, acre_lot, and house_size
        bed = 2
        bath = 3
        acre_lot = 0.16
        house_size = 2000

    # Create input array
        input_array = np.array([[bed, bath, acre_lot, house_size, state_value, city_value]])

    # Predict price using the model
        predicted_price = rf_model.predict(input_array)[0]
        averageStatePrices.append(predicted_price)

# Compute the average price of all states
    averagePrice = np.mean(averageStatePrices)
    st.write("Average of all States: ${:.2f}".format(averagePrice))


    if "prediction_color" not in st.session_state:
        st.session_state["prediction_color"] = "blue"
# Display the average price rounded to 2 decimal places
    def update_predictions():
        average_price = rf_model.predict(input_data)[0]
        st.session_state["average_prediction"] = average_price

        # Check if the predicted price is higher or lower than the average price
        if average_price is not None and st.session_state["average_prediction"] is not None:
            if st.session_state["average_prediction"] > averagePrice:
                color = "red"
                
            elif st.session_state["average_prediction"] < averagePrice:
                color = "green"
            else:
                color = "black"
        else:
            color = "black"
        st.session_state["prediction_color"] = color
        st.session_state["percentageColor"] = color
        # Display the predicted price with appropriate color
        # st.write(f"Average Predicted Price: <span style='color:{color}; font-size:24px'>${st.session_state['average_prediction']}</span>", unsafe_allow_html=True)
        # average_price = rf_model.predict(input_data)[0]
        st.session_state["average_prediction"] = average_price
        st.session_state["percentageDifference"] = (((st.session_state["average_prediction"]) - averagePrice) / st.session_state["average_prediction"]) * 100
        # random_number = random.randint(1, 100)
        number_of_points = 1
        exact_Location = get_city_location(selected_state, selected_city)
        st.session_state["exactLocation"] = exact_Location
        random_points = get_random_points(selected_state, selected_city, num_points=number_of_points)
        st.session_state["random_points"] = random_points

        if random_points is None:
            st.error("Could not generate random points. Please check the city and state values.")
            return

        # Generate predictions for each point
        st.session_state["point_predictions"] = [
            generate_deviation(average_price) for _ in random_points
        ]

    if st.sidebar.button("Predict"):
        update_predictions()

    if st.session_state["average_prediction"] is not None:
        color = st.session_state["prediction_color"]
        st.write(f"Average Predicted Price: <span style='color:{color}; font-size:24px'>${st.session_state['average_prediction']}</span>", unsafe_allow_html=True)
    
    if st.session_state["percentageDifference"] is not None:
        difference = st.session_state["percentageDifference"]
        if difference > 0:
            sign = "+"
            color = "red"  # You mentioned to use red for an increase
        elif difference < 0:
            sign = ""
            color = "green"  # You mentioned to use green for a decrease
        else:
            sign = ""
            color = "black"  # You can change this color if needed

        st.write(f"Percentage Difference: <span style='color:{color}; font-size:24px'>{sign}{difference:.2f}%</span>", unsafe_allow_html=True)
        
    if st.session_state["exactLocation"]:
        m = folium.Map(location = st.session_state["exactLocation"], zoom_start = 12)
        folium.Marker(
            location = st.session_state["exactLocation"],
            popup = f"Predicted Price: ${st.session_state["average_prediction"]:,.2f}",
            icon = folium.Icon(color = 'blue')
        ).add_to(m)

        # Display the map in Streamlit
        st_folium(m, width=700, height=500)
    else:
        st.write("Choose a location to predict")
else:
    st.write("Please select valid state and city values to get predictions.")
