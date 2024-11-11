import streamlit as st
import pandas as pd
from pydantic import BaseModel
import csv
from csv import writer
import pickle

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_price(model, device):
    # Prepare input features from device data
    prediction = model.predict(device)
    return prediction

def write_row(file, L):
    with open(file, 'a') as f_object:

        writer_object = writer(f_object)


        writer_object.writerow(L)

        f_object.close()
class Device(BaseModel):
    id:int
    battery_power:int
    blue:int	
    clock_speed:int
    dual_sim:int
    fc:int
    four_g:int
    int_memory:int
    m_dep:int
    mobile_wt:int
    n_cores:int
    pc:int
    px_height:int	
    px_width:int	
    ram	:int
    sc_h:int
    sc_w:int
    talk_time:int
    three_g:int
    touch_screen:int	
    wifi:int



# Load the ML model
model = load_model('rfmodel.pkl')
data = pd.read_csv('test.csv')

# Function to get all devices
def get_devices():
    return data['id'].tolist()

# Function to get specific device details
def get_device(device_id: int):
    device = data.loc[data['id'] == device_id][['battery_power', 'int_memory',
                                                'px_height', 'px_width', 'ram', 'sc_h', 'sc_w']]
    return device.to_dict()

# Function to add a device
def add_device(device: Device):
    device_data = list(device.__dict__.values())
    with open('devices.csv', 'a', newline='') as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerow(device_data)
    write_row('test.csv', device_data)
    return "Device added successfully"

# Function to predict price for a device
def predict_price_for_device(device_id: int):
    device = data.loc[data['id'] == device_id][['battery_power', 'int_memory',
                                                'px_height', 'px_width', 'ram', 'sc_h', 'sc_w']]
    if device.empty:
        return "Device not found"
    
    device_features = device.values.reshape(1, -1)
    device_features = pd.DataFrame(device_features, columns=['battery_power', 'int_memory', 'px_height',
                                                             'px_width', 'ram', 'sc_h', 'sc_w'])
    try:
        predicted_price = predict_price(model, device_features)
    except Exception as e:
        return f"Error: {str(e)}"
    
    return predicted_price.tolist()[0]

# Streamlit app
st.title("Device Price Predictor")

# Get devices
st.header("Available Devices")
device_ids = get_devices()
st.write(device_ids)

# Get specific device details
device_id = st.number_input("Enter Device ID to get details", min_value=0)
if st.button("Get Device Details"):
    device_details = get_device(device_id)
    st.write(device_details)

# Predict price for a device
st.header("Predict Device Price")
device_id_for_prediction = st.number_input("Enter Device ID for price prediction", min_value=0)
if st.button("Predict Price"):
    price = predict_price_for_device(device_id_for_prediction)
    st.write(f"Predicted Price: {price}")

# Add a new device
st.header("Add a New Device")
battery_power = st.number_input("Battery Power", min_value=0)
int_memory = st.number_input("Internal Memory", min_value=0)
px_height = st.number_input("Pixel Height", min_value=0)
px_width = st.number_input("Pixel Width", min_value=0)
ram = st.number_input("RAM", min_value=0)
sc_h = st.number_input("Screen Height", min_value=0)
sc_w = st.number_input("Screen Width", min_value=0)

if st.button("Add Device"):
    new_device = Device(
        battery_power=battery_power,
        int_memory=int_memory,
        px_height=px_height,
        px_width=px_width,
        ram=ram,
        sc_h=sc_h,
        sc_w=sc_w
    )
    message = add_device(new_device)
    st.write(message)
