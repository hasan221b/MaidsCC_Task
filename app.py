from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  
from utils import load_model, predict_price
from models import Device
import pandas as pd
import csv
from WF import write_row

import streamlit as st
import requests
import uvicorn
from threading import Thread

# FastAPI app
app = FastAPI()

# Load the ML model
model = load_model('rfmodel.pkl')

data = pd.read_csv('test.csv')

@app.get("/devices")
async def get_devices():
    result = data['id'].tolist()
    return result

@app.get("/devices/{device_id}")
async def get_device(device_id: int):
    device = data.loc[data['id'] == device_id][['battery_power','int_memory',
                       'px_height','px_width','ram','sc_h','sc_w']]
    device_dict = device.to_dict()

    return device_dict

@app.post("/devices")
async def add_device(device: Device):
    with open('devices.csv', 'a', newline='') as f_object:
        writer_object = csv.writer(f_object)
        device_data = list(device.__dict__.values())

        write_row('test.csv', device_data)
        return {"message": "Device added successfully"}

@app.post("/predict/{device_id}")
async def predict_price_for_device(device_id: int):
    device = data.loc[data['id'] == device_id][['battery_power','int_memory',
                       'px_height','px_width','ram','sc_h','sc_w']]
    
    if device.empty:
        raise HTTPException(status_code=404, detail="Device not found")

    device_features = device.values.reshape(1, -1)
    device_features = pd.DataFrame(device_features, columns=['battery_power', 'int_memory', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w'])
    try:
        predicted_price = predict_price(model, device_features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"predicted_price": predicted_price.tolist()[0]}

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Streamlit app
def run_streamlit():
    st.title("Device Price Prediction")

    device_id = st.number_input("Enter Device ID", min_value=0, step=1)
    if st.button("Get Device Details"):
        response = requests.get(f"http://localhost:8000/devices/{device_id}")
        if response.status_code == 200:
            st.json(response.json())
        else:
            st.error("Device not found")

    if st.button("Predict Price"):
        response = requests.post(f"http://localhost:8000/predict/{device_id}")
        if response.status_code == 200:
            predicted_price = response.json().get("predicted_price")
            st.success(f"Predicted Price: {predicted_price}")
        else:
            st.error("Prediction failed")

if __name__ == "__main__":
    # Run FastAPI in a separate thread
    thread = Thread(target=run_fastapi)
    thread.start()

    # Run Streamlit
    run_streamlit()