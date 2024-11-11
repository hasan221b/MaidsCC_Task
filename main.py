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

        print(device_data)
        write_row('test.csv',device_data)
        return {"message": "Device added successfully"}

        #writer_object.writerow(device_data)
    
        
    #with open("devices.csv", "a", newline="") as csvfile:
    
        #csvwriter = csv.writer(csvfile)
        #device_data = list(device.__dict__.values())
        #csvwriter.writerow(device_data)
        #print(device_data)
        
    


@app.post("/predict/{device_id}")
async def predict_price_for_device(device_id: int):

    device = data.loc[data['id'] == device_id][['battery_power','int_memory',
                       'px_height','px_width','ram','sc_h','sc_w']]
    
        
    if device.empty:
        raise HTTPException(status_code=404, detail="Device not found")

    # Make sure it's a 2D array
    device_features = device.values.reshape(1, -1)
    device_features = pd.DataFrame(device_features,columns=['battery_power', 'int_memory', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w'])
    try:
        predicted_price = predict_price(model, device_features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    print(predicted_price)
    return {"predicted_price": predicted_price.tolist()[0]}

