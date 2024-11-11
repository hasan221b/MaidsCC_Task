import pickle

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_price(model, device):
    # Prepare input features from device data
    prediction = model.predict(device)
    return prediction