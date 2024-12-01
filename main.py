from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import numpy as np
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle


app = FastAPI()


with open("car_price_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
scaler = model_data["scaler"]
columns = model_data["columns"]
categorical_cols = model_data["categorical_cols"]

class Item(BaseModel):
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    seats: float


class Items(BaseModel):
    objects: List[Item]


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    missing_cols = set(columns) - set(data.columns)
    for col in missing_cols:
        data[col] = 0
    data = data[columns]
    data_scaled = scaler.transform(data)
    return data_scaled


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = pd.DataFrame([item.dict()])
    data_scaled = preprocess_data(data)
    prediction = model.predict(data_scaled)[0]
    return round(float(prediction), 2)


@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)) -> FileResponse:
    data = pd.read_csv(file.file)
    data_scaled = preprocess_data(data)
    predictions = model.predict(data_scaled)
    data["selling_price"] = predictions
    output_file = "selling_price.csv"
    data.to_csv(output_file, index=False)
    return FileResponse( path=output_file, media_type="text/csv", filename=output_file)