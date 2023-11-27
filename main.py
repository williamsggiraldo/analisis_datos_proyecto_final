from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

modelo_mlp = joblib.load('resources/modelo_mlp.pkl')


@app.post("/predict")
def predict(request: dict):
    print("request_body = ", request)
    values_list = list(request.values())
    data = np.array(values_list, dtype=int)
    print("values", data)
    prediction = modelo_mlp.predict([data])
    return prediction.tolist()
