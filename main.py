from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

app = FastAPI()

modelo_mlp = joblib.load('resources/modelo_mlp.pkl')


@app.post("/predict")
def predict(request: dict):
    print("request_body = ", request)
    if request is None:
        raise HTTPException(status_code=400, detail='Request body cannot be null')
    else:
        values_list = list(request.values())
        data = np.array(values_list, dtype=int)
        print("values", data)
        prediction = modelo_mlp.predict([data])
        return prediction.tolist()
