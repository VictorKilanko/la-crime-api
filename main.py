from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="LA Crime Predictor",
    description="Predict Violent Crime Rate per 100k in Los Angeles using linear regression.",
    version="1.0"
)

# NEW: Landing page route
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the LA Crime Predictor API!",
        "how_to_use": "Go to /docs for the interactive interface or POST to /predict with your input data."
    }

# Input schema
class CrimeInput(BaseModel):
    Homicide_per_100k: float
    ForRape_per_100k: float
    Robbery_per_100k: float
    AggAssault_per_100k: float
    TruckDrivers: float
    MaleVietnamVeterans: float

# Prediction endpoint
@app.post("/predict")
def predict_crime(data: CrimeInput):
    prediction = (
        1.2789
        + 0.0274 * data.Homicide_per_100k
        + 0.0885 * data.ForRape_per_100k
        + 0.3119 * data.Robbery_per_100k
        + 0.6143 * data.AggAssault_per_100k
        - 0.0622 * data.TruckDrivers
        + 0.0238 * data.MaleVietnamVeterans
    )
    return {"Violent_per_100k_Predicted": round(prediction, 2)}
