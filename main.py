from fastapi import FastAPI
from pydantic import BaseModel, Field, computed_field
import pickle
import pandas as pd
import math
from datetime import datetime, date

# -------------------------------
# Load trained model
# -------------------------------
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

# Get feature order from trained model
FEATURES = list(model.feature_names_in_)

app = FastAPI()


# -------------------------------
# Request Schema
# -------------------------------
class FraudPredictionRequest(BaseModel):

    # -------- RAW INPUT --------
    category: str = Field(..., example="shopping_pos")
    amt: float = Field(..., example=2500.50)
    gender: str = Field(..., example="male")

    lat: float = Field(..., example=28.6139)
    long: float = Field(..., example=77.2090)

    city_pop: int = Field(..., example=500000)

    merch_lat: float = Field(..., example=28.7041)
    merch_long: float = Field(..., example=77.1025)

    trans_datetime: datetime = Field(..., example="2024-01-01T22:30:00")
    dob: date = Field(..., example="1990-05-15")

    # -------- CATEGORY MAP --------
    _category_map = {
        "travel": 0,
        "shopping_pos": 1,
        "shopping_net": 2,
        "misc_net": 3,
        "misc_pos": 4,
        "entertainment": 5,
        "food_dining": 6,
        "personal_care": 7,
        "kids_pets": 8,
        "home": 9,
        "health_fitness": 10,
        "grocery_pos": 11,
        "grocery_net": 12,
        "gas_transport": 13
    }

    _gender_map = {
        "male": 0,
        "female": 1
    }

    # -------- COMPUTED FIELDS --------
    @computed_field
    @property
    def hour(self) -> int:
        return self.trans_datetime.hour

    @computed_field
    @property
    def age(self) -> int:
        today = date.today()
        return today.year - self.dob.year - (
            (today.month, today.day) < (self.dob.month, self.dob.day)
        )

    @computed_field
    @property
    def distance(self) -> float:
        R = 6371

        lat1 = math.radians(self.lat)
        lon1 = math.radians(self.long)
        lat2 = math.radians(self.merch_lat)
        lon2 = math.radians(self.merch_long)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    @computed_field
    @property
    def category_encoded(self) -> int:
        return self._category_map[self.category.lower()]

    @computed_field
    @property
    def gender_encoded(self) -> int:
        return self._gender_map[self.gender.lower()]


# -------------------------------
# Prediction Endpoint
# -------------------------------
@app.post("/predict")
def predict(data: FraudPredictionRequest):
    try:

        input_dict = {
            "amt": data.amt,
            "city_pop": data.city_pop,
            "lat": data.lat,
            "long": data.long,
            "merch_lat": data.merch_lat,
            "merch_long": data.merch_long,
            "distance": data.distance,
            "age": data.age,
            "category": data.category_encoded,
            "gender": data.gender_encoded,
            "hour": data.hour
        }

        df = pd.DataFrame([input_dict])[FEATURES]

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return {
            "fraud": int(prediction),
            "probability": float(probability),
            "label": "Fraud" if prediction == 1 else "Legit"
        }

    except Exception as e:
        return {"error": str(e)}


# -------------------------------
# Health Check
# -------------------------------
@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}