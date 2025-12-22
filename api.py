from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
from car_price_model import CarPricePredictor

app = FastAPI(title="Car Price Prediction API")

predictor = CarPricePredictor()

try:
    predictor.load_model('car_price_model.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load model - {e}")
    predictor = None

class CarInput(BaseModel):
    make: str = Field(..., example="Audi")
    model: str = Field(..., example="A5")
    year: int = Field(..., ge=1990, le=2025, example=2013)
    body_type: Optional[str] = Field(None, example="Coupe")
    fuel: str = Field(..., example="benzyna")
    engine_cc: Optional[int] = Field(None, ge=500, le=10000, example=1984)
    engine_power: Optional[int] = Field(None, ge=30, le=1000, example=211)
    transmission: Optional[str] = Field(None, example="Automatyczna")
    drive: Optional[str] = Field(None, example="AWD")
    mileage: int = Field(..., ge=0, le=1000000, example=150000)
    seller_type: Optional[str] = Field("private", example="private")
    is_damaged: Optional[bool] = Field(False, example=False)
    color: Optional[str] = Field(None, example="Niebieski")
    right_hand: Optional[bool] = Field(None, example=False)

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_range: dict
    input_data: dict

@app.get("/")
def root():
    return {
        "message": "Car Price Prediction API",
        "status": "ready" if predictor and predictor.model else "not ready",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_stats": predictor.stats
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_price(car: CarInput):
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        car_dict = car.dict()
        result = predictor.predict(car_dict)
        
        return PredictionResponse(
            predicted_price=result['predicted_price'],
            confidence_range=result['confidence_range'],
            input_data=car_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
