import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib
from car_price_model import CarPricePredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Car Price Prediction API")

predictor = CarPricePredictor()

try:
    predictor.load_model('car_price_model.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load model - {e}")
    predictor = None

class CarInput(BaseModel):
    make: str = Field(..., example="audi")
    model: str = Field(..., example="a5")
    year: int = Field(..., ge=1990, le=2025, example=2009)
    body_type: Optional[str] = Field(None, example="coupe")
    fuel: str = Field(..., example="benzyna")
    engine_cc: Optional[int] = Field(None, ge=500, le=10000, example=1984)
    engine_power: Optional[int] = Field(None, ge=30, le=1000, example=211)
    transmission: Optional[str] = Field(None, example="manualna")
    drive: Optional[str] = Field(None, example="awd")
    mileage: int = Field(..., ge=0, le=1000000, example=345000)
    seller_type: Optional[str] = Field("private", example="private")
    is_damaged: Optional[bool] = Field(False, example=False)
    color: Optional[str] = Field(None, example="niebieski")
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
        car_dict = {k: v for k, v in car.dict().items() if v is not None}
        
        logger.info(f"Received prediction request: {car_dict}")
        
        text_fields = ['make', 'model', 'body_type', 'fuel', 
                       'transmission', 'drive', 'seller_type', 'color']
        
        for field in text_fields:
            if car_dict.get(field) and isinstance(car_dict[field], str):
                car_dict[field] = car_dict[field].lower().strip()
        
        result = predictor.predict(car_dict)

        logger.info(f"Returned prediction: price={result['predicted_price']:.2f} PLN, range={result['confidence_range']}")
        
        return PredictionResponse(
            predicted_price=result['predicted_price'],
            confidence_range=result['confidence_range'],
            input_data=car_dict
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)