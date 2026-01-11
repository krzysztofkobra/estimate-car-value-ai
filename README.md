# Car Price Prediction API

AI-powered car price prediction microservice using LightGBM machine learning. This service is part of the [autoanaliza.pl](https://autoanaliza.pl) platform.

## 🚀 Live Demo

Try it live at **[www.autoanaliza.pl](https://autoanaliza.pl)** - create a free account to test the AI price predictions in action!

## Overview

This microservice provides real-time car price predictions based on vehicle specifications. It uses a LightGBM gradient boosting model trained on hundreds of thousands of car listings from the Polish automotive market.

### Key Features

- **Fast predictions**: < 10ms response time per vehicle
- **Accuracy**: ~8,800 PLN MAE on test set
- **Handles missing data**: Works with incomplete vehicle information
- **RESTful API**: Easy integration via FastAPI
- **Automatic feature engineering**: Creates derived features for better predictions

## Tech Stack

- **Python 3.12**
- **FastAPI** - Modern web framework for building APIs
- **LightGBM** - Gradient boosting framework for machine learning
- **PostgreSQL** - Database for car listings
- **scikit-learn** - Machine learning utilities
- **pandas & numpy** - Data processing

## How It Works

The model uses 18 features to predict car prices:

### Base Features (13)
- Make, model, year, body type, fuel type
- Engine displacement (cc), engine power (HP)
- Transmission, drive type, mileage
- Seller type, damage status, color, steering side

### Engineered Features (5)
- Car age (calculated from year)
- Mileage per year (mileage / age)
- Power-to-displacement ratio
- Average price for make
- Average price for model

The model automatically normalizes text inputs and handles missing values using LightGBM's native capabilities.

## API Endpoints

### `GET /`
Health check and API information

### `GET /health`
Detailed health status with model statistics

### `POST /predict`
Predict car price based on vehicle specifications

**Request body:**
```json
{
  "make": "audi",
  "model": "a5",
  "year": 2013,
  "body_type": "coupe",
  "fuel": "benzyna",
  "engine_cc": 1984,
  "engine_power": 211,
  "transmission": "manualna",
  "drive": "awd",
  "mileage": 150000,
  "seller_type": "private",
  "is_damaged": false,
  "color": "niebieski",
  "right_hand": false
}
```

**Response:**
```json
{
  "predicted_price": 45000.0,
  "confidence_range": {
    "min": 38250.0,
    "max": 51750.0
  },
  "input_data": { ... }
}
```

## Installation & Setup

### Prerequisites
- Python 3.10+
- PostgreSQL database with car listings

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/estimate-car-value-ai.git
cd estimate-car-value-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
Create a `.env` file:
```env
DBNAME=your_database
USER=your_user
PASSWORD=your_password
HOST=your_host
PORT=5432
```

### 4. Train the model (optional)
```bash
python car_price_model.py
```

This will:
- Load data from PostgreSQL
- Engineer features
- Train the LightGBM model
- Save the model to `car_price_model.pkl`

### 5. Run the API
```bash
python api.py
```

The API will be available at `http://localhost:8000`

## Usage Examples

### Python
```python
import requests

response = requests.post('http://localhost:8000/predict', json={
    'make': 'audi',
    'model': 'a5',
    'year': 2013,
    'fuel': 'benzyna',
    'mileage': 150000
})

result = response.json()
print(f"Predicted price: {result['predicted_price']:,.0f} PLN")
print(f"Range: {result['confidence_range']['min']:,.0f} - {result['confidence_range']['max']:,.0f} PLN")
```

### cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "make": "audi",
    "model": "a5",
    "year": 2013,
    "fuel": "benzyna",
    "mileage": 150000
  }'
```

## Model Performance

Trained on ~320,000 car listings:
- **MAE (Mean Absolute Error)**: 8,800 PLN
- **R² Score**: 0.9
- **Training time**: 2 minutes
- **Prediction time**: < 10ms per vehicle

## Project Structure

```
├── api.py                  # FastAPI application
├── car_price_model.py      # Model training and prediction logic
├── fetch_data.py           # Database data fetching utilities
├── analyze_model.py        # Model analysis tools
├── test_predictions.py     # Testing utilities
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
└── car_price_model.pkl    # Trained model (generated)
```

## Docker Deployment

```bash
docker build -t car-price-api .
docker run -p 8000:8000 --env-file .env car-price-api
```

## License

MIT License - feel free to use this project for your own purposes.

## About autoanaliza.pl

This microservice is part of [autoanaliza.pl](https://www.autoanaliza.pl), a comprehensive platform for analyzing the Polish automotive market. Create a free account to access:
- AI-powered price predictions
- Market analysis and trends
- Vehicle history insights
- And more!

---

**Note**: This model is optimized for the Polish automotive market and works best with vehicles from 1990-2025 with prices between 1,000-3,000,000 PLN.
