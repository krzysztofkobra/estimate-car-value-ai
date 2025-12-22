# Car Price Prediction Model

System do predykcji cen samochodów oparty na LightGBM.

## 🚀 Szybki Start / Quick Start

### 1. Instalacja / Install
```bash
pip install pandas numpy lightgbm psycopg2-binary scikit-learn python-dotenv joblib matplotlib seaborn uvicorn fastapi
```

### 2. Konfiguracja / Setup
Utwórz plik `.env` / Create `.env` file:
```env
USER=your_user
PASSWORD=your_password
HOST=your_host
PORT=5432
DBNAME=your_db
```

### 3. Trening / Train
```bash
python car_price_model.py
```

### 4. Analiza / Analyze
```bash
python analyze_model.py
```

### 5. API
```bash
uvicorn api:app --reload
```

---


## Dlaczego LightGBM?

✅ **Natywna obsługa brakujących danych** - nie wymaga wypełniania pustych wartości
✅ **Doskonała wydajność z danymi kategorycznymi** - automatyczne enkodowanie
✅ **Szybki trening** - efektywny nawet na 270k rekordów
✅ **Resistance to outliers** - mniej wrażliwy na ekstremalne wartości
✅ **Feature importance** - widoczność, które cechy są najważniejsze

## Instalacja

```bash
pip install -r requirements.txt
```

## 1. Trening modelu

```python
from car_price_model import CarPricePredictor

connection_params = {
    'dbname': 'your_database',
    'user': 'your_user',
    'password': 'your_password',
    'host': 'localhost',
    'port': 5432
}

predictor = CarPricePredictor()
metrics = predictor.train(connection_params)
predictor.save_model('car_price_model.pkl')
```

## 2. Predykcja pojedynczego auta

```python
from car_price_model import CarPricePredictor

predictor = CarPricePredictor()
predictor.load_model('car_price_model.pkl')

car_data = {
    'make': 'Audi',
    'model': 'A5',
    'year': 2013,
    'body_type': 'Coupe',
    'fuel': 'benzyna',
    'engine_cc': 1984,
    'engine_power': 211,
    'transmission': 'Automatyczna',
    'drive': 'AWD',
    'mileage': 150000,
    'seller_type': 'private',
    'is_damaged': False,
    'color': 'Niebieski',
    'right_hand': False
}

result = predictor.predict(car_data)
print(f"Predicted price: {result['predicted_price']:,.0f} PLN")
print(f"Range: {result['confidence_range']['min']:,.0f} - {result['confidence_range']['max']:,.0f} PLN")
```

## 3. REST API

Uruchomienie API:

```bash
python api.py
```

API będzie dostępne na `http://localhost:8000`

### Przykładowe wywołanie:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "make": "Audi",
    "model": "A5",
    "year": 2013,
    "fuel": "benzyna",
    "engine_cc": 1984,
    "engine_power": 211,
    "transmission": "Automatyczna",
    "drive": "AWD",
    "mileage": 150000,
    "seller_type": "private",
    "is_damaged": false
  }'
```

### Python requests:

```python
import requests

response = requests.post('http://localhost:8000/predict', json={
    'make': 'Audi',
    'model': 'A5',
    'year': 2013,
    'fuel': 'benzyna',
    'engine_cc': 1984,
    'engine_power': 211,
    'transmission': 'Automatyczna',
    'drive': 'AWD',
    'mileage': 150000,
    'seller_type': 'private',
    'is_damaged': False
})

result = response.json()
print(f"Price: {result['predicted_price']:,.0f} PLN")
```

## Obsługa brakujących danych

Model radzi sobie z brakującymi wartościami:

```python
minimal_car = {
    'make': 'Toyota',
    'model': 'Corolla',
    'year': 2015,
    'fuel': 'benzyna',
    'mileage': 100000
}

result = predictor.predict(minimal_car)
```

Pole `body_type`, `transmission`, `drive`, `engine_cc`, `engine_power` mogą być puste - model użyje swojej wiedzy o podobnych autach.

## Feature Engineering

Model automatycznie tworzy dodatkowe cechy:

- **car_age** - wiek auta
- **mileage_per_year** - średni przebieg rocznie
- **power_to_cc_ratio** - stosunek mocy do pojemności
- **make_avg_price** - średnia cena dla marki
- **model_avg_price** - średnia cena dla modelu

## Używane pola z bazy danych

### Pola bazowe (13):
- make, model, year, body_type, fuel
- engine_cc, engine_power, transmission, drive
- mileage, seller_type, is_damaged
- **color** - kolor (ważny dla rzadkich kolorów)
- **right_hand** - kierownica po prawej (duży wpływ na cenę w Polsce)

### Pola engineered (5):
- car_age, mileage_per_year, power_to_cc_ratio
- make_avg_price, model_avg_price

**Razem: 18 features**

### Pola NIE używane:
- doors, nr_of_seats, location - często brakuje, mały wpływ
- vin, external_id, url - identyfikatory bez wartości predykcyjnej
- listing_date - nie wpływa bezpośrednio na wartość auta

## Wydajność

Na zbiorze ~270k rekordów:
- **Train time**: ~5-10 minut (zależnie od hardware)
- **Prediction**: < 10ms per car
- **Expected MAE**: 3000-5000 PLN
- **Expected R²**: 0.85-0.92

## Struktura projektu

```
├── car_price_model.py      # Główna klasa modelu
├── api.py                   # FastAPI endpoint
├── requirements.txt         # Zależności
└── car_price_model.pkl      # Wytrenowany model (po treningu)
```

## Diagnostyka

```python
predictor = CarPricePredictor()
predictor.load_model('car_price_model.pkl')

print(predictor.stats)
```

Output:
```
{
    'mean_price': 45000.0,
    'median_price': 38000.0,
    'test_mae': 4200.0,
    'test_r2': 0.89,
    'test_mape': 15.5
}
```

## Retraining

Model powinien być retrenowany regularnie (np. raz w miesiącu) gdy pojawią się nowe dane:

```bash
python -c "from car_price_model import CarPricePredictor; p = CarPricePredictor(); p.train({...}); p.save_model()"
```

## Uwagi

- Model działa najlepiej dla aut z lat 1990-2025
- Ceny są filtrowane do zakresu 1,000 - 1,000,000 PLN
- Przebiegi są ograniczone do 1,000,000 km
