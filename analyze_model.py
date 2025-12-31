import pandas as pd
import numpy as np
from car_price_model import CarPricePredictor
import matplotlib.pyplot as plt
import seaborn as sns

predictor = CarPricePredictor()
predictor.load_model('car_price_model.pkl')

print("=" * 80)
print("ANALIZA MODELU PREDYKCJI CEN")
print("=" * 80)

print("\n1. STATYSTYKI MODELU")
print("-" * 80)
for key, value in predictor.stats.items():
    if isinstance(value, float):
        if 'price' in key:
            print(f"  {key:25s}: {value:,.0f} PLN")
        elif 'mape' in key or 'r2' in key:
            print(f"  {key:25s}: {value:.2f}%") if 'mape' in key else print(f"  {key:25s}: {value:.4f}")
        else:
            print(f"  {key:25s}: {value:.2f}")

print("\n2. FEATURE IMPORTANCE (Top 20)")
print("-" * 80)
importance_df = pd.DataFrame({
    'feature': predictor.model.feature_name(),
    'importance': predictor.model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

total_importance = importance_df['importance'].sum()
importance_df['percentage'] = (importance_df['importance'] / total_importance * 100)

for idx, row in importance_df.head(20).iterrows():
    bar = '█' * int(row['percentage'] / 2)
    print(f"  {row['feature']:25s} {bar:30s} {row['percentage']:5.1f}%")

print("\n3. FEATURE CATEGORIES")
print("-" * 80)

categorical_features = ['make', 'model', 'body_type', 'fuel', 'transmission', 'drive', 'seller_type', 'color']
engineered_features = ['car_age', 'mileage_per_year', 'power_to_cc_ratio', 'make_avg_price', 'model_avg_price']
base_features = ['year', 'engine_cc', 'engine_power', 'mileage', 'is_damaged', 'right_hand']

categories = {
    'Categorical Features': categorical_features,
    'Engineered Features': engineered_features,
    'Base Features': base_features
}

for category, features in categories.items():
    category_importance = importance_df[importance_df['feature'].isin(features)]['percentage'].sum()
    print(f"  {category:25s}: {category_importance:5.1f}%")

print("\n4. WPŁYW POSZCZEGÓLNYCH CECH (symulacja)")
print("-" * 80)

base_car = {
    'make': 'audi',
    'model': 'a5',
    'year': 2009,
    'fuel': 'benzyna',
    'engine_cc': 1984,
    'engine_power': 211,
    'mileage': 345000
}

base_price = predictor.predict(base_car)['predicted_price']
print(f"\nAuto bazowe (Audi A5 2009, 345k km): {base_price:,.0f} PLN")

tests = [
    ('Przebieg +50k km', {'mileage': base_car['mileage']+50000}),
    ('Przebieg -50k km', {'mileage': base_car['mileage']-50000}),
    ('Rok +3 lata', {'year': base_car['year']+3}),
    ('Rok -3 lata', {'year': base_car['year']-3}),
    ('Diesel zamiast benzyny', {'fuel': 'diesel'}),
    ('Większy silnik', {'engine_cc': base_car['engine_cc'], 'engine_power': base_car['engine_power']}),
    ('Sprzedawca: dealer', {'seller_type': 'dealer'}),
    ('Uszkodzony', {'is_damaged': True}),
    ('Kierownica po prawej (RHD)', {'right_hand': True}),
    ('Kolor: żółty', {'color': 'zolty'}),
    ('Kolor: czarny', {'color': 'czarny'}),
    ('Kolor: czarny', {'color': 'bialy'})
]

print("\nWpływ zmian:")
for description, changes in tests:
    modified_car = base_car.copy()
    modified_car.update(changes)
    new_price = predictor.predict(modified_car)['predicted_price']
    diff = new_price - base_price
    diff_pct = (diff / base_price) * 100

    sign = '+' if diff > 0 else ''
    print(f"  {description:30s}: {sign}{diff:,.0f} PLN ({sign}{diff_pct:.1f}%)")

print("\n5. PORÓWNANIE POPULARNYCH MODELI AUT (2016, 100k km)")
print("-" * 80)

popular_cars = [
    {'make': 'toyota', 'model': 'corolla', 'engine_cc': 1600, 'engine_power': 132},
    {'make': 'volkswagen', 'model': 'golf', 'engine_cc': 1400, 'engine_power': 122},
    {'make': 'bmw', 'model': '3', 'engine_cc': 2000, 'engine_power': 184},
    {'make': 'audi', 'model': 'a4', 'engine_cc': 2000, 'engine_power': 190},
    {'make': 'mercedes-benz', 'model': 'c', 'engine_cc': 2000, 'engine_power': 184},
    {'make': 'skoda', 'model': 'octavia', 'engine_cc': 1400, 'engine_power': 150},
    {'make': 'ford', 'model': 'focus', 'engine_cc': 1600, 'engine_power': 125},
    {'make': 'opel', 'model': 'astra', 'engine_cc': 1400, 'engine_power': 140},
    {'make': 'renault', 'model': 'megane', 'engine_cc': 1500, 'engine_power': 110},
    {'make': 'peugeot', 'model': '308', 'engine_cc': 1600, 'engine_power': 120},
    {'make': 'hyundai', 'model': 'i30', 'engine_cc': 1400, 'engine_power': 140},
    {'make': 'kia', 'model': 'ceed', 'engine_cc': 1400, 'engine_power': 140},
    {'make': 'mazda', 'model': '3', 'engine_cc': 2000, 'engine_power': 165},
    {'make': 'honda', 'model': 'civic', 'engine_cc': 1800, 'engine_power': 142},
    {'make': 'nissan', 'model': 'qashqai', 'engine_cc': 1600, 'engine_power': 130},
]

car_prices = []
for car_spec in popular_cars:
    car = {
        'make': car_spec['make'],
        'model': car_spec['model'],
        'year': 2016,
        'fuel': 'benzyna',
        'engine_cc': car_spec['engine_cc'],
        'engine_power': car_spec['engine_power'],
        'mileage': 100000
    }
    try:
        price = predictor.predict(car)['predicted_price']
        car_prices.append((f"{car['make']} {car['model']}", price))
    except Exception as e:
        pass

car_prices.sort(key=lambda x: x[1], reverse=True)

max_price = max([p for _, p in car_prices])
for name, price in car_prices:
    bar_length = int((price / max_price) * 40)
    bar = '█' * bar_length
    print(f"  {name:25s} {bar:40s} {price:>10,.0f} PLN")

print("\n6. PORÓWNANIE SEGMENTÓW (Sedan, SUV, Hatchback)")
print("-" * 80)

segments = [
    ('Compact/Hatchback', [
        {'make': 'volkswagen', 'model': 'golf', 'body_type': 'compact', 'engine_power': 122},
        {'make': 'toyota', 'model': 'corolla', 'body_type': 'compact', 'engine_power': 132},
        {'make': 'ford', 'model': 'focus', 'body_type': 'compact', 'engine_power': 125},
    ]),
    ('Sedan Premium', [
        {'make': 'bmw', 'model': '3', 'body_type': 'sedan', 'engine_power': 184},
        {'make': 'audi', 'model': 'a4', 'body_type': 'sedan', 'engine_power': 190},
        {'make': 'mercedes-benz', 'model': 'c', 'body_type': 'sedan', 'engine_power': 184},
    ]),
    ('SUV', [
        {'make': 'toyota', 'model': 'rav4', 'body_type': 'suv', 'engine_power': 175},
        {'make': 'nissan', 'model': 'qashqai', 'body_type': 'suv', 'engine_power': 130},
        {'make': 'volkswagen', 'model': 'tiguan', 'body_type': 'suv', 'engine_power': 150},
    ])
]

for segment_name, cars in segments:
    prices = []
    for car_spec in cars:
        car = {
            'make': car_spec['make'],
            'model': car_spec['model'],
            'year': 2016,
            'body_type': car_spec.get('body_type'),
            'fuel': 'benzyna',
            'engine_cc': 2000,
            'engine_power': car_spec['engine_power'],
            'mileage': 100000
        }
        try:
            price = predictor.predict(car)['predicted_price']
            prices.append(price)
        except:
            pass

    if prices:
        avg_price = np.mean(prices)
        print(f"  {segment_name:25s}: {avg_price:>10,.0f} PLN (średnia)")

print("\n7. WPŁYW WIEKU AUTA (VW Golf 1.4 TSI, 100k km)")
print("-" * 80)

base_car = {
    'make': 'volkswagen',
    'model': 'golf',
    'fuel': 'benzyna',
    'engine_cc': 1400,
    'engine_power': 122,
    'mileage': 100000
}

years = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
year_prices = []

for year in years:
    car = base_car.copy()
    car['year'] = year
    try:
        price = predictor.predict(car)['predicted_price']
        year_prices.append((year, price))
    except:
        pass

max_year_price = max([p for _, p in year_prices])
for year, price in year_prices:
    bar_length = int((price / max_year_price) * 40)
    bar = '█' * bar_length
    print(f"  {year}: {bar:40s} {price:>10,.0f} PLN")

print("\n8. WPŁYW PRZEBIEGU (VW Golf 2016, różne przebiegi)")
print("-" * 80)

base_car = {
    'make': 'volkswagen',
    'model': 'golf',
    'year': 2016,
    'fuel': 'benzyna',
    'engine_cc': 1400,
    'engine_power': 122
}

mileages = [20000, 50000, 100000, 150000, 200000, 250000, 300000]
mileage_prices = []

for mileage in mileages:
    car = base_car.copy()
    car['mileage'] = mileage
    try:
        price = predictor.predict(car)['predicted_price']
        mileage_prices.append((mileage, price))
    except:
        pass

max_mileage_price = max([p for _, p in mileage_prices])
for mileage, price in mileage_prices:
    bar_length = int((price / max_mileage_price) * 40)
    bar = '█' * bar_length
    print(f"  {mileage:>7,} km: {bar:40s} {price:>10,.0f} PLN")

print("\n9. PORÓWNANIE PALIW (VW Golf 2016, 2.0L)")
print("-" * 80)

base_car = {
    'make': 'volkswagen',
    'model': 'golf',
    'year': 2016,
    'engine_cc': 2000,
    'engine_power': 150,
    'mileage': 100000
}

fuels = ['benzyna', 'diesel', 'LPG', 'hybryda', 'elektryczny']
fuel_prices = []

for fuel in fuels:
    car = base_car.copy()
    car['fuel'] = fuel
    try:
        price = predictor.predict(car)['predicted_price']
        fuel_prices.append((fuel, price))
    except:
        pass

fuel_prices.sort(key=lambda x: x[1], reverse=True)
max_fuel_price = max([p for _, p in fuel_prices])

for fuel, price in fuel_prices:
    bar_length = int((price / max_fuel_price) * 40)
    bar = '█' * bar_length
    print(f"  {fuel:15s}: {bar:40s} {price:>10,.0f} PLN")

print("\n10. PREMIUM vs BUDGET (porównanie klas)")
print("-" * 80)

premium_cars = [
    {'make': 'bmw', 'model': '5', 'engine_power': 245},
    {'make': 'audi', 'model': 'a6', 'engine_power': 252},
    {'make': 'mercedes-benz', 'model': 'e', 'engine_power': 245},
]

budget_cars = [
    {'make': 'dacia', 'model': 'logan', 'engine_power': 90},
    {'make': 'skoda', 'model': 'fabia', 'engine_power': 95},
    {'make': 'fiat', 'model': 'panda', 'engine_power': 69},
]

print("\nPremium (2016, 100k km):")
premium_prices = []
for car_spec in premium_cars:
    car = {
        'make': car_spec['make'],
        'model': car_spec['model'],
        'year': 2016,
        'fuel': 'benzyna',
        'engine_cc': 3000,
        'engine_power': car_spec['engine_power'],
        'mileage': 100000
    }
    try:
        price = predictor.predict(car)['predicted_price']
        premium_prices.append(price)
        print(f"  {car['make']} {car['model']:15s}: {price:>10,.0f} PLN")
    except:
        pass

print("\nBudget (2016, 100k km):")
budget_prices = []
for car_spec in budget_cars:
    car = {
        'make': car_spec['make'],
        'model': car_spec['model'],
        'year': 2016,
        'fuel': 'benzyna',
        'engine_cc': 1200,
        'engine_power': car_spec['engine_power'],
        'mileage': 100000
    }
    try:
        price = predictor.predict(car)['predicted_price']
        budget_prices.append(price)
        print(f"  {car['make']} {car['model']:15s}: {price:>10,.0f} PLN")
    except:
        pass

if premium_prices and budget_prices:
    premium_avg = np.mean(premium_prices)
    budget_avg = np.mean(budget_prices)
    diff = premium_avg - budget_avg
    print(f"\nRóżnica średnia: {diff:,.0f} PLN")
    print(f"Premium jest droższe o: {(diff / budget_avg) * 100:.1f}%")
    print(f"Budget jest tańsze o: {(diff / premium_avg) * 100:.1f}%")

print("\n" + "=" * 80)