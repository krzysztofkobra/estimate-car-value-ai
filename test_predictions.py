from car_price_model import CarPricePredictor

predictor = CarPricePredictor()

predictor.load_model('car_price_model.pkl')

test_cases = [
    {
        'name': 'Audi A5 2013 - wszystkie dane',
        'data': {
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
            'is_damaged': False
        }
    },
    {
        'name': 'Toyota Corolla 2015 - minimalne dane',
        'data': {
            'make': 'Toyota',
            'model': 'Corolla',
            'year': 2015,
            'fuel': 'benzyna',
            'mileage': 100000
        }
    },
    {
        'name': 'BMW X5 2018 - premium SUV',
        'data': {
            'make': 'BMW',
            'model': 'X5',
            'year': 2018,
            'fuel': 'diesel',
            'engine_cc': 2993,
            'engine_power': 265,
            'transmission': 'Automatyczna',
            'drive': 'AWD',
            'mileage': 80000,
            'seller_type': 'dealer',
            'is_damaged': False
        }
    },
    {
        'name': 'VW Golf 2010 - uszkodzony',
        'data': {
            'make': 'Volkswagen',
            'model': 'Golf',
            'year': 2010,
            'fuel': 'diesel',
            'engine_cc': 1968,
            'engine_power': 140,
            'mileage': 200000,
            'is_damaged': True
        }
    },
    {
        'name': 'Tesla Model 3 2020 - elektryczny',
        'data': {
            'make': 'Tesla',
            'model': 'Model 3',
            'year': 2020,
            'fuel': 'elektryczny',
            'engine_power': 283,
            'mileage': 50000,
            'seller_type': 'dealer'
        }
    },
    {
        'name': 'Toyota Supra 1998 - RHD import z Japonii',
        'data': {
            'make': 'Toyota',
            'model': 'Supra',
            'year': 1998,
            'fuel': 'benzyna',
            'engine_cc': 2997,
            'engine_power': 280,
            'mileage': 120000,
            'right_hand': True,
            'color': 'Czerwony'
        }
    }
]

print("="*80)
print("TESTY PREDYKCJI CEN SAMOCHODÓW")
print("="*80)

for test in test_cases:
    print(f"\n{test['name']}")
    print("-" * 80)
    
    for key, value in test['data'].items():
        print(f"  {key:20s}: {value}")
    
    try:
        result = predictor.predict(test['data'])
        
        print(f"\n  {'Predicted Price:':20s} {result['predicted_price']:,.0f} PLN")
        print(f"  {'Confidence Range:':20s} {result['confidence_range']['min']:,.0f} - {result['confidence_range']['max']:,.0f} PLN")
        
    except Exception as e:
        print(f"\n  ERROR: {e}")

print("\n" + "="*80)
print("\nModel statistics:")
for key, value in predictor.stats.items():
    if isinstance(value, float):
        if 'price' in key:
            print(f"  {key:20s}: {value:,.0f}")
        else:
            print(f"  {key:20s}: {value:.4f}")
