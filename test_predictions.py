"""Testy modelu wyceny aut — kontrakt + niezmienniki.

Uruchomienie:
    python test_predictions.py          # standalone (exit 1 przy błędzie)
    pytest test_predictions.py           # jako pytest

Wymaga załadowanego car_price_model.pkl (obok tego pliku) i zależności z requirements.txt.
Niezmienniki to własności, które DOBRY model wyceny musi spełniać (kierunkowo), niezależnie
od konkretnej wersji modelu — jeśli któryś pęknie, to sygnał regresji jakości modelu.
"""
import os
import sys

from car_price_model import CarPricePredictor, extract_voivodeship

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "car_price_model.pkl")

_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        p = CarPricePredictor()
        p.load_model(MODEL_PATH)
        _predictor = p
    return _predictor


def _price(extra=None):
    base = dict(make="audi", model="a5", year=2015, fuel="benzyna", mileage=120000,
                engine_cc=1984, engine_power=190)
    if extra:
        base.update(extra)
    return _get_predictor().predict(base)


# --- Kontrakt odpowiedzi ---
def test_contract_keys():
    r = _price()
    assert "predicted_price" in r and isinstance(r["predicted_price"], float)
    assert "confidence_range" in r and {"min", "max"} <= set(r["confidence_range"])
    assert r["predicted_price"] > 0


def test_confidence_range_contains_prediction():
    r = _price()
    cr = r["confidence_range"]
    assert cr["min"] <= r["predicted_price"] <= cr["max"]
    assert cr["min"] <= cr["max"]


# --- Niezmienniki jakości ---
def test_damaged_cheaper_than_clean():
    assert _price({"is_damaged": True})["predicted_price"] < _price({"is_damaged": False})["predicted_price"]


def test_higher_mileage_not_more_expensive():
    assert _price({"mileage": 300000})["predicted_price"] < _price({"mileage": 120000})["predicted_price"]


def test_newer_more_expensive_than_older():
    assert _price({"year": 2020})["predicted_price"] > _price({"year": 2008})["predicted_price"]


def test_quantile_ordering_if_present():
    # Tylko dla modelu kwantylowego (po retreningu). Stary .pkl pomija ten niezmiennik.
    p = _get_predictor()
    if not p.models:
        return
    import numpy as np
    X = p._build_features_for_predict(dict(make="audi", model="a5", year=2015,
                                           fuel="benzyna", mileage=120000))
    lo = float(np.expm1(p.models["lower"].predict(X)[0]))
    med = float(np.expm1(p.models["median"].predict(X)[0]))
    hi = float(np.expm1(p.models["upper"].predict(X)[0]))
    assert lo <= med <= hi


# --- Pomocnicze ---
def test_voivodeship_extraction():
    assert extract_voivodeship("Warszawa (Mazowieckie)") == "mazowieckie"
    assert extract_voivodeship("brak nawiasu") is None
    assert extract_voivodeship(None) is None


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} testów OK")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
