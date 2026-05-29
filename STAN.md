# STAN — ML `estimate-car-value-ai`

FastAPI + LightGBM mikroserwis predykcji ceny aut. Wołany przez backend (`AiPredictionService` → `POST {PREDICTION_API_URL}/predict`). Brak testów.

## Pliki
`api.py` (FastAPI), `car_price_model.py` (model+trening), `car_price_model.pkl` (17MB, **commitowany w 9 commitach → .git ~36MB**), `fetch_data.py` (EDA, ~nieużywany), `analyze_model.py` (CLI EDA), `Dockerfile`, `requirements.txt` (**niepinowany**), `runtime.txt` (3.10.15, README mówi 3.12), `README.md` (drift).

## API (kontrakt — krytyczny punkt integracji)
- **`POST /predict`** — req `CarInput`: wymagane `make, model, year(⚠️ ge=1990, le=2025), fuel, mileage`; opcjonalne `body_type, engine_cc(500-10000), engine_power(30-1000), transmission, drive, seller_type(def private), is_damaged(def false), color, right_hand`. Resp: `{predicted_price, confidence_range:{min,max}, input_data}`.
- `GET /health` — `{status, model_stats}` (statyczne metryki treningowe, 503 gdy model niezaładowany).
- Model ładowany globalnie przy starcie (relative path, CWD-zależny). **Brak auth na /predict.** CORS: tylko autoanaliza.pl.

## Model
LightGBM (Booster), joblib. Target `price` (RAW, **bez log-transform** — błąd skaluje źle na drogich autach 1k-9.99M). Filtr `price 1k-3M, mileage>=0`. **19 cech** (README mówi 18): base + engineered (`car_age, mileage_per_year, power_to_cc_ratio, make_avg_price, model_avg_price`). 8 categorical. Target-encoding na train split (OK). Params: num_leaves 127, lr 0.05, max_depth 15, 1000 rounds, early-stop 50. Metryki: MAE ~8277zł, R²~0.9, MAPE liczony ale tylko agregat.

## Dane
Źródło: Postgres `car_listings` (psycopg2, creds z `.env`). **966k wierszy, 225k aktywnych — trening NIE filtruje `is_active` → 77% danych to sprzedane/stare!** `raw_location` 100% wypełnione ("City (Voivodeship)", 16 województw) — **nieużywane (brak cen regionalnych)**. `listing_date` 100% — nieużywane. `vin/doors/seats` ~0 wierszy (blokuje scraper).

## Krytyczne bugi
1. **`year le=2025`** (`api.py:45`) — odrzuca auta 2026 DZIŚ (776 wierszy ma year>2025). 2. **confidence_range FAKE** — sztywne `*0.85/*1.15` (`car_price_model.py:320-323`). 3. **Indentacja** `prepare_features` (54-79) — bloki engineered wewnątrz `for col in text_columns` (fragile; train vs predict 275-302 zdublowane/rozbieżne). 4. **`car_age` drift** — `datetime.now().year` przy train+infer. 5. **requirements niepinowane** → joblib/pickle version drift = cichy outage (model→None→503). 6. `analyze_model.py:4-5` importuje matplotlib/seaborn (brak w requirements) → ImportError clean install. 7. `.pkl` w gicie; brak wersjonowania (model_version/date/hash). 8. Brak retreningu (manualny ~miesięcznie), brak snapshotu danych (nie-reprodukowalne), brak drift monitoring.

## Scraper (backend `OtomotoScraper.js` — feeduje dane)
Single `axios.get`+cheerio. **Anty-bot: ZERO** (UA Chrome91/2021, brak proxy/retry/Cloudflare). Selektory = hashowane klasy build (`h3.efzkujb1`…) → kruche. Otomoto Cloudflare-chronione → blok. Kolejka in-memory. Bulk-ingestion 966k **POZA tym repo** (osobny proces — znaleźć).

## Substrat istniejący (do wykorzystania)
`valuations` ma już kolumny `price_min/max`, `confidence_score` (NULL), `model_version` (nieużywana) → gotowe pod real confidence + wersjonowanie + prediction logging.

Pula zadań: `PLAN-ROZWOJU.md`.
