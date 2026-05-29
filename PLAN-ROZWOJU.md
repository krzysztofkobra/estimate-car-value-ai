# PLAN-ROZWOJU — ML `estimate-car-value-ai` (+ scraper)

Priorytety: **P0** poprawność/trust/nie-psuj-deploya · **P1** sygnał/MLOps/scraper · **P2** explainability/monitoring.

## P0 — Poprawność + trust + deploy
- **Pin requirements.txt** (do wersji z działającego env; zweryfikuj load `.pkl` w czystym obrazie PRZED merge) + fix `analyze_model.py:4-5` (matplotlib/seaborn). ⚠️ pierwsze (reszta pod pinned). [S]
- **Dynamiczny year cap** — `api.py:45` `le=2025` → `field_validator` `current_year+1`. [S]
- **Unifikacja feature-buildera** — `build_features(df, stats, fit)` jeden dla train+predict; fix indentacji (`car_price_model.py:54-79`), usuń duplikat 275-302; zachowaj encoding na train-split (no leakage). [M]
- **`car_age` reference-year** — zapisz `training_year` w payload, użyj spójnie train+infer. [S]
- **log1p target** — `y=np.log1p(price)`, metryki/predict na `expm1`. [S]
- **Real confidence (quantile regression)** — 3 modele P10/P50/P90 (`objective=quantile`), interval=[P10,P90], clamp crossing; resp `confidence_level:0.8`. Backend `AiPredictionService.js:302` insert realny `confidence_score`. [M]
- **Filtr `is_active=true` w treningu** (`car_price_model.py:42-46`) — 966k→225k świeżych; A/B na time-split. [S]
- **Auth na ML /predict** — `X-API-Key` (FastAPI Depends) + key w `AiPredictionService.js`; deploy oba naraz. [S-M]

## P1 — Sygnał + MLOps + scraper
- **Ceny regionalne** — `raw_location` → `voivodeship` categorical (regex `\(([^)]+)\)`) + target-encoded; dodaj do features; `api.py CarInput` przyjmij `location`. [M]
- **Recency/time-trend** — `listing_date`; ostrożnie z inference (brak daty dla hipotetycznego auta) — preferuj active-only (P0) + ewentualnie CPI multiplier post-process. [M]
- **Per-segment MAPE + /health** — loop po price-bands + top-makes → `stats['segments']` (płynie do /health). [S-M]
- **Wersjonowanie** — payload+resp: `model_version, trained_at, training_data_hash, n_train, lightgbm_version`; populuj `valuations.model_version`. [S]
- **Pipeline retreningu** — scheduled + snapshot danych (parquet+hash) + gate MAPE (promuj tylko jeśli ≥ obecny). [L]
- **Model registry / object storage** — `.pkl`→S3/Supabase Storage keyed by version; `*.pkl` do `.gitignore`; (opcjonalnie BFG purge 36MB historii). [M]
- **Testy** — `test_predictions.py` (README go reklamuje, brak): contract (/predict, year 2026, schema), golden (ceny±tolerancja), invariants (P10≤P50≤P90, damaged<clean, więcej-km<mniej). [M]
- **Dockerfile** — HEALTHCHECK, non-root, abs model path; reconcile runtime 3.10 vs README 3.12. [S]
- **README drift** — 636k→225k, 3.12 vs 3.10, usuń phantom test_predictions ref, fix fake-confidence przykład. [S]
- **Scraper hardening** (`autoanaliza/src/services/OtomotoScraper.js`): (1) externalizuj selektory → **JSON-LD/`__NEXT_DATA__` parsing** (stabilniejsze niż hashowane klasy; zweryfikuj live); (2) UA rotation + retry/backoff + jitter; (3) proxy rotation + Cloudflare (headless fallback — skill `browser-use`); (4) VIN extraction; (5) success-rate monitoring + alert. **Zbadać oficjalne API/feed Otomoto** (strukturalne ryzyko). Re-enable inactive cron (`manager.js:2517`). Persystentna kolejka (BullMQ). [L]

## P2 — Explainability + monitoring
- **SHAP per-prediction** — `pred_contrib=True` → top kontrybutorzy ("+4200 niski przebieg, −9800 uszkodzony"); resp `explanation`; premium differentiator. [M]
- **Prediction logging** — substrat w `valuations` istnieje; dopełnij confidence_score+model_version; structured log po stronie ML. [S]
- **Drift monitoring** — join `valuations` ⋈ `car_listings` (relist/sale) → realny MAPE w czasie → trigger retreningu; + input-feature drift. [L]

## ZREALIZOWANE — FAZA 9 (kontynuacja, weryfikowane na zainstalowanych libach)
- **requirements.txt PRZYPIĘTE** do wersji zweryfikowanych: ładują commitowany `.pkl` + predict + start API (lightgbm 4.6.0, scikit-learn 1.8.0, pandas 3.0.3, numpy 2.4.6, joblib 1.5.3, fastapi 0.136.3, pydantic 2.13.4, …).
- **X-API-Key na /predict** — opt-in (`API_KEY` env; gdy nieustawiony → auth wyłączony, backward-compat). Zweryfikowane: brak/zły klucz → 401, dobry → przechodzi.
- **Ceny regionalne (kod, backward-compatible)** — `voivodeship` z `raw_location` (`extract_voivodeship`, regex), dodany do `feature_cols`/categorical w treningu + ścieżce predykcji; aktywne dopiero po retreningu (stary `.pkl` ignoruje). `CarInput.location` może je zasilić — **TODO: dodać pole do CarInput** (obecnie z `raw_location`).
- **Wersjonowanie** — `stats`: `lightgbm_version`, `training_data_hash` (sha256 X_train), `model_version` (`<data>_qP10P50P90`); zwracane w `/predict`.
- **SHAP / explanation** — `predict` zwraca top kontrybutorów (`pred_contrib`), defensywnie (błąd → None).
- **Dockerfile** — non-root (uid 10001), HEALTHCHECK (/health, stdlib), domyślny `PORT`, abs ścieżka modelu (`MODEL_PATH`).
- **psycopg2 import leniwy** (API nie wymaga sterownika DB) + `_read_sql` fallback (pandas 3.x bez SQLAlchemy).
- **test_predictions.py** — kontrakt + niezmienniki (damaged<clean, +przebieg→taniej, nowszy>starszy, P10≤P50≤P90, voivodeship); 7/7 OK; **dodane do CI** (pip install + run).
- **README drift** — 3.12→3.10, 636k→225k aktywnych, 18→19 cech, przykład odpowiedzi (confidence_level/model_version/explanation), zakres lat dynamiczny.

## Wymaga zewn./infra (flaga — NADAL otwarte)
Creds do live Postgres (**retrening** — kwantyle/log/regionalne aktywują się po nim; obecny `.pkl` to stary model) · object storage (registry, `.pkl` poza gitem) · proxy pool (scraper) · Redis (kolejka) · scheduler (retraining/cron) · drift monitoring (join valuations⋈car_listings) · znalezienie bulk-ingestion pipeline (poza repo).
