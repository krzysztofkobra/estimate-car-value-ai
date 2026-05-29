import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import psycopg2
from datetime import datetime
import json


class CarPricePredictor:
    def __init__(self):
        self.model = None          # P50 (median) booster — używany też przez /health i feature_importance
        self.models = None         # quantile boosters: {'lower':P10, 'median':P50, 'upper':P90}
        self.feature_columns = None
        self.categorical_features = None
        self.stats = {}

    def connect_db(self, connection_params):
        return psycopg2.connect(**connection_params)

    def load_data(self, connection_params):
        conn = self.connect_db(connection_params)

        # Filtr is_active = true: trenujemy na AKTUALNYM rynku (wcześniej ~77% danych to sprzedane/stare ogłoszenia).
        query = """
        SELECT
            make,
            model,
            year,
            body_type,
            fuel,
            engine_cc,
            engine_power,
            transmission,
            drive,
            mileage,
            seller_type,
            is_damaged,
            color,
            right_hand,
            price
        FROM car_listings
        WHERE price > 1000
            AND price < 3000000
            AND mileage >= 0
            AND is_active = true
        """

        df = pd.read_sql(query, conn)
        conn.close()

        print(f"Loaded {len(df)} records from database")
        return df

    def prepare_features(self, df, reference_year=None):
        df = df.copy()
        if reference_year is None:
            reference_year = datetime.now().year

        # Normalizacja tekstu
        text_columns = ['make', 'model', 'body_type', 'fuel',
                        'transmission', 'drive', 'seller_type', 'color']
        for col in text_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.lower().str.strip()

        # Cechy inżynierowane — DEDENTOWANE poza pętlę (wcześniej błędnie liczone N razy w pętli text_columns).
        # car_age liczone względem reference_year (zapisanego przy treningu) — spójne train/inference, brak driftu.
        if 'year' in df.columns:
            df['car_age'] = reference_year - df['year']

        if 'mileage' in df.columns and 'car_age' in df.columns:
            df['mileage_per_year'] = df['mileage'] / (df['car_age'] + 1)

        if 'engine_power' in df.columns and 'engine_cc' in df.columns:
            df['power_to_cc_ratio'] = df['engine_power'] / (df['engine_cc'] + 1)

        if 'right_hand' in df.columns:
            df['right_hand'] = df['right_hand'].fillna(False)
            if df['right_hand'].dtype == 'object':
                df['right_hand'] = df['right_hand'].astype(str).str.lower().isin(['true', '1', 't'])
            df['right_hand'] = df['right_hand'].astype(bool)

        return df

    def train(self, connection_params, test_size=0.2, random_state=42):
        print("Loading data...")
        df = self.load_data(connection_params)

        reference_year = datetime.now().year
        self.stats['reference_year'] = reference_year
        self.stats['training_date'] = datetime.now().isoformat()

        print("Engineering features...")
        df = self.prepare_features(df, reference_year=reference_year)

        feature_cols = [
            'make', 'model', 'year', 'body_type', 'fuel',
            'engine_cc', 'engine_power', 'transmission', 'drive',
            'mileage', 'seller_type', 'is_damaged', 'color', 'right_hand',
            'car_age', 'mileage_per_year', 'power_to_cc_ratio',
            'make_avg_price', 'model_avg_price'
        ]

        self.categorical_features = [
            'make', 'model', 'body_type', 'fuel',
            'transmission', 'drive', 'seller_type', 'color'
        ]

        X = df.drop(columns=['price'])
        y = df['price']
        # log1p target — błąd skaluje się z ceną (przedziały multiplikatywne, lepsze na drogich autach).
        y_log = np.log1p(y)

        X_train_raw, X_test_raw, y_train, y_test, y_train_log, y_test_log = train_test_split(
            X, y, y_log, test_size=test_size, random_state=random_state
        )

        print("Calculating target encodings (Training set only)...")

        train_joined = X_train_raw.copy()
        train_joined['price'] = y_train

        global_median = y_train.median()
        self.stats['global_median_price'] = float(global_median)

        make_stats = train_joined.groupby('make')['price'].median()
        self.stats['make_encoding'] = make_stats.to_dict()

        model_stats = train_joined.groupby(['make', 'model'])['price'].median()
        self.stats['model_encoding'] = model_stats.to_dict()

        def apply_encodings(data_df):
            data = data_df.copy()
            data['make_avg_price'] = data['make'].map(self.stats['make_encoding']).fillna(global_median)

            def get_model_price(row):
                val = self.stats['model_encoding'].get((row['make'], row['model']))
                if pd.isna(val):
                    val = self.stats['make_encoding'].get(row['make'])
                if pd.isna(val):
                    val = global_median
                return val

            data['model_avg_price'] = data.apply(get_model_price, axis=1)
            return data

        X_train = apply_encodings(X_train_raw)[feature_cols]
        X_test = apply_encodings(X_test_raw)[feature_cols]

        for col in self.categorical_features:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')

        self.feature_columns = feature_cols

        self.stats.update({
            'mean_price': float(y_train.mean()),
            'median_price': float(y_train.median()),
            'std_price': float(y_train.std()),
            'min_price': float(y_train.min()),
            'max_price': float(y_train.max())
        })

        print(f"\nTraining set: {len(X_train)} records | Test set: {len(X_test)} records")

        # Trenujemy na LOG-celu.
        train_data = lgb.Dataset(X_train, label=y_train_log,
                                 categorical_feature=self.categorical_features, free_raw_data=False)
        test_data = lgb.Dataset(X_test, label=y_test_log,
                                categorical_feature=self.categorical_features, reference=train_data, free_raw_data=False)

        base_params = {
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 50,
            'max_depth': 15,
            'verbose': -1,
            'force_row_wise': True
        }

        # 3 modele kwantylowe → realny przedział ufności [P10, P90] (~80% pokrycia), zamiast sztywnego ±15%.
        quantiles = {'lower': 0.1, 'median': 0.5, 'upper': 0.9}
        self.models = {}
        for name, alpha in quantiles.items():
            print(f"\nTraining quantile model: {name} (alpha={alpha})...")
            params_q = {**base_params, 'objective': 'quantile', 'alpha': alpha, 'metric': 'quantile'}
            self.models[name] = lgb.train(
                params_q, train_data, num_boost_round=1000,
                valid_sets=[test_data], valid_names=['test'],
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=200)]
            )

        # P50 jako model główny (predykcja punktowa + feature importance + /health)
        self.model = self.models['median']

        print("\nEvaluating model (na oryginalnej skali cen)...")
        y_pred_test = np.expm1(self.models['median'].predict(X_test))
        y_pred_train = np.expm1(self.models['median'].predict(X_train))

        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_mape = float(np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100)
        train_mape = float(np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100)

        # Pokrycie przedziału [P10, P90] na teście (powinno być ~0.80).
        p_low = np.expm1(self.models['lower'].predict(X_test))
        p_high = np.expm1(self.models['upper'].predict(X_test))
        coverage = float(np.mean((y_test >= np.minimum(p_low, p_high)) & (y_test <= np.maximum(p_low, p_high))))

        # MAPE per pasmo cenowe (jeden globalny MAE ukrywa, że drogie/rzadkie auta są dużo gorsze).
        bands = [(0, 30000), (30000, 60000), (60000, 100000), (100000, 200000), (200000, 3000000)]
        segments = {}
        for lo, hi in bands:
            mask = (y_test >= lo) & (y_test < hi)
            if mask.sum() > 0:
                seg_mape = float(np.mean(np.abs((y_test[mask] - y_pred_test[mask]) / y_test[mask])) * 100)
                seg_cov = float(np.mean((y_test[mask] >= np.minimum(p_low, p_high)[mask]) &
                                        (y_test[mask] <= np.maximum(p_low, p_high)[mask])))
                segments[f"{lo}-{hi}"] = {'count': int(mask.sum()), 'mape': round(seg_mape, 2), 'coverage': round(seg_cov, 3)}

        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE (P50, oryginalna skala)")
        print("=" * 60)
        print(f"  Test MAE:  {test_mae:,.0f} PLN")
        print(f"  Test RMSE: {test_rmse:,.0f} PLN")
        print(f"  Test R²:   {test_r2:.4f}")
        print(f"  Test MAPE: {test_mape:.2f}%")
        print(f"  Pokrycie [P10,P90]: {coverage:.3f} (cel ~0.80)")
        print(f"  Segmenty: {json.dumps(segments, ensure_ascii=False)}")

        self.stats.update({
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_mape': float(train_mape),
            'test_mape': float(test_mape),
            'interval_coverage': coverage,
            'segments': segments,
            'n_train': int(len(X_train)),
            'confidence_level': 0.8
        })

        return {'test': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2, 'mape': test_mape, 'coverage': coverage}}

    def _build_features_for_predict(self, car_data):
        df = pd.DataFrame([car_data])
        reference_year = self.stats.get('reference_year', datetime.now().year)

        text_columns = ['make', 'model', 'body_type', 'fuel',
                        'transmission', 'drive', 'seller_type', 'color']
        for col in text_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.lower().str.strip()

        if 'car_age' not in df.columns and 'year' in df.columns:
            df['car_age'] = reference_year - df['year']
        if 'mileage_per_year' not in df.columns:
            df['mileage_per_year'] = df['mileage'] / (df['car_age'] + 1)
        if 'power_to_cc_ratio' not in df.columns and 'engine_power' in df.columns and 'engine_cc' in df.columns:
            df['power_to_cc_ratio'] = df['engine_power'] / (df['engine_cc'] + 1)

        global_median = self.stats.get('global_median_price', 50000)
        if 'make_avg_price' not in df.columns:
            df['make_avg_price'] = df['make'].map(self.stats.get('make_encoding', {})).fillna(global_median)
        if 'model_avg_price' not in df.columns:
            model_encoding = self.stats.get('model_encoding', {})
            make_encoding = self.stats.get('make_encoding', {})

            def get_model_price(row):
                val = model_encoding.get((row['make'], row['model']))
                if val is None or pd.isna(val):
                    val = make_encoding.get(row['make'])
                if val is None or pd.isna(val):
                    val = global_median
                return val

            df['model_avg_price'] = df.apply(get_model_price, axis=1)

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = None if col in self.categorical_features else 0

        X = df[self.feature_columns].copy()
        for col in self.categorical_features:
            X[col] = X[col].astype('category')
        return X

    def predict(self, car_data):
        if self.model is None and not self.models:
            raise ValueError("Model not trained yet. Call train() first.")

        X = self._build_features_for_predict(car_data)

        # NOWA ścieżka: modele kwantylowe (trenowane na log-celu) → realny przedział [P10, P90].
        if self.models:
            p_med = float(np.expm1(self.models['median'].predict(X)[0]))
            p_low = float(np.expm1(self.models['lower'].predict(X)[0]))
            p_high = float(np.expm1(self.models['upper'].predict(X)[0]))
            lo, hi = sorted([p_low, p_high])  # zabezpieczenie przed quantile-crossing
            return {
                'predicted_price': p_med,
                'confidence_range': {'min': max(0.0, lo), 'max': hi},
                'confidence_level': self.stats.get('confidence_level', 0.8),
                'model_version': self.stats.get('training_date')
            }

        # BACKWARD-COMPAT: stary pojedynczy model (raw price) — sztywny ±15% jak dawniej.
        prediction = float(self.model.predict(X)[0])
        return {
            'predicted_price': prediction,
            'confidence_range': {'min': float(prediction * 0.85), 'max': float(prediction * 1.15)}
        }

    def save_model(self, filepath='car_price_model.pkl'):
        model_data = {
            'model': self.model,
            'models': self.models,
            'feature_columns': self.feature_columns,
            'categorical_features': self.categorical_features,
            'stats': self.stats
        }
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to {filepath}")

    def load_model(self, filepath='car_price_model.pkl'):
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.models = model_data.get('models')  # None dla starego .pkl (backward-compat)
        self.feature_columns = model_data['feature_columns']
        self.categorical_features = model_data['categorical_features']
        self.stats = model_data['stats']
        print(f"Model loaded from {filepath} (quantile: {'tak' if self.models else 'nie (stary model)'})")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    connection_params = {
        'dbname': os.getenv('DBNAME'),
        'user': os.getenv('USER'),
        'password': os.getenv('PASSWORD'),
        'host': os.getenv('HOST'),
        'port': os.getenv('PORT')
    }

    predictor = CarPricePredictor()
    metrics = predictor.train(connection_params)
    predictor.save_model('car_price_model.pkl')

    test_car = {
        'make': 'audi', 'model': 'a5', 'year': 2009, 'body_type': 'coupe',
        'fuel': 'benzyna', 'engine_cc': 1984, 'engine_power': 211,
        'transmission': 'manualna', 'drive': 'fwd', 'mileage': 345000,
        'seller_type': 'private', 'is_damaged': False
    }

    result = predictor.predict(test_car)
    print(f"\n\nExample prediction for Audi A5 2009:")
    print(f"  Predicted price: {result['predicted_price']:,.0f} PLN")
    print(f"  Confidence range: {result['confidence_range']['min']:,.0f} - {result['confidence_range']['max']:,.0f} PLN")
