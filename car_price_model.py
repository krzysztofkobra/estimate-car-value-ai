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
        self.model = None
        self.feature_columns = None
        self.categorical_features = None
        self.stats = {}

    def connect_db(self, connection_params):
        return psycopg2.connect(**connection_params)

    def load_data(self, connection_params):
        conn = self.connect_db(connection_params)

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
        """

        df = pd.read_sql(query, conn)
        conn.close()

        print(f"Loaded {len(df)} records from database")
        return df

    def prepare_features(self, df):
        df = df.copy()
    
        # NORMALIZUJ TEKST DO LOWERCASE
        text_columns = ['make', 'model', 'body_type', 'fuel', 
                        'transmission', 'drive', 'seller_type', 'color']
        for col in text_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.lower().str.strip()
    
            if 'year' in df.columns:
                df['car_age'] = datetime.now().year - df['year']

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

        print("Engineering features...")
        df = self.prepare_features(df)

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

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
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

        X_train = apply_encodings(X_train_raw)
        X_test = apply_encodings(X_test_raw)

        X_train = X_train[feature_cols]
        X_test = X_test[feature_cols]

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

        print(f"\nDataset statistics (Train set):")
        print(f"  Records: {len(X_train)}")
        print(f"  Price range: {self.stats['min_price']:.0f} - {self.stats['max_price']:.0f} PLN")
        print(f"  Mean price: {self.stats['mean_price']:.0f} PLN")

        print(f"\nTraining set: {len(X_train)} records")
        print(f"Test set: {len(X_test)} records")

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=self.categorical_features,
            free_raw_data=False
        )

        test_data = lgb.Dataset(
            X_test,
            label=y_test,
            categorical_feature=self.categorical_features,
            reference=train_data,
            free_raw_data=False
        )

        params = {
            'objective': 'regression',
            'metric': 'mae',
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

        print("\nTraining model...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'test'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )

        print("\nEvaluating model...")
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        train_mape = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
        test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE")
        print("=" * 60)
        print(f"\nTrain Set:")
        print(f"  MAE:  {train_mae:,.0f} PLN")
        print(f"  RMSE: {train_rmse:,.0f} PLN")
        print(f"  R²:   {train_r2:.4f}")
        print(f"  MAPE: {train_mape:.2f}%")

        print(f"\nTest Set:")
        print(f"  MAE:  {test_mae:,.0f} PLN")
        print(f"  RMSE: {test_rmse:,.0f} PLN")
        print(f"  R²:   {test_r2:.4f}")
        print(f"  MAPE: {test_mape:.2f}%")

        print("\n" + "=" * 60)
        print("TOP 20 FEATURE IMPORTANCE")
        print("=" * 60)
        importance = pd.DataFrame({
            'feature': self.model.feature_name(),
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        for idx, row in importance.head(20).iterrows():
            print(f"  {row['feature']:25s} {row['importance']:10.0f}")

        self.stats.update({
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_mape': float(train_mape),
            'test_mape': float(test_mape)
        })

        return {
            'train': {'mae': train_mae, 'rmse': train_rmse, 'r2': train_r2, 'mape': train_mape},
            'test': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2, 'mape': test_mape}
        }

    def predict(self, car_data):
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        df = pd.DataFrame([car_data])

        # NORMALIZUJ WEJŚCIE DO LOWERCASE
        # To kluczowe, bo model był trenowany na małych literach
        text_columns = ['make', 'model', 'body_type', 'fuel', 
                        'transmission', 'drive', 'seller_type', 'color']
        for col in text_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.lower().str.strip()

        if 'car_age' not in df.columns and 'year' in df.columns:
            df['car_age'] = datetime.now().year - df['year']

        if 'mileage_per_year' not in df.columns:
            df['mileage_per_year'] = df['mileage'] / (df['car_age'] + 1)

        if 'power_to_cc_ratio' not in df.columns and 'engine_power' in df.columns and 'engine_cc' in df.columns:
            df['power_to_cc_ratio'] = df['engine_power'] / (df['engine_cc'] + 1)

        global_median = self.stats.get('global_median_price', 50000)

        if 'make_avg_price' not in df.columns:
            make_encoding = self.stats.get('make_encoding', {})
            df['make_avg_price'] = df['make'].map(make_encoding).fillna(global_median)

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
                if col in self.categorical_features:
                    df[col] = None
                else:
                    df[col] = 0

        X = df[self.feature_columns].copy()

        for col in self.categorical_features:
            X[col] = X[col].astype('category')

        prediction = self.model.predict(X)[0]

        return {
            'predicted_price': float(prediction),
            'confidence_range': {
                'min': float(prediction * 0.85),
                'max': float(prediction * 1.15)
            }
        }

    def save_model(self, filepath='car_price_model.pkl'):
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'categorical_features': self.categorical_features,
            'stats': self.stats
        }
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to {filepath}")

    def load_model(self, filepath='car_price_model.pkl'):
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.categorical_features = model_data['categorical_features']
        self.stats = model_data['stats']
        print(f"Model loaded from {filepath}")


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
        'make': 'audi',
        'model': 'a5',
        'year': 2009,
        'body_type': 'coupe',
        'fuel': 'benzyna',
        'engine_cc': 1984,
        'engine_power': 211,
        'transmission': 'manualna',
        'drive': 'fwd',
        'mileage': 345000,
        'seller_type': 'private',
        'is_damaged': False
    }

    result = predictor.predict(test_car)
    print(f"\n\nExample prediction for Audi A5 2009:")
    print(f"  Predicted price: {result['predicted_price']:,.0f} PLN")
    print(
        f"  Confidence range: {result['confidence_range']['min']:,.0f} - {result['confidence_range']['max']:,.0f} PLN")