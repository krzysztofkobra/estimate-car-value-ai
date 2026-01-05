import pandas as pd
import psycopg2
from datetime import datetime

class DataFetcher:
    def __init__(self, db_params):
        self.db_params = db_params

    def get_connection(self):
        return psycopg2.connect(**self.db_params)

    def check_database_stats(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        queries = {
            'Total records': 'SELECT COUNT(*) FROM car_listings',
            'Active records': 'SELECT COUNT(*) FROM car_listings WHERE is_active = true',
            'Inactive records': 'SELECT COUNT(*) FROM car_listings WHERE is_active = false',
            'With price': 'SELECT COUNT(*) FROM car_listings WHERE price IS NOT NULL',
            'NULL prices': 'SELECT COUNT(*) FROM car_listings WHERE price IS NULL',
            'Active + price': 'SELECT COUNT(*) FROM car_listings WHERE is_active = true AND price IS NOT NULL',
            'Price > 0': 'SELECT COUNT(*) FROM car_listings WHERE price > 0'
        }

        print("\nDatabase Statistics:")
        for label, query in queries.items():
            cursor.execute(query)
            count = cursor.fetchone()[0]
            print(f"  {label}: {count:,}")

        cursor.close()
        conn.close()

    def fetch_listings(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM car_listings')
        total_count = cursor.fetchone()[0]
        print(f"\nTotal records in database: {total_count:,}")

        query = """
        SELECT 
            id,
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
            price,
            raw_location,
            seller_type,
            is_damaged,
            color,
            doors,
            nr_of_seats,
            right_hand,
            is_active,
            listing_date,
            first_seen
        FROM car_listings
        """

        print("Fetching data...")
        try:
            df = pd.read_sql(query, conn)
            
            # Normalize text to lowercase
            text_columns = ['make', 'model', 'body_type', 'fuel', 
                            'transmission', 'drive', 'seller_type', 'color']
            for col in text_columns:
                if col in df.columns and df[col].dtype == 'object':
                    df[col] = df[col].str.lower().str.strip()

            print(f"Successfully loaded {len(df):,} listings")
            print(f"Fetched {len(df) / total_count * 100:.1f}% of total records")
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
        finally:
            cursor.close()
            conn.close()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    DB_PARAMS = {
        'dbname': os.getenv('DBNAME'),
        'user': os.getenv('USER'),
        'password': os.getenv('PASSWORD'),
        'host': os.getenv('HOST'),
        'port': os.getenv('PORT')
    }

    fetcher = DataFetcher(DB_PARAMS)

    fetcher.check_database_stats()

    df = fetcher.fetch_listings()

    if not df.empty:
        print("\nDataset Info:")
        print(df.info())

        print("\nSample Data:")
        print(df.head())