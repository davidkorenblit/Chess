from database.db_manager import DatabaseManager
import pandas as pd


class FeatureExtractor:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.db_manager.connect()

    def load_game_features(self):
        """Load all features from the game_features view"""
        try:
            query = "SELECT * FROM game_features"

            # Get data from database
            cursor = self.db_manager.connection.cursor()
            cursor.execute(query)
            data = cursor.fetchall()

            # Get column names
            columns = [desc[0] for desc in cursor.description]
            cursor.close()

            # Create DataFrame
            df = pd.DataFrame(data, columns=columns)
            print(f"Loaded {len(df)} games with {len(df.columns)} features")
            print(f"Columns: {list(df.columns)}")

            return df

        except Exception as e:
            print(f"Error loading features: {e}")
            return None

    def check_data_quality(self, df):
        """Basic data quality checks"""
        print("\n=== Data Quality Check ===")
        print(f"Total games: {len(df)}")
        print(f"Missing values per column:")
        print(df.isnull().sum())
        print(f"\nResult distribution:")
        print(df['result'].value_counts())
        print(f"\nRating diff range: {df['rating_diff'].min()} to {df['rating_diff'].max()}")

    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'db_manager'):
            self.db_manager.disconnect()


# Test the feature extractor
if __name__ == "__main__":
    extractor = FeatureExtractor()

    # Load features
    df = extractor.load_game_features()

    if df is not None:
        # Check data quality
        extractor.check_data_quality(df)

        # Show sample
        print("\nSample data:")
        print(df.head())