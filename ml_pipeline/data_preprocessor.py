import subprocess
import sys

def install_package_with_trusted_hosts(package):
    """Install package bypassing SSL verification"""
    try:
        cmd = [
            sys.executable, "-m", "pip", "install",
            "--trusted-host", "pypi.org",
            "--trusted-host", "pypi.python.org",
            "--trusted-host", "files.pythonhosted.org",
            package
        ]
        subprocess.check_call(cmd)
        print(f"Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")

# Try to import, if fails - install with trusted hosts
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    print("sklearn imported successfully!")
except ImportError:
    print("sklearn not found, installing with SSL bypass...")
    install_package_with_trusted_hosts("scikit-learn")
    # Try importing again
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

# Rest of your code...

# Rest of your code
from feature_engineering.feature_extractor import FeatureExtractor
import pandas as pd
import numpy as np


class DataPreprocessor:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()  # משתמש במחלקה קיימת
        self.scaler = StandardScaler()

    def load_and_clean_data(self):
        """Load data using existing FeatureExtractor and clean it"""
        # Use existing class to load data
        df = self.feature_extractor.load_game_features()

        if df is None:
            return None

        print(f"Original data: {len(df)} games")

        # Remove rows with missing win_rates (keep it simple)
        df_clean = df.dropna(subset=['white_win_rate', 'black_win_rate'])
        print(f"After removing missing win_rates: {len(df_clean)} games")

        return df_clean

    def prepare_features_and_target(self, df):
        """Prepare X (features) and y (target) for ML"""
        # Select features for the model
        feature_columns = [
            'rating_diff',
            'white_win_rate',
            'black_win_rate',
            'white_avg_rating',
            'black_avg_rating'
        ]

        X = df[feature_columns].copy()
        y = df['result'].copy()

        print(f"Features selected: {feature_columns}")
        print(f"Target distribution: {y.value_counts().to_dict()}")

        return X, y

    def split_and_scale_data(self, X, y, test_size=0.2, random_state=42):
        """Split into train/test and scale features"""
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Train set: {len(X_train)} games")
        print(f"Test set: {len(X_test)} games")

        # Scale features (fit on train, transform both)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert back to DataFrames for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def prepare_ml_dataset(self):
        """Main method - prepare complete dataset for ML"""
        print("=== Preparing ML Dataset ===")

        # Load and clean data using existing infrastructure
        df = self.load_and_clean_data()
        if df is None:
            return None

        # Prepare features and target
        X, y = self.prepare_features_and_target(df)

        # Split and scale
        X_train, X_test, y_train, y_test = self.split_and_scale_data(X, y)

        print("=== Dataset Ready for ML ===")
        return X_train, X_test, y_train, y_test


# Test the preprocessor
if __name__ == "__main__":
    preprocessor = DataPreprocessor()

    # Prepare dataset
    data = preprocessor.prepare_ml_dataset()

    if data:
        X_train, X_test, y_train, y_test = data
        print(f"\nFinal dataset shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"Sample features:\n{X_train.head()}")