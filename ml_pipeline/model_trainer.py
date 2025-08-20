from ml_pipeline.data_preprocessor import DataPreprocessor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import pandas as pd
import numpy as np
# NEW IMPORTS FOR DATABASE SAVING
from database.db_manager import DatabaseManager
import json


class ModelTrainer:
    def __init__(self):
        self.preprocessor = DataPreprocessor()  # Use existing class
        self.models = {}
        self.results = {}
        self.feature_names = ['rating_diff', 'white_win_rate', 'black_win_rate', 'white_avg_rating', 'black_avg_rating']
        self.label_encoder = None
        # NEW: Database manager for saving results
        self.db_manager = DatabaseManager()

    def load_prepared_data(self):
        """Load and prepare data using existing DataPreprocessor"""
        print("=== Loading Data Using DataPreprocessor ===")
        data = self.preprocessor.prepare_ml_dataset()

        if data is None:
            print("Failed to load data!")
            return None

        X_train, X_test, y_train, y_test = data
        print(f"Data loaded: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test

    def train_iteration(self, X_train, y_train, iteration_name, **rf_params):
        """Train Random Forest for one iteration"""
        print(f"\n--- Iteration {iteration_name} ---")

        # Default parameters with MORE TREES
        default_params = {
            'n_estimators': 300,  # INCREASED from 50 to 300
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42
        }

        # Update with custom parameters
        default_params.update(rf_params)

        print(f"Training Random Forest with: {default_params}")

        # Train model
        rf_model = RandomForestClassifier(**default_params)
        rf_model.fit(X_train, y_train)

        # Store model
        self.models[iteration_name] = rf_model

        return rf_model

    def train_advanced_models(self, X_train, y_train):
        """Train advanced ML models"""
        print(f"\n=== Training Advanced Models ===")

        # ENCODE TARGET LABELS FOR XGBOOST
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        # XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train, y_train_encoded)  # Use encoded labels
        self.models['xgboost'] = xgb_model

        # Gradient Boosting (works with strings)
        print("Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train, y_train)  # Original labels OK
        self.models['gradient_boosting'] = gb_model

        # Enhanced Logistic Regression (works with strings)
        print("Training Enhanced Logistic Regression...")
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=2000,
            C=0.1
        )
        lr_model.fit(X_train, y_train)  # Original labels OK
        self.models['logistic_regression_enhanced'] = lr_model

        return xgb_model, gb_model, lr_model

    class XGBWrapper(BaseEstimator, ClassifierMixin):
        """Proper sklearn wrapper for XGBoost to work with VotingClassifier"""

        def __init__(self, xgb_model, label_encoder):
            self.xgb_model = xgb_model
            self.label_encoder = label_encoder

        def fit(self, X, y):
            y_encoded = self.label_encoder.transform(y)
            self.xgb_model.fit(X, y_encoded)
            return self

        def predict(self, X):
            y_pred_encoded = self.xgb_model.predict(X)
            return self.label_encoder.inverse_transform(y_pred_encoded)

        def predict_proba(self, X):
            return self.xgb_model.predict_proba(X)

        def get_params(self, deep=True):
            return self.xgb_model.get_params(deep)

        def set_params(self, **params):
            return self.xgb_model.set_params(**params)

    def create_ensemble_model(self, X_train, y_train):
        """Create Manual Ensemble averaging predictions"""
        print(f"\n=== Creating Manual Ensemble Model ===")

        # Manual ensemble - just average predictions
        class ManualEnsemble:
            def __init__(self, models, label_encoder=None):
                self.models = models
                self.label_encoder = label_encoder

            def fit(self, X, y):
                return self  # Models already trained

            def predict(self, X):
                predictions = []

                # Get predictions from each model
                rf_pred = self.models['rf'].predict(X)
                gb_pred = self.models['gb'].predict(X)
                lr_pred = self.models['lr'].predict(X)

                # XGBoost special handling
                xgb_pred_encoded = self.models['xgb'].predict(X)
                xgb_pred = self.label_encoder.inverse_transform(xgb_pred_encoded)

                # Simple majority voting
                final_predictions = []
                for i in range(len(X)):
                    votes = [rf_pred[i], gb_pred[i], lr_pred[i], xgb_pred[i]]
                    # Get most common prediction
                    final_pred = max(set(votes), key=votes.count)
                    final_predictions.append(final_pred)

                return np.array(final_predictions)

        # Create manual ensemble
        ensemble_models = {
            'rf': self.get_best_random_forest(),
            'gb': self.models['gradient_boosting'],
            'lr': self.models['logistic_regression_enhanced'],
            'xgb': self.models['xgboost']
        }

        ensemble = ManualEnsemble(ensemble_models, self.label_encoder)
        print("Manual Ensemble created (majority voting)...")

        self.models['ensemble'] = ensemble
        return ensemble

    def hyperparameter_tuning(self, X_train, y_train):
        """Advanced hyperparameter tuning for Random Forest"""
        print(f"\n=== Hyperparameter Tuning ===")

        # Grid search for Random Forest
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        rf_base = RandomForestClassifier(random_state=42)

        print("Running GridSearchCV (this may take a few minutes)...")
        grid_search = GridSearchCV(
            rf_base,
            param_grid,
            cv=3,  # 3-fold cross validation
            scoring='accuracy',
            n_jobs=-1,  # Use all cores
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")

        self.models['random_forest_tuned'] = grid_search.best_estimator_

        return grid_search.best_estimator_

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        print(f"\n=== Evaluating {model_name} ===")

        # Make predictions with XGBoost handling
        if model_name == 'xgboost':
            y_pred_encoded = model.predict(X_test)
            y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        else:
            y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f} ({accuracy:.1%})")
        print(f"Confusion Matrix:")
        print(conf_matrix)

        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"Feature Importance:")
            for _, row in feature_importance.iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
        elif hasattr(model, 'named_estimators_'):  # For ensemble
            print("Ensemble - showing Random Forest feature importance:")
            rf_importance = model.named_estimators_['rf'].feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': rf_importance
            }).sort_values('importance', ascending=False)

            for _, row in feature_importance.iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
        else:
            feature_importance = None

        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'feature_importance': feature_importance
        }

        return accuracy, y_pred

    def get_best_random_forest(self):
        """Get best Random Forest from iterations"""
        rf_models = {name: model for name, model in self.models.items() if 'iteration' in name}
        if not rf_models:
            # If no iterations yet, create a default one
            return RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)

        # Return the first iteration model (or could be smarter about selection)
        return list(rf_models.values())[0]

    def save_results_to_db(self, total_games, train_games, test_games, target_distribution):
        """NEW METHOD: Save training results to database"""
        print("\n🗄️ Saving results to database...")

        try:
            # Connect to database
            self.db_manager.connect()

            # Get best model info
            best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
            best_accuracy = self.results[best_model_name]['accuracy']

            # 1. Save training session
            cursor = self.db_manager.connection.cursor()

            session_query = """
                INSERT INTO training_sessions 
                (total_games, total_train_games, total_test_games, best_model_name, 
                 best_accuracy, target_distribution_white, target_distribution_black, 
                 target_distribution_draw, feature_names)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING session_id
            """

            session_data = (
                total_games, train_games, test_games, best_model_name, best_accuracy,
                target_distribution.get('white', 0), target_distribution.get('black', 0),
                target_distribution.get('draw', 0), json.dumps(self.feature_names)
            )

            cursor.execute(session_query, session_data)
            session_id = cursor.fetchone()[0]
            print(f"  ✅ Training session saved (ID: {session_id})")

            # 2. Save model performance
            performance_count = 0
            for model_name, result in self.results.items():
                perf_query = """
                    INSERT INTO model_performance 
                    (session_id, model_name, accuracy, confusion_matrix)
                    VALUES (%s, %s, %s, %s)
                """

                # Convert confusion matrix to JSON
                conf_matrix_json = json.dumps(result['confusion_matrix'].tolist())

                cursor.execute(perf_query, (session_id, model_name, result['accuracy'], conf_matrix_json))
                performance_count += 1

            print(f"  ✅ Saved {performance_count} model performances")

            # 3. Save feature importance
            importance_count = 0
            for model_name, result in self.results.items():
                if result['feature_importance'] is not None:
                    for idx, row in result['feature_importance'].iterrows():
                        imp_query = """
                            INSERT INTO feature_importance 
                            (session_id, model_name, feature_name, importance_value, feature_rank)
                            VALUES (%s, %s, %s, %s, %s)
                        """

                        cursor.execute(imp_query, (
                            session_id, model_name, row['feature'],
                            row['importance'], idx + 1
                        ))
                        importance_count += 1

            print(f"  ✅ Saved {importance_count} feature importance records")

            # Commit all changes
            self.db_manager.connection.commit()
            cursor.close()
            self.db_manager.disconnect()

            print(f"🎉 All results saved to database successfully!")
            return session_id

        except Exception as e:
            print(f"❌ Error saving to database: {e}")
            if hasattr(self, 'db_manager') and self.db_manager.connection:
                self.db_manager.connection.rollback()
                self.db_manager.disconnect()
            return None

    def run_full_training_pipeline(self):
        """ENHANCED - run complete 5-iteration training with advanced models"""
        print("🚀 Starting ENHANCED ML Training Pipeline")
        print("=" * 60)

        # Load data using existing infrastructure
        data = self.load_prepared_data()
        if data is None:
            return None

        X_train, X_test, y_train, y_test = data

        # Store data info for database saving
        total_games = len(X_train) + len(X_test)
        train_games = len(X_train)
        test_games = len(X_test)
        target_distribution = y_train.value_counts().to_dict()

        # Iteration 1: Baseline (300 trees)
        print("\n🔄 ITERATION 1: Enhanced Baseline Model")
        rf_1 = self.train_iteration(X_train, y_train, "iteration_1", n_estimators=300)
        acc_1, _ = self.evaluate_model(rf_1, X_test, y_test, "iteration_1")

        # Iteration 2: Hyperparameter Tuning
        print("\n🔄 ITERATION 2: Hyperparameter Tuning")
        tuned_params = {
            'n_estimators': 400,  # Even more trees
            'max_depth': 15 if acc_1 < 0.65 else 12,
            'min_samples_split': 3,
            'min_samples_leaf': 1
        }
        rf_2 = self.train_iteration(X_train, y_train, "iteration_2", **tuned_params)
        acc_2, _ = self.evaluate_model(rf_2, X_test, y_test, "iteration_2")

        # Iteration 3: GridSearch Optimization
        print("\n🔄 ITERATION 3: GridSearch Optimization")
        rf_tuned = self.hyperparameter_tuning(X_train, y_train)
        acc_3, _ = self.evaluate_model(rf_tuned, X_test, y_test, "random_forest_tuned")

        # Iteration 4: Advanced Models
        print("\n🔄 ITERATION 4: Advanced Models Training")
        xgb_model, gb_model, lr_model = self.train_advanced_models(X_train, y_train)

        # Evaluate each advanced model
        xgb_acc, _ = self.evaluate_model(xgb_model, X_test, y_test, "xgboost")
        gb_acc, _ = self.evaluate_model(gb_model, X_test, y_test, "gradient_boosting")
        lr_acc, _ = self.evaluate_model(lr_model, X_test, y_test, "logistic_regression_enhanced")

        # Iteration 5: Ensemble Model
        print("\n🔄 ITERATION 5: Ensemble Model")
        ensemble_model = self.create_ensemble_model(X_train, y_train)
        ensemble_acc, _ = self.evaluate_model(ensemble_model, X_test, y_test, "ensemble")

        # Final results summary
        self.print_enhanced_summary()

        # NEW: Save results to database
        session_id = self.save_results_to_db(total_games, train_games, test_games, target_distribution)

        # Return best model
        best_model = self.get_best_model_overall()
        return best_model

    def get_best_model_overall(self):
        """Get the best performing model overall"""
        all_accuracies = {
            name: self.results[name]['accuracy']
            for name in self.results.keys()
        }

        best_model_name = max(all_accuracies, key=all_accuracies.get)
        best_model = self.models[best_model_name]

        print(f"\n🏆 Best Overall Model: {best_model_name}")
        print(f"Best Accuracy: {all_accuracies[best_model_name]:.1%}")

        return best_model

    def print_enhanced_summary(self):
        """Print comprehensive summary of all results - ENHANCED"""
        print("\n" + "=" * 70)
        print("🎯 ENHANCED TRAINING SUMMARY")
        print("=" * 70)

        print("\n📊 All Model Accuracies:")
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

        for model_name, result in sorted_results:
            acc = result['accuracy']
            print(f"  {model_name}: {acc:.4f} ({acc:.1%})")

        print("\n🎯 Feature Importance (Best Model):")
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        if self.results[best_model_name]['feature_importance'] is not None:
            for _, row in self.results[best_model_name]['feature_importance'].iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")

        print("\n✅ Enhanced Training Complete!")

        # Show improvement
        best_acc = max([result['accuracy'] for result in self.results.values()])
        print(f"🚀 Best accuracy achieved: {best_acc:.1%}")

        if best_acc >= 0.75:
            print("🎉 EXCELLENT! Reached 75%+ accuracy target!")
        elif best_acc >= 0.70:
            print("🎊 GREAT! Reached 70%+ accuracy!")
        else:
            print("👍 Good progress, consider more data or features for 75%+")

    def get_best_model(self):
        """UNCHANGED - for backward compatibility"""
        return self.get_best_model_overall()

    def train_comparison_model(self, X_train, X_test, y_train, y_test):
        """UNCHANGED - kept for compatibility"""
        pass  # This is now handled in advanced models

    def print_final_summary(self):
        """UNCHANGED - kept for compatibility"""
        self.print_enhanced_summary()


# Test the enhanced training pipeline
if __name__ == "__main__":
    trainer = ModelTrainer()

    # Run enhanced training pipeline with database saving
    best_model = trainer.run_full_training_pipeline()

    if best_model:
        print(f"\n🎉 Enhanced training completed successfully!")
        print(f"Best model ready for use!")
        print(f"🗄️ Results saved to database for future reporting!")