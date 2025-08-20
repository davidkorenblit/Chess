# Chess Game Outcome Prediction 

A machine learning project that predicts chess game outcomes (White wins/Black wins/Draw) using player ratings and historical performance data.

## Project Overview

This project uses multiple machine learning models including **Random Forest**, **XGBoost**, **Gradient Boosting**, and **Ensemble methods** to predict chess game results with **66-68% accuracy** based on:
- Player rating differences
- Historical win rates
- Average player ratings

Built as a preparation project for an AI thesis, focusing on practical SQL skills, API integration, machine learning implementation, and automated reporting.

## Results

- **Best Model Accuracy**: 66-68% (varies by training session)
- **Dataset Size**: ~17,800 high-quality chess games
- **Players Analyzed**: 250+ active players (1000+ rating)
- **Feature Importance**: Rating difference (52-55%), Win rates (25-30%)
- **Models Trained**: Random Forest iterations, XGBoost, Gradient Boosting, Logistic Regression, Ensemble

## Architecture

```
Chess.com API → Data Collection → PostgreSQL → Feature Engineering → ML Pipeline → Database Storage → HTML Reports
```

### Key Components:
- **Data Collection**: Automated player discovery via snowball sampling
- **Database**: PostgreSQL with optimized schemas for chess data and ML results
- **Feature Engineering**: SQL-based statistical calculations  
- **ML Pipeline**: Multi-model training with hyperparameter tuning
- **Results Storage**: Training results persisted in database tables
- **Automated Reporting**: HTML report generation from stored results

## Project Structure

```
chess_project/
├── config/
│   ├── database.py              # Database connection management
│   └── create_ml_tables.py      # ML results table creation
├── data_collection/
│   ├── chess_api_client.py      # Chess.com API interface
│   ├── player_collector.py      # Player discovery system
│   └── game_collector.py        # Game data collection
├── database/
│   ├── db_manager.py            # Database operations
│   └── schema.sql               # Database schema
├── feature_engineering/
│   ├── feature_extractor.py     # Feature preparation
│   └── data_preprocessor.py     # Data preparation for ML
├── ml_pipeline/
│   └── model_trainer.py         # Multi-model training with DB storage
├── analysis/
│   └── model_reporter.py        # HTML report generation
├── reports/                     # Generated HTML reports
└── sql_scripts/
    └── create_tables.sql        # Database setup
```

## Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL
- Internet connection for Chess.com API

### Installation
```bash
git clone https://github.com/yourusername/chess-prediction.git
cd chess-prediction
pip install -r requirements.txt
```

### Database Setup
```sql
-- Create database and tables
CREATE DATABASE chess_db;
-- Run sql_scripts/create_tables.sql
-- Run config/create_ml_tables.py for ML results storage
```

### Run the Complete Pipeline
```python
# 1. Collect players (snowball method)
from data_collection.player_collector import PlayerCollector
collector = PlayerCollector()
players = collector.snowball_collect_players(target_count=250)

# 2. Collect games
from data_collection.game_collector import GameCollector
game_collector = GameCollector()
game_collector.collect_all_games(target_games=15000)

# 3. Train models and save results
from ml_pipeline.model_trainer import ModelTrainer
trainer = ModelTrainer()
best_model = trainer.run_full_training_pipeline()  # Automatically saves to DB

# 4. Generate HTML report
from analysis.model_reporter import generate_latest_report
report_path = generate_latest_report()  # Fast - reads from DB
```

## Model Performance

| Model | Typical Accuracy | Notes |
|-------|------------------|-------|
| Random Forest (Tuned) | 66-68% | Best performing model |
| XGBoost | 65-67% | Advanced gradient boosting |
| Gradient Boosting | 64-66% | Scikit-learn implementation |
| Ensemble (Voting) | 66-68% | Combines multiple models |
| Logistic Regression | 63-65% | Baseline comparison |

### Feature Importance:
1. **Rating Difference** (52-55%) - Primary predictor
2. **White Win Rate** (13-15%) - Historical white performance  
3. **Black Win Rate** (12-14%) - Historical black performance
4. **White Avg Rating** (10-12%) - Player strength indicator
5. **Black Avg Rating** (8-10%) - Player strength indicator

## Technical Highlights

- **Multi-Model Training**: 5-iteration pipeline with advanced algorithms
- **Hyperparameter Optimization**: GridSearchCV for optimal model parameters
- **Database-Driven Architecture**: Results stored in PostgreSQL for persistence
- **Separation of Concerns**: Training and reporting completely decoupled
- **Fast Report Generation**: HTML reports generated in seconds from stored data
- **Snowball Sampling**: Discovered 250+ players from 12 seed players
- **SQL Feature Engineering**: Complex queries for win rates and statistics
- **Rate Limited API**: Respectful Chess.com API integration
- **Clean Architecture**: Modular design with clear responsibilities

## Reporting System

The project includes an automated HTML reporting system:

### Database Tables for ML Results:
- **training_sessions**: Session metadata and best model info
- **model_performance**: Accuracy and confusion matrices per model
- **feature_importance**: Feature rankings and importance values

### Report Features:
- Real-time data from latest training session
- Interactive confusion matrices
- Feature importance visualizations
- Model comparison charts
- Performance recommendations
- Hebrew interface with professional styling

## Future Improvements

- **More Data**: Expand to 30,000+ games across multiple months
- **Additional Features**: Opening analysis, time control impact, player momentum
- **Advanced Models**: Neural networks, deep learning approaches
- **Real-time Prediction**: Live game outcome prediction API
- **Multi-Session Analysis**: Compare training sessions over time
- **A/B Testing**: Model performance comparison framework

## Technologies Used

- **Python**: Core programming language
- **PostgreSQL**: Data storage, feature engineering, and results persistence
- **scikit-learn**: Machine learning framework
- **XGBoost**: Advanced gradient boosting
- **pandas/numpy**: Data manipulation
- **Chess.com API**: Game data source
- **HTML/CSS**: Professional report generation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---

**Note**: This project is for educational purposes. Please respect Chess.com's API terms of service.
