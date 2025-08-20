# Chess Game Outcome Prediction 

A machine learning project that predicts chess game outcomes (White wins/Black wins/Draw) using player ratings and historical performance data.

## Project Overview

This project uses a **Random Forest** model to predict chess game results with **68.7% accuracy** based on:
- Player rating differences
- Historical win rates
- Average player ratings

Built as a preparation project for an AI thesis, focusing on practical SQL skills, API integration, and machine learning implementation.

##  Results

- **Final Model Accuracy**: 68.7%
- **Dataset Size**: 4,680 high-quality chess games
- **Players Analyzed**: 251 active players (1000+ rating)
- **Feature Importance**: Rating difference (51.6%), Win rates (27.1%)

##  Architecture

```
Chess.com API → Data Collection → PostgreSQL → Feature Engineering → Random Forest
```

### Key Components:
- **Data Collection**: Automated player discovery via snowball sampling
- **Database**: PostgreSQL with optimized schemas for chess data
- **Feature Engineering**: SQL-based statistical calculations  
- **ML Pipeline**: Iterative Random Forest training with hyperparameter tuning

##  Project Structure

```
chess_project/
├── config/
│   └── database.py              # Database connection management
├── data_collection/
│   ├── chess_api_client.py      # Chess.com API interface
│   ├── player_collector.py      # Player discovery system
│   └── game_collector.py        # Game data collection
├── database/
│   ├── db_manager.py            # Database operations
│   └── schema.sql               # Database schema
├── feature_engineering/
│   └── feature_extractor.py     # Feature preparation
├── ml_pipeline/
│   ├── data_preprocessor.py     # Data preparation for ML
│   └── model_trainer.py         # Random Forest training
└── sql_scripts/
    └── create_tables.sql        # Database setup
```

##  Quick Start

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
```

### Run the Pipeline
```python
# Collect players (snowball method)
from data_collection.player_collector import PlayerCollector
collector = PlayerCollector()
players = collector.snowball_collect_players(target_count=250)

# Collect games
from data_collection.game_collector import GameCollector
game_collector = GameCollector()
game_collector.collect_all_games(target_games=5000)

# Train model
from ml_pipeline.model_trainer import ModelTrainer
trainer = ModelTrainer()
best_model = trainer.run_full_training_pipeline()
```

##  Model Performance

| Model | Accuracy | Notes |
|-------|----------|-------|
| Random Forest (Final) | 68.7% | Best performing model |
| Random Forest (Baseline) | 65.9% | Initial model |
| Logistic Regression | 64.4% | Comparison baseline |

### Feature Importance:
1. **Rating Difference** (51.6%) - Primary predictor
2. **White Win Rate** (14.1%) - Historical white performance  
3. **Black Win Rate** (13.0%) - Historical black performance
4. **Average Ratings** (21.4%) - Combined player strength

##  Technical Highlights

- **Snowball Sampling**: Discovered 251 players from 12 seed players
- **SQL Feature Engineering**: Complex queries for win rates and statistics
- **Iterative Training**: 3-phase model improvement process
- **Rate Limited API**: Respectful Chess.com API integration
- **Clean Architecture**: Separation of concerns across modules

##  Future Improvements

- **More Data**: Expand to 15,000+ games across multiple months
- **Additional Features**: Opening analysis, time control impact
- **Advanced Models**: Ensemble methods, neural networks
- **Real-time Prediction**: Live game outcome prediction

##  Technologies Used

- **Python**: Core programming language
- **PostgreSQL**: Data storage and feature engineering
- **scikit-learn**: Machine learning framework
- **pandas/numpy**: Data manipulation
- **Chess.com API**: Game data source

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---

**Note**: This project is for educational purposes. Please respect Chess.com's API terms of service.
