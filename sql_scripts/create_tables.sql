CREATE TABLE players (
    player_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    current_rating INTEGER,
    peak_rating INTEGER,
    games_played INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE games (
    game_id SERIAL PRIMARY KEY,
    white_player_id INTEGER REFERENCES players(player_id),
    black_player_id INTEGER REFERENCES players(player_id),
    white_rating INTEGER NOT NULL,
    black_rating INTEGER NOT NULL,
    result VARCHAR(10) CHECK (result IN ('white', 'black', 'draw')),
    time_control VARCHAR(20),
    opening_name VARCHAR(100),
    game_date DATE NOT NULL,
    moves_count INTEGER,
    game_duration_minutes INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE player_stats (
    stat_id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(player_id),
    wins_as_white INTEGER DEFAULT 0,
    wins_as_black INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    avg_game_length DECIMAL(5,2),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);