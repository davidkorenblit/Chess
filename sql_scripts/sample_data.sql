INSERT INTO players (username, current_rating, peak_rating, games_played) VALUES
('magnus_carlsen', 2830, 2882, 1500),
('hikaru_nakamura', 2736, 2816, 2200),
('ding_liren', 2806, 2816, 890);

INSERT INTO games (white_player_id, black_player_id, white_rating, black_rating, result, game_date, moves_count) VALUES
(1, 2, 2830, 2736, 'white', '2024-01-15', 45),
(2, 3, 2736, 2806, 'black', '2024-01-16', 38),
(1, 3, 2830, 2806, 'draw', '2024-01-17', 67);