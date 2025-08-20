import psycopg2
from config.database import get_db_connection


class DatabaseManager:
    def __init__(self):
        self.connection = None

    def connect(self):
        """Connect to database"""
        self.connection = get_db_connection()
        return self.connection is not None

    def disconnect(self):
        """Disconnect from database"""
        if self.connection:
            self.connection.close()
            self.connection = None

    def add_player(self, username, current_rating, peak_rating=None, games_played=0):
        """Add new player"""
        try:
            cursor = self.connection.cursor()
            query = """
                INSERT INTO players (username, current_rating, peak_rating, games_played)
                VALUES (%s, %s, %s, %s) RETURNING player_id
            """
            cursor.execute(query, (username, current_rating, peak_rating, games_played))
            player_id = cursor.fetchone()[0]
            self.connection.commit()
            cursor.close()
            print(f"Player {username} added successfully! ID: {player_id}")
            return player_id
        except Exception as e:
            print(f"Error adding player: {e}")
            return None

    def add_game(self, white_player_id, black_player_id, white_rating, black_rating,
                 result, game_date, moves_count=None):
        """Add new game"""
        try:
            cursor = self.connection.cursor()
            query = """
                INSERT INTO games (white_player_id, black_player_id, white_rating, 
                                 black_rating, result, game_date, moves_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING game_id
            """
            cursor.execute(query, (white_player_id, black_player_id, white_rating,
                                   black_rating, result, game_date, moves_count))
            game_id = cursor.fetchone()[0]
            self.connection.commit()
            cursor.close()
            print(f"Game added successfully! ID: {game_id}")
            return game_id
        except Exception as e:
            print(f"Error adding game: {e}")
            return None

    def get_all_players(self):
        """Get all players"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM players ORDER BY current_rating DESC")
            players = cursor.fetchall()
            cursor.close()
            return players
        except Exception as e:
            print(f"Error fetching players: {e}")
            return []

    def get_all_games(self):
        """Get all games"""
        try:
            cursor = self.connection.cursor()
            query = """
                SELECT g.*, p1.username as white_player, p2.username as black_player
                FROM games g
                JOIN players p1 ON g.white_player_id = p1.player_id
                JOIN players p2 ON g.black_player_id = p2.player_id
                ORDER BY g.game_date DESC
            """
            cursor.execute(query)
            games = cursor.fetchall()
            cursor.close()
            return games
        except Exception as e:
            print(f"Error fetching games: {e}")
            return []