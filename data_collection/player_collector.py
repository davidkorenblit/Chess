from chess_api_client import ChessAPIClient
from datetime import datetime
from database.db_manager import DatabaseManager


class PlayerCollector:
    def __init__(self):
        self.api_client = ChessAPIClient()
        # All 12 verified players from our previous test
        self.seed_players = [
            "magnuscarlsen", "hikaru", "lachesisq", "levonaronian",
            "fabianocaruana", "gmwso", "rpragchess", "anishgiri",
            "grischuk", "arjunerigaisi", "mvl", "teimour"
        ]
        self.discovered_players = set()  # Avoid duplicates
        self.valid_players = []

        # Database connection
        self.db_manager = DatabaseManager()
        self.db_manager.connect()

    def save_player_to_db(self, username, rating):
        """Save discovered player to database"""
        try:
            player_id = self.db_manager.add_player(
                username=username,
                current_rating=rating,
                peak_rating=rating,  # Same as current for now
                games_played=0  # Will update later
            )
            return player_id
        except Exception as e:
            print(f"Error saving {username}: {e}")
            return None

    def extract_players_from_games(self, username, year=2024, month=8):
        """Extract opponent usernames from player's games"""
        games_data = self.api_client.get_player_games_by_month(username, year, month)

        if not games_data or 'games' not in games_data:
            return []

        opponents = []
        for game in games_data['games']:
            # Extract white and black player usernames
            white = game.get('white', {}).get('username', '').lower()
            black = game.get('black', {}).get('username', '').lower()

            # Add opponents (not the original player)
            if white and white != username.lower():
                opponents.append(white)
            if black and black != username.lower():
                opponents.append(black)

        return list(set(opponents))  # Remove duplicates

    def snowball_collect_players(self, target_count=3000, min_rating=1000):
        """Collect players using snowball method"""
        to_process = self.seed_players.copy()
        processed = set()

        print(f"Starting snowball collection targeting {target_count} players...")
        print(f"Starting with {len(self.seed_players)} verified players")

        while to_process and len(self.discovered_players) < target_count:
            current_player = to_process.pop(0)

            if current_player in processed:
                continue

            processed.add(current_player)
            print(f"\nProcessing: {current_player} (Found: {len(self.discovered_players)})")

            # Check if current player is valid
            stats = self.api_client.get_player_stats(current_player)
            if stats and 'chess_rapid' in stats:
                rating = stats['chess_rapid'].get('last', {}).get('rating', 0)
                if rating >= min_rating:
                    self.discovered_players.add(current_player)
                    self.save_player_to_db(current_player, rating)  # Save to database
                    print(f"✅ Added {current_player}: {rating}")
                else:
                    print(f"❌ {current_player}: Rating too low ({rating})")
                    continue
            else:
                print(f"❌ {current_player}: No stats available")
                continue

            # Extract opponents from this player's games
            opponents = self.extract_players_from_games(current_player)
            print(f"Found {len(opponents)} opponents")

            # Add new opponents to processing queue
            new_opponents = [op for op in opponents if op not in processed and op not in to_process]
            to_process.extend(new_opponents[:20])  # Limit to avoid explosion

            print(f"Added {len(new_opponents[:20])} new players to queue")

        print(f"\nSnowball collection complete!")
        print(f"Total players discovered: {len(self.discovered_players)}")
        return list(self.discovered_players)

    def get_valid_players(self, min_rating=1200):
        """Original method - for testing individual players"""
        fide_to_chesscom = [
            ("Magnus Carlsen", "magnuscarlsen"),
            ("Hikaru Nakamura", "hikaru"),
            ("Fabiano Caruana", "fabianocaruana"),
            ("Ian Nepomniachtchi", "lachesisq"),
            ("Levon Aronian", "levonaronian"),
            ("Wesley So", "gmwso"),
            ("Praggnanandhaa", "rpragchess"),
            ("Anish Giri", "anishgiri"),
            ("Alexander Grischuk", "grischuk"),
            ("Arjun Erigaisi", "arjunerigaisi"),
            ("Maxime Vachier-Lagrave", "mvl"),
            ("Teimour Radjabov", "teimour")
        ]

        valid_players = []

        for fide_name, username in fide_to_chesscom:
            print(f"Checking {fide_name} ({username})...")

            stats = self.api_client.get_player_stats(username)
            if not stats:
                continue

            rapid_rating = None
            if 'chess_rapid' in stats:
                rapid_rating = stats['chess_rapid'].get('last', {}).get('rating', 0)

            if rapid_rating and rapid_rating >= min_rating:
                valid_players.append({
                    'fide_name': fide_name,
                    'username': username,
                    'rating': rapid_rating
                })
                print(f"✅ {fide_name}: {rapid_rating}")
            else:
                print(f"❌ {username}: Rating too low or unavailable")

        return valid_players

    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'db_manager'):
            self.db_manager.disconnect()


# Test the snowball collection
if __name__ == "__main__":
    collector = PlayerCollector()

    # Run snowball collection
    all_players = collector.snowball_collect_players(target_count=200)

    print(f"\nSnowball collection found {len(all_players)} players")
    print("Sample players:", all_players[:10])