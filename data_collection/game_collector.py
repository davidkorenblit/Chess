from chess_api_client import ChessAPIClient
from database.db_manager import DatabaseManager
from datetime import datetime, date


class GameCollector:
    def __init__(self):
        self.api_client = ChessAPIClient()
        self.db_manager = DatabaseManager()
        self.db_manager.connect()

        # הגדרת חודשים לאיסוף
        self.target_months = [
            (2024, 6),  # יוני 2024
            (2024, 7),  # יולי 2024
            (2024, 8),  # אוגוסט 2024
            (2024, 9),  # ספטמבר 2024
            (2024, 10)  # אוקטובר 2024
        ]

    def get_players_from_db(self):
        """Get all players from database"""
        try:
            players = self.db_manager.get_all_players()
            print(f"Found {len(players)} players in database")
            return players
        except Exception as e:
            print(f"Error getting players from DB: {e}")
            return []

    def collect_games_for_player_single_month(self, player_username, year, month):
        """Collect games for a specific player for ONE month"""
        games_data = self.api_client.get_player_games_by_month(player_username, year, month)

        if not games_data or 'games' not in games_data:
            return []

        collected_games = []
        for game in games_data['games']:
            # Extract game info
            white_data = game.get('white', {})
            black_data = game.get('black', {})

            white_username = white_data.get('username', '').lower()
            black_username = black_data.get('username', '').lower()
            white_rating = white_data.get('rating', 0)
            black_rating = black_data.get('rating', 0)

            # Get result - FIXED LOGIC
            white_result = white_data.get('result', '')
            black_result = black_data.get('result', '')

            if white_result == 'win':
                result = 'white'
            elif black_result == 'win':
                result = 'black'
            else:
                result = 'draw'

            # Get game date
            end_time = game.get('end_time', 0)
            game_date = datetime.fromtimestamp(end_time).date() if end_time else date.today()

            # Get moves count (optional)
            moves_count = game.get('pgn', '').count('.') if 'pgn' in game else None

            collected_games.append({
                'white_username': white_username,
                'black_username': black_username,
                'white_rating': white_rating,
                'black_rating': black_rating,
                'result': result,
                'game_date': game_date,
                'moves_count': moves_count,
                'month': f"{year}-{month:02d}"  # Track which month
            })

        return collected_games

    def collect_games_multiple_months(self, player_username, months_list=None):
        """Collect games for a player across multiple months"""
        if months_list is None:
            months_list = self.target_months

        all_games = []

        print(f"  Collecting from {len(months_list)} months for {player_username}...")

        for year, month in months_list:
            print(f"    {year}-{month:02d}: ", end="")
            monthly_games = self.collect_games_for_player_single_month(player_username, year, month)
            print(f"{len(monthly_games)} games")
            all_games.extend(monthly_games)

        print(f"  Total: {len(all_games)} games for {player_username}")
        return all_games

    def save_game_to_db(self, game_data, player_mapping):
        """Save game to database if both players exist in our DB"""
        white_username = game_data['white_username']
        black_username = game_data['black_username']

        # Check if both players exist in our database
        white_player_id = player_mapping.get(white_username)
        black_player_id = player_mapping.get(black_username)

        if white_player_id and black_player_id:
            # Save game to database
            game_id = self.db_manager.add_game(
                white_player_id=white_player_id,
                black_player_id=black_player_id,
                white_rating=game_data['white_rating'],
                black_rating=game_data['black_rating'],
                result=game_data['result'],
                game_date=game_data['game_date'],
                moves_count=game_data['moves_count']
            )
            return game_id
        return None

    def collect_all_games_extended(self, target_games=15000):
        """Enhanced method to collect games from multiple months"""
        print("🚀 Starting EXTENDED Game Collection (Multiple Months)")
        print("=" * 60)

        # Get players from database
        players = self.get_players_from_db()
        if not players:
            print("No players found in database!")
            return

        # Create username to player_id mapping
        player_mapping = {}
        for player in players:
            player_id, username = player[0], player[1]
            player_mapping[username.lower()] = player_id

        print(f"Target: {target_games} games from {len(players)} players")
        print(f"Months: {', '.join([f'{y}-{m:02d}' for y, m in self.target_months])}")
        print()

        total_games_saved = 0
        monthly_stats = {f"{y}-{m:02d}": 0 for y, m in self.target_months}

        for i, player in enumerate(players, 1):
            player_id, username = player[0], player[1]
            print(f"[{i}/{len(players)}] Processing {username}...")

            # Collect games across all months for this player
            all_games = self.collect_games_multiple_months(username)

            if not all_games:
                print(f"  No games found for {username}")
                continue

            # Save games to database
            saved_count = 0
            for game_data in all_games:
                if self.save_game_to_db(game_data, player_mapping):
                    saved_count += 1
                    monthly_stats[game_data['month']] += 1

            total_games_saved += saved_count
            print(f"  ✅ Saved {saved_count}/{len(all_games)} games for {username}")
            print(f"  📊 Total games so far: {total_games_saved}")

            # Progress update every 10 players
            if i % 10 == 0:
                print(f"\\n📈 Progress Update - {i}/{len(players)} players processed")
                print(f"📊 Games collected: {total_games_saved}")
                print("📅 Monthly breakdown:")
                for month, count in monthly_stats.items():
                    print(f"    {month}: {count} games")
                print()

            # Stop if we reached our target
            if total_games_saved >= target_games:
                print(f"🎯 Reached target of {target_games} games!")
                break

        print("\\n" + "=" * 60)
        print("🎉 EXTENDED COLLECTION COMPLETE!")
        print("=" * 60)
        print(f"📊 Final Statistics:")
        print(f"  Total games collected: {total_games_saved}")
        print(f"  Players processed: {min(i, len(players))}")
        print(f"  Average games per player: {total_games_saved / min(i, len(players)):.1f}")

        print(f"\\n📅 Monthly Breakdown:")
        for month, count in monthly_stats.items():
            percentage = (count / total_games_saved * 100) if total_games_saved > 0 else 0
            print(f"  {month}: {count:,} games ({percentage:.1f}%)")

        return total_games_saved

    def collect_all_games(self, target_games=5000):
        """Original method - for backward compatibility"""
        return self.collect_all_games_extended(target_games)

    def get_collection_statistics(self):
        """Get statistics about collected games"""
        try:
            # Basic counts
            total_games = len(self.db_manager.get_all_games())
            total_players = len(self.db_manager.get_all_players())

            print(f"\\n📊 Current Database Statistics:")
            print(f"  Total games: {total_games:,}")
            print(f"  Total players: {total_players:,}")

            return {
                'total_games': total_games,
                'total_players': total_players
            }
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return None

    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'db_manager'):
            self.db_manager.disconnect()


# Test the enhanced game collector
if __name__ == "__main__":
    collector = GameCollector()

    # Show current stats
    collector.get_collection_statistics()

    # Run extended collection
    total_collected = collector.collect_all_games_extended(target_games=15000)  # Start with 1000 for testing

    if total_collected:
        print(f"\\n🎉 Collection completed successfully!")
        print(f"📈 Games collected: {total_collected}")

        # Show final stats
        collector.get_collection_statistics()