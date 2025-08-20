import requests
import time
import json
from datetime import datetime


class ChessAPIClient:
    def __init__(self):
        self.base_url = "https://api.chess.com/pub"
        self.rate_limit_delay = 1  # 1 second between requests
        self.headers = {
            'User-Agent': 'Chess-Project/1.0 (Educational purposes)'
        }

    def _make_request(self, url):
        """Make HTTP request with rate limiting"""
        try:
            time.sleep(self.rate_limit_delay)  # Respect rate limits
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Error {response.status_code}: {url}")
                return None

        except Exception as e:
            print(f"Request failed: {e}")
            return None

    def get_player_profile(self, username):
        """Get player profile information"""
        url = f"{self.base_url}/player/{username}"
        return self._make_request(url)

    def get_player_stats(self, username):
        """Get player rating information"""
        url = f"{self.base_url}/player/{username}/stats"
        return self._make_request(url)

    def get_player_games_by_month(self, username, year, month):
        """Get player games for specific month"""
        url = f"{self.base_url}/player/{username}/games/{year}/{month:02d}"
        return self._make_request(url)

    def test_connection(self):
        """Test API connection with a known player"""
        print("Testing Chess.com API connection...")
        profile = self.get_player_profile("hikaru")

        if profile:
            print(f"✅ API Connection successful!")
            print(f"Player: {profile.get('username', 'Unknown')}")
            print(f"Title: {profile.get('title', 'No title')}")
            return True
        else:
            print("❌ API Connection failed")
            return False


# הוסף בסוף הקובץ
def test_game_api_response():
    """Test what Chess.com API actually returns for games"""
    from data_collection.chess_api_client import ChessAPIClient

    api = ChessAPIClient()

    # Test with a known active player
    print("Testing game API response format...")
    games_data = api.get_player_games_by_month("hikaru", 2024, 8)

    if games_data and 'games' in games_data:
        game = games_data['games'][0]  # First game
        print("\nSample game structure:")
        print(f"Keys in game: {list(game.keys())}")
        print(f"White data: {game.get('white', {})}")
        print(f"Black data: {game.get('black', {})}")
        print(f"Winner field: {game.get('winner', 'NOT_FOUND')}")
        print(f"Result field: {game.get('result', 'NOT_FOUND')}")
        print(f"Full game sample: {game}")
    else:
        print("No games found!")


if __name__ == "__main__":
    test_game_api_response()

# Test the API client
if __name__ == "__main__":
    api = ChessAPIClient()
    api.test_connection()