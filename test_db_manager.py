from database.db_manager import DatabaseManager
from datetime import date


def test_database_manager():
    # יצירת מנהל מסד נתונים
    db = DatabaseManager()

    # התחברות
    if db.connect():
        print("Connected to database successfully!")

        # הוספת שחקן חדש
        player_id = db.add_player("test_player_python", 1400, 1500, 25)

        # הוספת משחק
        if player_id:
            game_id = db.add_game(
                white_player_id=1,  # magnus_carlsen
                black_player_id=player_id,  # השחקן החדש
                white_rating=2830,
                black_rating=1400,
                result='white',
                game_date=date.today(),
                moves_count=32
            )

        # הצגת כל השחקנים
        print("\nAll players:")
        players = db.get_all_players()
        for player in players:
            print(player)

        # הצגת כל המשחקים
        print("\nAll games:")
        games = db.get_all_games()
        for game in games:
            print(game)

        # ניתוק
        db.disconnect()
        print("\nDisconnected from database")
    else:
        print("Failed to connect to database")


if __name__ == "__main__":
    test_database_manager()