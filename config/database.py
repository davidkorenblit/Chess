import psycopg2

def get_db_connection():
    try:
        connection = psycopg2.connect(
            host="localhost",
            port="5432",
            database="chess_db",
            user="postgres",
            password="213871734"
        )
        return connection
    except Exception as e:
        print(f"שגיאה בחיבור למסד נתונים: {e}")
        return None

def test_connection():
    conn = get_db_connection()
    if conn:
        print("חיבור למסד נתונים הצליח!")
        conn.close()
        return True
    else:
        print("חיבור למסד נתונים נכשל!")
        return False