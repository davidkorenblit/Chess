from config.database import get_db_connection


def create_ml_tables():
    """Create tables for storing ML training results"""

    # SQL commands to create tables
    tables_sql = [
        """
        CREATE TABLE IF NOT EXISTS training_sessions (
            session_id SERIAL PRIMARY KEY,
            training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_games INTEGER NOT NULL,
            total_train_games INTEGER NOT NULL,
            total_test_games INTEGER NOT NULL,
            best_model_name VARCHAR(100) NOT NULL,
            best_accuracy DECIMAL(5,4) NOT NULL,
            target_distribution_white INTEGER,
            target_distribution_black INTEGER,
            target_distribution_draw INTEGER,
            feature_names TEXT
        );
        """,

        """
        CREATE TABLE IF NOT EXISTS model_performance (
            performance_id SERIAL PRIMARY KEY,
            session_id INTEGER REFERENCES training_sessions(session_id),
            model_name VARCHAR(100) NOT NULL,
            accuracy DECIMAL(5,4) NOT NULL,
            confusion_matrix TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,

        """
        CREATE TABLE IF NOT EXISTS feature_importance (
            importance_id SERIAL PRIMARY KEY,
            session_id INTEGER REFERENCES training_sessions(session_id),
            model_name VARCHAR(100) NOT NULL,
            feature_name VARCHAR(100) NOT NULL,
            importance_value DECIMAL(6,5) NOT NULL,
            feature_rank INTEGER
        );
        """
    ]

    try:
        # Connect to database
        conn = get_db_connection()
        if not conn:
            print("❌ Failed to connect to database!")
            return False

        cursor = conn.cursor()

        print("🔄 Creating ML results tables...")

        # Execute each table creation
        for i, sql in enumerate(tables_sql, 1):
            table_name = ["training_sessions", "model_performance", "feature_importance"][i - 1]
            print(f"  Creating table {i}/3: {table_name}")
            cursor.execute(sql)

        # Commit changes
        conn.commit()
        print("✅ All tables created successfully!")

        # Verify tables exist
        print("\n🔍 Verifying tables exist:")
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('training_sessions', 'model_performance', 'feature_importance')
            ORDER BY table_name;
        """)

        tables = cursor.fetchall()
        for table in tables:
            print(f"  ✅ {table[0]}")

        # Close connection
        cursor.close()
        conn.close()

        print(f"\n🎉 Successfully created {len(tables)} tables!")
        return True

    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return False


def check_existing_tables():
    """Check what tables already exist"""
    try:
        conn = get_db_connection()
        if not conn:
            print("❌ Failed to connect to database!")
            return

        cursor = conn.cursor()

        print("📋 Checking existing tables:")
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)

        tables = cursor.fetchall()
        for table in tables:
            print(f"  📊 {table[0]}")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"❌ Error checking tables: {e}")


if __name__ == "__main__":
    print("🚀 Starting ML Tables Creation")
    print("=" * 40)

    # First check existing tables
    check_existing_tables()

    print("\n" + "=" * 40)

    # Create new tables
    success = create_ml_tables()

    if success:
        print("\n✅ Ready for ML results storage!")
    else:
        print("\n❌ Failed to create tables")