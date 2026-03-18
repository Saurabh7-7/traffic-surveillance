import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect("surveillance.db")
    cursor = conn.cursor()

    # table for all detected plates
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plate_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT,
            vehicle_type TEXT,
            track_id INTEGER,
            timestamp TEXT,
            flagged INTEGER DEFAULT 0
        )
    ''')

    # table for watchlist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT UNIQUE,
            reason TEXT,
            added_on TEXT
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized!")

def log_plate(plate_number, vehicle_type, track_id, flagged=False):
    conn = sqlite3.connect("surveillance.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO plate_logs (plate_number, vehicle_type, track_id, timestamp, flagged)
        VALUES (?, ?, ?, ?, ?)
    ''', (plate_number, vehicle_type, track_id,
          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          1 if flagged else 0))
    conn.commit()
    conn.close()

def is_flagged(plate_number):
    conn = sqlite3.connect("surveillance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT reason FROM watchlist WHERE plate_number = ?",
                   (plate_number,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def add_to_watchlist(plate_number, reason):
    conn = sqlite3.connect("surveillance.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO watchlist (plate_number, reason, added_on)
        VALUES (?, ?, ?)
    ''', (plate_number, reason,
          datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    print(f"Added {plate_number} to watchlist!")

def get_all_logs():
    conn = sqlite3.connect("surveillance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM plate_logs ORDER BY timestamp DESC")
    logs = cursor.fetchall()
    conn.close()
    return logs

if __name__ == "__main__":
    init_db()
    # add some test plates to watchlist
    add_to_watchlist("MH12AB1234", "Stolen vehicle")
    add_to_watchlist("DL3CAF1234", "Wanted suspect")
    print("Watchlist populated!")
