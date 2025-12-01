import sqlite3

# Create / connect to database file
conn = sqlite3.connect("prediction_logs.db")
c = conn.cursor()

# Create table for logs
c.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    disease TEXT,
    inputs TEXT,
    prediction TEXT,
    confidence REAL
)
""")

conn.commit()
conn.close()

print("Database created successfully!")
