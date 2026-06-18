import sqlite3

# 1. Connect to the database
conn = sqlite3.connect('drift.db')
cursor = conn.cursor()

# 2. Add the missing column (we use try/except in case you run this twice)
try:
    cursor.execute("ALTER TABLE projects ADD COLUMN owner_email TEXT")
    print("✅ Successfully added 'owner_email' column to the database.")
except sqlite3.OperationalError:
    print("⚠️ Column 'owner_email' already exists.")

# 3. Assign existing projects to your email so you don't lose them
# Replace this with the EXACT Google email you used to log in!
my_google_email = "abc@gmail.com"  

cursor.execute("UPDATE projects SET owner_email = ?", (my_google_email.lower(),))
conn.commit()
print(f"✅ Assigned all current projects to: {my_google_email}")

conn.close()