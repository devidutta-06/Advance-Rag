import sqlite3
conn = sqlite3.connect("storage/rag.db")

for row in conn.execute("SELECT chunk_id, page_number, text FROM sparse_index WHERE text LIKE '%PaddleOCR%'"):
    print(row[0], "| page", row[1])
    print(row[2])
    print("---")