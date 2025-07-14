# load the data
import pandas as pd
import sqlite3

def load_data(db_path="data/bmarket.db"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM bank_marketing;", conn)
    conn.close()
    return df
