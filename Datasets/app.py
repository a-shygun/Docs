import pandas as pd
import mysql.connector

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='royalblue',
    database='hospital'
)

df = pd.read_sql("SELECT * FROM hospitals", conn)
df.to_csv("/Users/dev/DSML/Databases/hospital.csv", index=False)
conn.close()
