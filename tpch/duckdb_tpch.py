import duckdb

con = duckdb.connect()

con.execute("INSTALL tpch;")
con.execute("LOAD tpch;")
con.execute("DROP TABLE IF EXISTS customer;")
con.execute("DROP TABLE IF EXISTS lineitem;")
con.execute("DROP TABLE IF EXISTS nation;")
con.execute("DROP TABLE IF EXISTS orders;")
con.execute("DROP TABLE IF EXISTS part;")
con.execute("DROP TABLE IF EXISTS partsupp;")
con.execute("DROP TABLE IF EXISTS region;")
con.execute("DROP TABLE IF EXISTS supplier;")
con.execute("CALL dbgen(sf=1);")
# df = con.execute("PRAGMA tpch(4);").fetchdf()
df = con.execute("SELECT * FROM tpch_queries();").fetchdf()
i = 10
query = df[df["query_nr"] == i]["query"].values[0]
print(query)

exit()
# df = con.execute("SELECT * FROM Q1();").fetchdf()

# with open(f"tpch/queries/tpch_query_{i}.sql", "w") as f:
#     f.write(query)

# df.to_csv("tpch_queries.csv", index=False)


# df = con.execute("SELECT * FROM tpch_answers() WHERE scale_factor = 1 ORDER BY query_nr;").fetchdf()
# for i in range(1, 23):
#     answer = df[df['query_nr'] == i].answer.values[0]
#     # convert answer to pandas dataframe. different rows are separated by \n
#     answer = answer.split("\n")
#     answer = [row.split("|") for row in answer]
#     answer = pd.DataFrame(answer[1:], columns=answer[0]).dropna()
#     answer.to_csv(f"tpch/answers/tpch_answer_{i}.csv", index=False)

# # df.to_csv("tpch_answers.csv", index=False)
