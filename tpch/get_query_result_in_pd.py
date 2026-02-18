import pandas as pd

customer = pd.read_csv("./csv_out/customer.csv")
orders = pd.read_csv("./csv_out/orders.csv")
partsupp = pd.read_csv("./csv_out/partsupp.csv")
lineitem = pd.read_csv("./csv_out/lineitem.csv")
nation = pd.read_csv("./csv_out/nation.csv")
part = pd.read_csv("./csv_out/part.csv")
region = pd.read_csv("./csv_out/region.csv")
supplier = pd.read_csv("./csv_out/supplier.csv")

# convert all column names to upper case
partsupp.columns = partsupp.columns.str.upper()
part.columns = part.columns.str.upper()
customer.columns = customer.columns.str.upper()
orders.columns = orders.columns.str.upper()
lineitem.columns = lineitem.columns.str.upper()
nation.columns = nation.columns.str.upper()
region.columns = region.columns.str.upper()
supplier.columns = supplier.columns.str.upper()

orders["O_ORDERDATE"] = pd.to_datetime(orders.O_ORDERDATE, format="%Y-%m-%d")

# lineitem
lineitem["L_SHIPDATE"] = pd.to_datetime(lineitem.L_SHIPDATE, format="%Y-%m-%d")
lineitem["L_RECEIPTDATE"] = pd.to_datetime(lineitem.L_RECEIPTDATE, format="%Y-%m-%d")
lineitem["L_COMMITDATE"] = pd.to_datetime(lineitem.L_COMMITDATE, format="%Y-%m-%d")

# Query like below.
# q10(lineitem, orders, customer, nation)
