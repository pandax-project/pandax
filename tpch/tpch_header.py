import shutil
from pathlib import Path

s1 = Path("/home/dias-benchmarks/tpch/data/factor_1")
s100 = Path("/home/tpch-dbgen")
out = Path("/home/dias-benchmarks/tpch/data/factor_6")
out.mkdir(exist_ok=True)

tables = [
    "region",
    "nation",
    "supplier",
    "customer",
    "part",
    "partsupp",
    "orders",
    "lineitem",
]

for tbl in tables:
    print("processing", tbl)
    hdr = s1.joinpath(f"{tbl}.tbl").open().readline()
    with (
        s100.joinpath(f"{tbl}.tbl").open("r") as fin,
        out.joinpath(f"{tbl}.tbl").open("w") as fout,
    ):
        fout.write(hdr)
        shutil.copyfileobj(fin, fout)
