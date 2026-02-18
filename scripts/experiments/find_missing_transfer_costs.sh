# Find which benchmarks are missing transfer_costs.csv.
find . -type f -name 'bench.ipynb' \
  -print0 \
| xargs -0 -n1 dirname \
| sort -u \
| while read dir; do
    if [ ! -f "$dir/transfer_costs.csv" ]; then
      echo "$dir"
    fi
  done
