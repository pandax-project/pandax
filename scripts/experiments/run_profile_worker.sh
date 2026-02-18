#!/bin/bash

QUEUE_FILE="scripts/profile_queue.txt"

while true; do
  if [ ! -s "$QUEUE_FILE" ]; then
    echo "Queue is empty. Sleeping..."
    sleep 10
    continue
  fi

  # Read first line
  JOB=$(head -n 1 "$QUEUE_FILE")
  FACTOR=$(echo "$JOB" | cut -d'|' -f1)
  NOTEBOOK=$(echo "$JOB" | cut -d'|' -f2)

  echo "Running: $NOTEBOOK"

# Run the Python script
  FACTOR=$FACTOR python scripts/plot_curve_all_cells.py "$NOTEBOOK"

  tail -n +2 "$QUEUE_FILE" > "${QUEUE_FILE}.tmp" && mv "${QUEUE_FILE}.tmp" "$QUEUE_FILE"
done
