#!/bin/bash

NOTEBOOK_PATH=$1
if [ -z "$NOTEBOOK_PATH" ]; then
  echo "Usage: ./enqueue_profile.sh <notebook_path>"
  exit 1
fi

for FACTOR in $(seq 100 10 200); do
  echo "$FACTOR|$NOTEBOOK_PATH" >> profile_queue.txt
done

echo "Enqueued profiling for $NOTEBOOK_PATH"
