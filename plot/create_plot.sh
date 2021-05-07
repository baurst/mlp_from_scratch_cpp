#!/bin/bash

set -euo pipefail

log_file="/tmp/mlp_log.txt"
acc_file="/tmp/acc_log.csv"

awk '/VAL Acc/ {print $2 ";" $7 }' "$log_file" > "$acc_file"
awk '/Test Acc/ {print $2 ";" $6 }' "$log_file" >> "$acc_file"

python3 plot.py --ours_acc_file "$acc_file" --tf_log_file $1
