#!/bin/bash
filename="$1"
ids="$2"

IFS=',' read -ra id_array <<< "$ids"
for id in "${id_array[@]}"; do
    grep "SLURM_ID $id:" "$filename"
done