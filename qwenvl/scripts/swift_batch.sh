#!/bin/bash

# List of (dataset_name, system_prompt) tuples
declare -a configs=(
    # "surgeryvid default"
    # "surgeryvid video"
    # "surgeryvid surgeryvid"
    "fashion_final#1000 default"
    "fashion_final#1000 video"
)

# Iterate over the configurations and call swift.sh
for config in "${configs[@]}"; do
    set -- $config
    dataset_name=$1
    system_prompt=$2

    sbatch qwenvl/scripts/swift.sh "$dataset_name" "$system_prompt"
done
