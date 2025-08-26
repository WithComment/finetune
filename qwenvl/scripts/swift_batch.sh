#!/bin/bash

# List of (job_config, model_name, dataset_name, system_prompt) tuples
declare -a configs=(
)

# Iterate over the configurations and call swift.sh
for config in "${configs[@]}"; do
    set -- $config
    job_config=$1
    model_name=$2
    dataset_name=$3
    system_prompt=$4

    bash qwenvl/scripts/swift.sh --config "$job_config" --model_name "$model_name" --dataset_name "$dataset_name" --system_prompt "$system_prompt"
done
