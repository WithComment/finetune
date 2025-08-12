#!/bin/bash

# List of (dataset_name, system_prompt) tuples
declare -a configs=(
    "surgeryvid_test default Qwen2_5_7_surgeryvid_default/last"
    "surgeryvid_test surgeryvid Qwen2_5_7_surgeryvid_default/last"
    "surgeryvid_test video Qwen2_5_7_surgeryvid_video/last"
    "surgeryvid_test surgeryvid Qwen2_5_7_surgeryvid_surgeryvid/last"
)

# Iterate over the configurations and call swift.sh
for config in "${configs[@]}"; do
    set -- $config
    dataset_name=$1
    system_prompt=$2
    checkpoint_name=$3

    sbatch qwenvl/scripts/swift_gen.sh "$dataset_name" "$system_prompt" "$checkpoint_name"
done
