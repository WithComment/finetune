#!/bin/bash

# List of (dataset_name, system_prompt) tuples
declare -a configs=(
    "surgeryvid_test#1000 default Qwen2_5_7_surgeryvid_default/last"
    "surgeryvid_test#1000 default Qwen2_5_7_surgeryvid_v1_0/last"
    "surgeryvid_test#1000 v1_0 Qwen2_5_7_surgeryvid_v1_0/last"
    "surgeryvid_test#1000 v1_4 Qwen2_5_7_surgeryvid_v1_4/last"
    # "surgeryvid_test#500 default Qwen2_5_7_surgeryvid_simple/last"
    # "surgeryvid_test#500 default Qwen2_5_7_surgeryvid_v1_2/last"
    # "surgeryvid_test#500 default Qwen2_5_7_surgeryvid_v2_2/last"
    # "surgeryvid_test#500 default Qwen2_5_7_surgeryvid_v1_3/last"
    # "surgeryvid_test#500 default Qwen2_5_7_surgeryvid_v2_3/last"
    # "surgeryvid_test#500 default Qwen2_5_7_surgeryvid_generic/last"
    # "surgeryvid_test#500 default Qwen2_5_7_surgeryvid_attention/last"
)

# Iterate over the configurations and call swift.sh
for config in "${configs[@]}"; do
    set -- $config
    dataset_name=$1
    system_prompt=$2
    checkpoint_name=$3

    sbatch qwenvl/scripts/swift_gen.sh "$dataset_name" "$system_prompt" "$checkpoint_name"
done
