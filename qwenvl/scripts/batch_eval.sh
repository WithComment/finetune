cd ~/finetune/sandbox-ethan

declare -a configs=(
    # "surgeryvid_test#500 default Qwen2_5_7_surgeryvid_default"
    # "surgeryvid_test#500 default Qwen2_5_7_surgeryvid_v1_2"
    # "surgeryvid_test#500 default Qwen2_5_7_surgeryvid_generic"
    # "surgeryvid_test#500 default Qwen2_5_7_surgeryvid_attention"
    "surgeryvid_test#1000 default Qwen2_5_7_surgeryvid_default"
    "surgeryvid_test#1000 default Qwen2_5_7_surgeryvid_v1_0"
    "surgeryvid_test#1000 v1_0 Qwen2_5_7_surgeryvid_v1_0"
    "surgeryvid_test#1000 v1_4 Qwen2_5_7_surgeryvid_v1_4"
)

# Iterate over the configurations and call swift.sh
for config in "${configs[@]}"; do
    set -- $config
    dataset_name=$1
    system_prompt=$2
    checkpoint_name=$3

    bash /home/xiaowenz/finetune/sandbox-ethan/single_eval.sh "$dataset_name" "$system_prompt" "$checkpoint_name" &
done