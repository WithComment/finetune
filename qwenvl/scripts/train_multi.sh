# sbatch qwenvl/scripts/train.sh --dataset_use openbiomedvid_qa --sys_prompt default --packing False
# sbatch qwenvl/scripts/train.sh --dataset_use openbiomedvid_qa --sys_prompt default,temporal --packing False
# sbatch qwenvl/scripts/train.sh --dataset_use openbiomedvid_qa --sys_prompt default,spatial --packing False
# sbatch qwenvl/scripts/train.sh --dataset_use openbiomedvid_qa --sys_prompt default,video --packing False
sbatch qwenvl/scripts/train.sh --dataset_use fashion_spatial --sys_prompt default --packing True
# sbatch qwenvl/scripts/train.sh --dataset_use fashion_spatial --sys_prompt default,spatial --packing True
sbatch qwenvl/scripts/train.sh --dataset_use fashion_temporal --sys_prompt default --packing True
sbatch qwenvl/scripts/train.sh --dataset_use fashion_temporal --sys_prompt default,temporal --packing True
# sbatch qwenvl/scripts/train.sh --dataset_use fashion_final --sys_prompt default --packing True
sbatch qwenvl/scripts/train.sh --dataset_use fashion_final --sys_prompt default,video --packing True