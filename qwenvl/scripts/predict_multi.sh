# bash qwenvl/scripts/predict.sh --dataset_use vqa_rad --sys_prompt default --usr_prompt vqa_rad,default --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct
# bash qwenvl/scripts/predict.sh --dataset_use vqa_rad --sys_prompt default --usr_prompt vqa_rad,default --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct
# bash qwenvl/scripts/predict.sh --dataset_use vqa_rad --sys_prompt default --usr_prompt default --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct
# bash qwenvl/scripts/predict.sh --dataset_use vqa_rad --sys_prompt default --usr_prompt default --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct
# bash qwenvl/scripts/predict.sh --dataset_use vqa_rad --sys_prompt default --usr_prompt generic_vqa,default --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct
# bash qwenvl/scripts/predict.sh --dataset_use vqa_rad --sys_prompt default --usr_prompt generic_vqa,default --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct
# bash qwenvl/scripts/predict.sh --dataset_use path_vqa --sys_prompt default --usr_prompt path_vqa,default --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct
# sbatch qwenvl/scripts/predict.sh --dataset_use path_vqa --sys_prompt default --usr_prompt path_vqa,default --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct
# sbatch qwenvl/scripts/predict.sh --dataset_use path_vqa --sys_prompt default --usr_prompt default --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct
# sbatch qwenvl/scripts/predict.sh --dataset_use path_vqa --sys_prompt default --usr_prompt default --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct
# sleep 300
# sbatch qwenvl/scripts/predict.sh --dataset_use path_vqa --sys_prompt default --usr_prompt generic_vqa,default --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct
# sleep 300
# sbatch qwenvl/scripts/predict.sh --dataset_use path_vqa --sys_prompt default --usr_prompt generic_vqa,default --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct

# sbatch qwenvl/scripts/predict.sh --dataset_use path_vqa --sys_prompt default --usr_prompt default --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct
# sleep 300
# sbatch qwenvl/scripts/predict.sh --dataset_use path_vqa --sys_prompt default,path_vqa --usr_prompt default --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct

# sbatch qwenvl/scripts/predict.sh --dataset_use path_vqa --sys_prompt default --usr_prompt default --model_name_or_path Qwen2.5-VL-3B-Instruct-path_vqa_sys_default
# sbatch qwenvl/scripts/predict.sh --dataset_use path_vqa --sys_prompt default,path_vqa --usr_prompt default --model_name_or_path Qwen2.5-VL-3B-Instruct-path_vqa_sys_default
# sbatch qwenvl/scripts/predict.sh --dataset_use path_vqa --sys_prompt default --usr_prompt default --model_name_or_path Qwen2.5-VL-3B-Instruct-path_vqa_sys_default_path_vqa
# sbatch qwenvl/scripts/predict.sh --dataset_use path_vqa --sys_prompt default,path_vqa --usr_prompt default --model_name_or_path Qwen2.5-VL-3B-Instruct-path_vqa_sys_default_path_vqa
# sbatch qwenvl/scripts/predict.sh --dataset_use surgeryvid --sys_prompt default --usr_prompt default --model_name_or_path Qwen2.5-VL-3B-Instruct-openbiomedvid_qa_sys_default
# sbatch qwenvl/scripts/predict.sh --dataset_use surgeryvid --sys_prompt default,surgeryvid --usr_prompt default --model_name_or_path Qwen2.5-VL-3B-Instruct-openbiomedvid_qa_sys_default
# sbatch qwenvl/scripts/predict.sh --dataset_use surgeryvid --sys_prompt default --usr_prompt default --model_name_or_path Qwen2.5-VL-3B-Instruct-openbiomedvid_qa_sys_default_surgeryvid
# sbatch qwenvl/scripts/predict.sh --dataset_use surgeryvid --sys_prompt default,surgeryvid --usr_prompt default --model_name_or_path Qwen2.5-VL-3B-Instruct-openbiomedvid_qa_sys_default_surgeryvid

# bash qwenvl/scripts/predict.sh --dataset_use surgeryvid_small --sys_prompt default --usr_prompt default --model_name_or_path Qwen2.5-VL-3B-Instruct-openbiomedvid_qa_sys_default
bash qwenvl/scripts/predict.sh --dataset_use surgeryvid_small --sys_prompt default --usr_prompt default --model_name_or_path Qwen2.5-VL-3B-Instruct-openbiomedvid_qa_sys_default_temporal &
bash qwenvl/scripts/predict.sh --dataset_use surgeryvid_small --sys_prompt default --usr_prompt default --model_name_or_path Qwen2.5-VL-3B-Instruct-openbiomedvid_qa_sys_default_spatial &
bash qwenvl/scripts/predict.sh --dataset_use surgeryvid_small --sys_prompt default --usr_prompt default --model_name_or_path Qwen2.5-VL-3B-Instruct-openbiomedvid_qa_sys_default_video &
sleep 60

bash qwenvl/scripts/predict.sh --dataset_use surgeryvid_small --sys_prompt default,temporal --usr_prompt default --model_name_or_path Qwen2.5-VL-3B-Instruct-openbiomedvid_qa_sys_default &
bash qwenvl/scripts/predict.sh --dataset_use surgeryvid_small --sys_prompt default,temporal --usr_prompt default --model_name_or_path Qwen2.5-VL-3B-Instruct-openbiomedvid_qa_sys_default_temporal &
# sleep 60

bash qwenvl/scripts/predict.sh --dataset_use surgeryvid_small --sys_prompt default,spatial --usr_prompt default --model_name_or_path Qwen2.5-VL-3B-Instruct-openbiomedvid_qa_sys_default &
bash qwenvl/scripts/predict.sh --dataset_use surgeryvid_small --sys_prompt default,spatial --usr_prompt default --model_name_or_path Qwen2.5-VL-3B-Instruct-openbiomedvid_qa_sys_default_spatial &
sleep 60

bash qwenvl/scripts/predict.sh --dataset_use surgeryvid_small --sys_prompt default,video --usr_prompt default --model_name_or_path Qwen2.5-VL-3B-Instruct-openbiomedvid_qa_sys_default &
bash qwenvl/scripts/predict.sh --dataset_use surgeryvid_small --sys_prompt default,video --usr_prompt default --model_name_or_path Qwen2.5-VL-3B-Instruct-openbiomedvid_qa_sys_default_video &
sleep 60

bash qwenvl/scripts/predict.sh --dataset_use surgeryvid_small --sys_prompt default --usr_prompt default --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct &
bash qwenvl/scripts/predict.sh --dataset_use surgeryvid_small --sys_prompt default,spatial --usr_prompt default --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct &
bash qwenvl/scripts/predict.sh --dataset_use surgeryvid_small --sys_prompt default,temporal --usr_prompt default --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct &
bash qwenvl/scripts/predict.sh --dataset_use surgeryvid_small --sys_prompt default,video --usr_prompt default --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct &
