#!/bin/sh
env="MPE"
scenario="simple_round_up2_tune"
num_landmarks=0
num_agents=5
algo="matd3"
exp="check"
seed_max=1
use_Relu=False
layer_N=2
hidden_size=32
save_data=False

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --save_gifs share_policy=1 --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --use_Relu ${use_Relu} --layer_N ${layer_N} --hidden_size ${hidden_size} \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 200 --render_episodes 100 \
    --model_dir "/home/sdc/dachuang_space/Project/zcy_space/MATD3_v2/onpolicy/scripts/results/MPE/simple_round_up2_tune/matd3/check/wandb/run-20240126_093652-j8wkcl6o/files"\
    --save_data ${save_data}
done
