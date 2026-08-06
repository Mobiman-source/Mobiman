#!/bin/sh
env="tgn"
scenario="multi-cell" 
num_cell=3
algo="mappo"
exp="check"
seed_max=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$(dirname "$SCRIPT_DIR"):$PYTHONPATH"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/runner/train_sc.py" --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_cell ${num_cell} --num_ue 8 --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 1 --episode_length 25 --num_env_steps 20000000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4\
    --wandb_name "xxx" --user_name "your-wandb-entity"
done

