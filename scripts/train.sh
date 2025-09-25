#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=long.nguyen@student.uni-tuebingen.de
#SBATCH --mem=48G

git rev-parse --short HEAD && git branch --show-current

git rev-parse --short HEAD >> $EXPERIMENT_OUTPUT_DIR/commit.txt
git branch --show-current >> $EXPERIMENT_OUTPUT_DIR/commit.txt

cp *.py $EXPERIMENT_OUTPUT_DIR/code/

scontrol show job $SLURM_JOB_ID

export RAY_DEDUP_LOGS=0
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=0

eval "$($ROOT/miniconda3/bin/conda shell.bash hook)"
source activate laser-hockey
which python3

nvidia-smi

# If on local PC, change experiment parameters
if ! which sbatch >/dev/null; then
	export EXPERIMENT_PARAMETERS="$EXPERIMENT_PARAMETERS evaluation.start_evaluation=0"
	# export EXPERIMENT_PARAMETERS="$EXPERIMENT_PARAMETERS wandb.activated=false"
	# export EXPERIMENT_PARAMETERS="$EXPERIMENT_PARAMETERS evaluation.n_games=10"
	# export EXPERIMENT_PARAMETERS="$EXPERIMENT_PARAMETERS ray.log_freq=1"
	# export EXPERIMENT_PARAMETERS="$EXPERIMENT_PARAMETERS rl.target_network_update_freq=1"
	export EXPERIMENT_PARAMETERS="$EXPERIMENT_PARAMETERS rl.n_warmup=100"
	echo $EXPERIMENT_PARAMETERS
fi

echo "Extra parameters: $EXPERIMENT_PARAMETERS"
python -u train.py $EXPERIMENT_PARAMETERS