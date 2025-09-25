#!/usr/bin/bash

set -e
shopt -s globstar

# Help to identify the run. Don't change this.
export SCRIPT_NAME=$(basename "$0" ".sh") # Script name
export SLURM_JOB_DATE=$(date +"%y%m%d_%H%M%S") # Current date
export EXPERIMENT_RUN_ID=${SCRIPT_NAME}_${SLURM_JOB_DATE}  # Experiment ID

# if Experiment ID has more than 64 characters, error
if [ ${#EXPERIMENT_RUN_ID} -gt 64 ]; then
	echo "Experiment ID too long: ${EXPERIMENT_RUN_ID}"
	exit 1
fi

function train() {
    export EXPERIMENT_OUTPUT_DIR="runs/$EXPERIMENT_RUN_ID"
	mkdir -p "$EXPERIMENT_OUTPUT_DIR" "$EXPERIMENT_OUTPUT_DIR/code"
	# Submit the job
	echo "$EXPERIMENT_OUTPUT_DIR"
	output_file=$EXPERIMENT_OUTPUT_DIR/out.txt
	error_file=$EXPERIMENT_OUTPUT_DIR/err.txt
	if which sbatch &>/dev/null; then
		echo "${output_file}"
		echo "${error_file}"
		sbatch --output "${output_file}" "--error" "${error_file}" "--job-name" "${EXPERIMENT_RUN_ID}" "--partition" ${SLURM_PARTITION} "$@" scripts/train.sh
	else
		bash scripts/train.sh # > "${output_file}" 2> "${error_file}"
	fi
}