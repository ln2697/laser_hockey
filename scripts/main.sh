#!/usr/bin/bash

export ROOT="/home/stud328"
export SLURM_PARTITION="day"

cd $ROOT/laser-hockey

alias tensorboard="tensorboard --logdir=runs"

# Git aliases
alias reset='git reset --hard'
alias switch='git switch'
alias pull='git pull'
alias log='git log --oneline'
alias diff='git diff'
alias checkout='git checkout'
alias fetch='git fetch'
alias add='git add .'
alias status='git status'
alias resource='source $ROOT/laser-hockey/scripts/main.sh'

commit() {
	add
	if [ -z "$1" ]; then
		git commit -m "Update"
	else
		git commit -m "$1"
	fi
}

push() {
	add
	if [ -z "$1" ]; then
		git commit -m "Update"
	else
		git commit -m "$1"
	fi
	git push
}