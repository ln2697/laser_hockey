# Pretrained weight
We included the pretrained-weight `pretrained_model.pth` and its config. To see how the model plays against the strong bot, run

```bash
python3 hockey_agent.py
```

# Set up steps
- Hook `scripts/main.sh` to `.bashrc`.
- Run an experiment
```bash
bash scripts/001_baseline.sh
```
- The training can take around 5 - 7 hours on a GTX 1080ti and 8-12 CPU cores with `max_avg_gradient_steps_per_frame=4`.
- Tensorboard logs, model and config will be saved at `run/001_baseline_<run_id>`