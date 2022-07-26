# Quantum Dueling Slimes

## Initial Setup

```
pyenv local 3.8.12
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Training

```
tensorboard --logdir training_files/logs  # Optional, can be opened in another shell or after training
python src/train/train.py -p [TRAINING_RUN_NAME]
```

## Inference

```
python src/run/run.py -n training_files/[TRAINING_RUN_NAME]/saved_models/epoch_[EPOCH_TO_LOAD]
```

# Notes

## Training runs

### encoder_16_rest_128

Encoder network 16 nodes per dense layer, other 2 networks 128.
Reward: `reward -= 100 * x_agent`
Trained to 275 epochs
Running with C == 0:
Fast action selections, generaly moves to the right, but still jumps around a bit

```
Action selected in Phase B - depth: 10 - repeats: 10
  Probs:  0.17 | 0.06 | 0.12 | 0.00 | 0.48 | 0.17
  Counts:    1 |    1 |    1 |    1 |   11 |    1
Action selected in Phase B - depth: 10 - repeats: 10
  Probs:  0.74 | 0.00 | 0.13 | 0.06 | 0.06 | 0.01
  Counts:   11 |    1 |    1 |    1 |    1 |    1
Action selected in Phase B - depth: 10 - repeats: 10
  Probs:  0.02 | 0.11 | 0.58 | 0.00 | 0.03 | 0.25
  Counts:    1 |    1 |   11 |    1 |    1 |    1
Action selected in Phase B - depth: 10 - repeats: 10
  Probs:  0.00 | 0.10 | 0.10 | 0.41 | 0.27 | 0.13
```

When changing reward to `reward -= 100 * -x_agent`, action selection goes all the way to 300

```
Action selected in Phase C - depth: 4 - repeats: 299
  Probs:  0.00 | 0.69 | 0.04 | 0.12 | 0.11 | 0.05
  Counts:   21 |   83 |  124 |   34 |    8 |   36
Action selected in Phase C - depth: 4 - repeats: 299
  Probs:  0.31 | 0.00 | 0.22 | 0.19 | 0.14 | 0.15
  Counts:  127 |    5 |  114 |   16 |    9 |   35
Action selected in Phase C - depth: 4 - repeats: 299
  Probs:  0.25 | 0.24 | 0.00 | 0.03 | 0.25 | 0.24
  Counts:  149 |   84 |    2 |    2 |   54 |   15
Action selected in Phase C - depth: 5 - repeats: 299
  Probs:  0.19 | 0.05 | 0.00 | 0.19 | 0.21 | 0.35
  Counts:   19 |    3 |   16 |   96 |   63 |  109
```

### all_128

All dense layers have 128 nodes
Reward: `reward -= 100 * -x_agent`
Trained to 400 epochs

tanh to 0-1 bound: (1 + tanh(x)) / 2
renorm: each / sum()

Use baseline to train habitual net

## Speedup

Parallelize repeated caluclation of G

### habitual uses

1. mcts step simulate
   first: node_state mean, predicted from observation by encoder
   later steps: full state predicted by transition_net (in original code, can be changed by 'use_means')
2. habitual training compute loss, full state from obs
3. active_inference_mcts, from obs, encoded, uses mean (changed)
