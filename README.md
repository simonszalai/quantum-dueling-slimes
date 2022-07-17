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
tensorboard --logdir training_files/logs
python src/train/train.py
```

## Inference

```
python src/run/run.py -n training_files/20220715-203315/saved_models/epoch_1000
```
