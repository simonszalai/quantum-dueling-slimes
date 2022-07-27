# Quantum Dueling Slimes

_by @pldallairedemers and @simonszalai_

## Instructions

### Initial Setup

```
pyenv local 3.8.12               # Set local Python version
python -m venv .venv             # Create virtual environment
source .venv/bin/activate        # Activate virtual environment
pip install -r requirements.txt  # Install required packages from PyPI
pip install -e .                 # Install 'quantum-dueling-slimes' in editable mode
pip install -e ./slimevolleygym  # Install 'slimevolleygym' in editable mode

```

### Training

```
python src/train/train.py -p [TRAINING_RUN_NAME] -r [RENDER] -e [EPOCHS]
```

TRAINING_RUN_NAME (str):
Folder name (within `training_files` folder) where the files related to the training (TensorBoard logs and saved models) should be saved. If empty, an date based auto-generated name will be used. The model gets automatically saved every 25 epochs.
RENDER (bool):
Shows training progress by rendering the environment, but also slows down training.
EPOCHS (int):
Number of epochs (games) to train. Default is 300, which is ideal for development as it trains relatively fast (~20min) and decent convergence can be observed. It could make sense to increase it to around 1000 as it improves convergence significantly.

### Inference

```
python src/run/run.py -n training_files/[TRAINING_RUN_NAME]/saved_models/epoch_[EPOCH_TO_LOAD] -ea [EXPORT_ACTIONS]
```

TRAINING_RUN_NAME (str): Name of the training run (same as in training) to be loaded.
EPOCH_TO_LOAD (int): Epoch number to be loaded. Make sure that it exists in the corresponding training_files folder.
EXPORT_ACTIONS (str): Name of the run where the actions are exported in the end (in `run_files` folder).

### Replay

MCTS can take a long time to compute, which dramatically reduces the frame rate. To have a more pleasant way to watch the games, the sequence of actions during inference are saved to a json file (specified under -ea flag). Using the command below, the pre-generated games can be smoothly replayed (using the same environment).

```
python src/run/replay.py -ia [IMPORT_ACTIONS]
```

IMPORT_ACTIONS (str): Name of the run where the actions were exported in the end of inference (in `run_files` folder). Corresponds to the EXPORT_ACTIONS flag on the inference script.

## Commands to recreate the videos

The commands below were the ones used to create the videos in the presentation. All the files required to run them are committed to the repo.

### Training

```
python src/train/train.py -p baseline -e 300
```

Training logs:

```
tensorboard --logdir training_files/logs
```

### Inference

Classical vs. classical:

```
python src/run/run.py -n training_files/baseline/saved_models/epoch_300 -ea classical-vs-classical
```

Classical vs. quantum:

```
python src/run/run.py -n training_files/baseline/saved_models/epoch_300 -ea classical-vs-quantum
```

### Replay

Classical vs. classical:

```
python src/run/replay.py -ia classical-vs-classical
```

Classical vs. quantum:

```
python src/run/replay.py -ia classical-vs-quantum
```

## Technical Explanation of the Codebase

### Neural Networks & Training

Within the active inference framework, 3 neural networks were trained. A brief summary of their purpose and method of training follows:

#### Habitual Network

**Purpose**: The main purpose of the habitual network is to increase the efficiency of MCTS (Monte Carlo Tree Search). It can be compared to habit forming of biological agents, where habits reduce the computational burden in frequently encountered situations. It takes a state as an input, and returns a probabilty distibution over actions as an output. This probability distribution is used in two ways within the active inference framework. First, if one of the actions has much higher probability than the rest, it is just selected, then the whole MCTS process can be skipped. Second, if this is not the case, it is used to bias the MCTS process by adding extra probability for those actions that were frequently selected according to the habitual network.
**Training**: The habitual network was trained to mimic the actions taken by the baseline policy provided by the `slimevolleygym` environment. This baseline policy is a tiny, 120-parameter Recurrent Neural Network (RNN). During training, besides the agent, a 'shadow' agent, an instance of the baseline policy was set up, which also predicted an action for the same observation as the agent. This predicted probability distribution over actions was passed to the habitual network, where the loss was defined as the Kullback-Leibler Divergence between this, passed distribution and the one predicted by the habitual network.

#### Transition Network

**Purpose**: The transition network is a variational autoencoder that takes a state and an action as its input, and predicts what the next state should be. This represents the agent's internal model, which can be thought of as the agent's understanding of the dynamics of the environment where it lives. With another words: if the agent is in a particular state and takes a specific action, what happens next? How will the envrionment react?
**Training**: During training, in each state the agents predict what the next state will be, given that a particular action is taken. After that, the same action is applied to the environment, which returns the next state, the actual result of that action. The loss is defined as the Kullback-Leibler Divergence between the predicted next state and the actual next state. The agent will strive to minimize the difference (surprise), effectively learning the dynamics of the environment.

#### Encoder Network

**Purpose**: The encoder network is also a variational autoencoder, which has the task of encoding a high-dimensional observation to a low-dimensional latent space. In our experiments we directly accessed the state of the environment, which is a 12-dimensional vector, so in this particular case the encoder network was not particularly useful. In other use cases, for example where the observations that the agent receives are raw pixels, the encoder network (for example by using a few convolutional layers) can transform these observations to tractable latent states.
**Training**: During training, an observation is passed to the encoder network. First the network encodes the observation to a latent state, then it decodes it back to a predicted observation. The binary cross-entropy between the actual and predicted observations will be the first term of the loss. The second term will be the Kullback-Leibler Divergence between the encoded state and the state predicted by the transition network.

### Monte Carlo Tree Search

Monte Carlo Tree Search is a technique to select the action that yields the lowest free energy, considering the foreseeable future. The way it works if the following:

- **Phase A**: This phase is just a performance optimization. The habitual network predicts a probabilty distribution over actions, and if one of the actions are much more likely than all the others, that action is selected and the rest of the MCTS process is skipped.
- **Phase B**: This is the main MCTS loop. It is executed many times (few hundreds), until one of the actions reach a probability that exceeds the threshold. In each iteration, first the most traversed path is traversed again, down to the last (leaf) node. Here Expected Free Energy is calculated for every action and stored in the node. After that, the node is expanded: a new child node is added for each possible action. These children nodes are initialized with states predicted by the transition network, using the state of their parent nodes and the action that leads to them as inputs. After that, an MCTS simulation is launched from the leaf node (the same node that got expanded): alternating the transition and habitual networks to a pre-configured depth, a likely path into the future is generated. Finally, the Expected Free Energy of this path is calculated, which is backpropagated to all the nodes in the chain that lead to the current leaf node.
  This loop is repeated until an action reaches the threshold: then the action with the lowest Expected Free Energy is selected. If none if them does until the repetition limit is reached, MCTS moves to the final phase, C.
- **Phase C**: Since no action is an obvious winner, but one still has to be selected, the final action is choosen the same way as if one would reach the threshold in Phase B: the starting action of the path that results in the lowest total Expected Free Energy.

### Quantum Noise

Quantum noise was retrieved though Xanadu Cloud using `strawberry_fields`. There are 2 files committed to the repo: `quantum_noise/x8.py` and `quantum_noise/borealis.py`. Both of them does the same thing: run a circuit to sample 10000 shots of quantum noise, then saves the results to several files, so they can be used later, during training or inference. This is done purely for performance optimization purposes, as it takes areound 20s to execute the quantum circuit, and if that has to be done every iteration of the training (which normally takes a few milliseconds), training becomes intractable very fast.

## Results

1. **Classical vs. classical - trained against baseline**

   ![results_1](assets/classical-vs-classical.gif)

   In this scanario, two identical classical agent play against each other. They have the same network architecture and the same weights. The weights were obtained by training them against the baseline RNN policy, as described below. Both of them use the habitual network for faster action selection (when the habitual network is confident enough), and also they use the haitual network to bias the MCTS process towards actions that are more likely to be selected by the habitual network. As it is visible on the video below, the agents mostly struggle to hit the ball, this is due to the lack of hyperparameter tuning (time constraints of the Hackathon), and selection of hyperparameters that prioritize speed instead of accuracy (training on a laptop CPU).

2. **Classical vs. quantum - trained against baseline**

   ![results_2](assets/classical-vs-classical.gif)

   In this scanario, the only difference from the above case is that the quantum agent (the blue one on the left), has some quantum noise obtained from Xanadu's X8 device injected as additional input of its habitual network. In the classical agents, the noise is replaced with zeros. As visible on the video, the agent performs slightly worse than the classical counterpart, as expected. The reason for this is that the classical agent wasn't trained against the quantum agent, so it is not in a disadvantage of predicting it's opponent's behavior. On the other hand, the quantum agent has extra noise added, so it can hit the ball even less accurately.
   Also, since the quantum layer was not trainable, and it didn't have feedback during training, the added quantum noise degraded the agent's performance. If the quantum sampler would be trainable as well (through PennyLane), it could add such a noise that still makes it unpredictable for the opponent, but does not degrade its performance.

3. **Classical vs. quantum - trained against each other**

   This is the final goal that we strive to achieve, but some dependencies could not be completed due to time constraints. Here we expect to see a clear advantage of the quantum agent, as it can produce behavior that is unexpected for it's opponent.

## Limitations & Future work

### Major items

- **Tuning the networks**: Given the 48h time constraint of the Hackathon, unfortunately we were not able to spend enough time experimenting and tuning the hyperparameters such that the active inference agent outperforms the baseline. To progress further, this has to be done first.
- **Make the quantum sampler trainable**: Since the quantum layer was not trainable, and it didn't have feedback during training, the added quantum noise degraded the agent's performance. If the quantum sampler would be trainable as well (through PennyLane), it could add such a noise that still makes it unpredictable for the opponent, but does not degrade its performance.
- **Train a classical agent against a quantum agent**: Once the classical agents have the ability to play the game with a relatively high accuracy, we can train a classical agent (no quantum noise added, zeros as placeholders instead) and a quantum agent (quantum noise added to the input of the habitual network). Since the agents have to model each other's behavior in order to win the game, we expect to see a clear advantage of the quantum agent, as the classical one will struggle to model it's opponent's behavior.
- **Better quantum noise**: During our experiments, unfortunately Xanadu's latest device, Borealis was offline, so we couldn't use it to generate the noise added to the habitual network. Instead, we used X8, which has 8 qmodes, opposed to the 216 modes of Borealis. The quantum distribution generated by X8 is not intractable classically, so - in theory - the classical agent could learn that distribution. In practice, however, the relatively small neural networks that we used (512-node dense layers), most likely lack the capacity to do that.
- **Speed up MCTS**: MCTS is the clear bottleneck during action selection. Several things can be done to make it faster:
  1. Refactor the code so it can run with Eager execution disabled
  2. Parallalize blocks of computation that doesn't depend on each other's outputs
  3. Tune and optimize hyperparameters of MCTS

### Minor items

- **Improve performance**: Since random sampling is involved for each free energy calculation, it reduces jitter and improves performance if free energy is calculated multiple times and an average is used. Currently this is calculated in a for loop. Since the calculations are independent from each other, this could be paralllelized using multiprocessing to speed up training and inference.
- **Reduce jitter**: When actions are predicted multiple timesteps in the future, sometimes it occurs that subsequent actions cancel each other out (e.g. left, then right). By removing these jitter could be reduced and performance could be increased.
