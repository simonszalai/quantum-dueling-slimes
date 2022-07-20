import numpy as np
import tensorflow as tf

import src.train.config as cfg
from src.model.mcts_utils import select_action_for_node


node_id = 0


class Node:
    def __init__(self, state, model, C, verbose=False, using_prior_for_exploration=False):
        global node_id

        # The latent state that corresponds to this node
        self.node_state = np.stack((state,) * cfg.action_dim, axis=0)  # NOTE: It's saved cfg.action_dim times to simplify calculations
        self.model = model
        self.verbose = verbose
        self.using_prior_for_exploration = using_prior_for_exploration
        self.visited = False  # For visualization
        self.node_id = node_id
        node_id += 1

        if self.verbose:
            print(f"Node-{node_id} created")

        # Define an accumulator to store total G for each action
        self.total_free_energy = np.zeros(cfg.action_dim)

        # Define an accumulator to store how many times each action was explored in this node
        self.exploration_counts_of_actions = np.zeros(cfg.action_dim)

        # Prior probability distribution for actions
        self.Q_action = np.zeros(cfg.action_dim)

        # Create placeholders for child nodes for each action
        self.child_nodes = [None for _ in range(cfg.action_dim)]

        # Higher value increases probability of choosing less explored actions
        # A constant in AlphaGo Zero: it was 1.0 in that paper. Range: 1.0 - 100.0
        self.C = C

        # This takes the index of the child node that is currently under investigation
        # It's important for the back-propagation of G, to remember which action was taken
        self.action_in_progress = -1

    def get_normalized_free_energy_of_actions(self):
        """
        Original name: Q
        """

        average_free_energy_of_actions = self.total_free_energy / self.exploration_counts_of_actions
        average_free_energy_of_actions -= average_free_energy_of_actions.min()
        average_free_energy_of_actions = average_free_energy_of_actions / average_free_energy_of_actions.sum()

        return average_free_energy_of_actions

    def get_probs_for_selection(self):
        norm_free_energy_of_actions = self.get_normalized_free_energy_of_actions()
        bonus_of_less_explored_actions = self.C * 1.0 / self.exploration_counts_of_actions

        if self.using_prior_for_exploration:
            bonus_of_less_explored_actions *= self.Q_action

        return norm_free_energy_of_actions + bonus_of_less_explored_actions

    def traverse_path_to_leaf(self, deterministic=True):
        path_of_nodes = []
        path_of_actions = []

        # Select action for current node
        self.action_in_progress = select_action_for_node(self, deterministic)

        # Add selected action to path_of_actions
        path_of_actions.append(self.action_in_progress)

        # Get child node that belongs to the selected action
        node_of_selected_action = self.child_nodes[self.action_in_progress]

        # Add child node to path_of_nodes
        path_of_nodes.append(node_of_selected_action)

        # Traverse the path until hitting a leaf node
        while None not in path_of_nodes[-1].child_nodes:
            # Get last node from path_of_nodes
            last_node = path_of_nodes[-1]

            # Select action for last node
            last_node.action_in_progress = select_action_for_node(last_node, deterministic)

            # Add child node of the last node that belongs to the selected action to path_of_nodes
            path_of_nodes.append(last_node.child_nodes[last_node.action_in_progress])

            # Add action selected for last node to path_of_actions
            path_of_actions.append(last_node.action_in_progress)

        if self.verbose:
            print(f"select_action of Node-{self.node_id}", "Node IDs:", [p.node_id for p in path_of_nodes], "actions:", path_of_actions)

        return path_of_nodes, path_of_actions

    def expand(self):
        """
        Creates child nodes for each possible action and assigns a state (predicted by the transition networks)
        and initial expected free energy (calculated from the current state and action belonging to the child node)
        """

        # Create a tensor that contains each possible actions one-hot encoded
        all_actions_onehot = tf.eye(cfg.action_dim, dtype=cfg.np_precision)

        # Calculate G from current state (same state is duplicated cfg.action_dim) times for each action
        # Also return the next state predicted by the transition network
        G, pred_next_states, _ = self.model.calculate_G(self.node_state, all_actions_onehot, average_G_over_N_samples=1)

        # Update accumulators
        self.total_free_energy -= G.numpy()  # NOTE: Negative expected free energy to be used as a Q value in RL applications
        self.exploration_counts_of_actions += 1.0

        # Assign a child node for each possible action
        for i in range(cfg.action_dim):
            self.child_nodes[i] = Node(state=pred_next_states[i], model=self.model, C=self.C, using_prior_for_exploration=self.using_prior_for_exploration)

        if self.verbose:
            print(f"Expanded Node-{self.node_id} |", "Total free energy:", self.total_free_energy, "Exploration count:", self.exploration_counts_of_actions)

    # @tf.function
    def backpropagate(self, path, G):
        if self.verbose:
            print("Back-propagate:", [p.node_id for p in path], G.numpy())
        for i in range(len(path)):
            if path[i].action_in_progress < 0:
                exit("Back-propagation error: " + str(path) + " " + str(i))
            path[i].total_free_energy[path[i].action_in_progress] -= G
            path[i].exploration_times[path[i].action_in_progress] += 1
            path[i].action_in_progress = -2  # just to remember it's been examined..
            if self.verbose:
                print("Propagating to node", path[i].node_id, "with N:", path[i].exploration_times)

    # @tf.function
    def action_selection(self, deterministic=True):
        path = []
        if deterministic:
            path.append(np.argmax(self.exploration_counts_of_actions))
        else:
            path.append(np.random.choice(cfg.action_dim, p=self.normalization(self.exploration_counts_of_actions)))
        node = self.child_nodes[path[-1]]
        if self.verbose:
            print(len(path), node.node_id)
        while None not in node.child_nodes:
            if deterministic:
                path.append(np.argmax(node.exploration_times))
            else:
                path.append(np.random.choice(cfg.action_dim, p=self.normalization(node.exploration_times)))
            node = node.child_nodes[path[-1]]
            if self.verbose:
                print(len(path), node.node_id)

        trimmed_path = []
        i = 0
        while i < len(path) - 1:
            if cfg.action_dim == 4:
                if (
                    (path[i] == 0 and path[i + 1] == 1)
                    or (path[i] == 1 and path[i + 1] == 0)
                    or (path[i] == 2 and path[i + 1] == 3)
                    or (path[i] == 3 and path[i + 1] == 2)
                ):
                    i += 2
                else:
                    trimmed_path.append(path[i])
                    i += 1
            elif cfg.action_dim == 3:
                if (path[i] == 1 and path[i + 1] == 2) or (path[i] == 2 and path[i + 1] == 1):
                    i += 2
                else:
                    trimmed_path.append(path[i])
                    i += 1
            else:
                exit("Error: Unknown number of pi_dim " + str(cfg.action_dim))
        if self.verbose:
            print("Action selection:", path, "trimmed path:", trimmed_path)
        return path
