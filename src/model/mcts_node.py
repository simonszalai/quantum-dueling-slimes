import numpy as np
import tensorflow as tf

import src.train.config as cfg
from src.model.mcts_utils import select_action_from_dist


node_id = 0


class Node:
    def __init__(self, state, model, C, verbose=False, using_prior_for_exploration=True):
        global node_id

        # The latent state that corresponds to this node NOTE: The same state is saved cfg.action_dim times to enable calculation of G as a batch
        self.node_state = np.stack((state,) * cfg.action_dim, axis=0)
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
        self.P_action = np.zeros(cfg.action_dim)

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
        average_free_energy_of_actions /= average_free_energy_of_actions.sum()

        return average_free_energy_of_actions

    def get_probs_for_selection(self):
        norm_free_energy_of_actions = self.get_normalized_free_energy_of_actions()
        bonus_of_less_explored_actions = self.C * 1.0 / self.exploration_counts_of_actions

        # Boost probability of actions that would be visited by habit but were not visited often
        if self.using_prior_for_exploration:
            bonus_of_less_explored_actions *= self.P_action

        return norm_free_energy_of_actions + bonus_of_less_explored_actions

    def traverse_path_to_leaf(self, deterministic=True):
        path_of_nodes = []
        path_of_actions = []

        # Select action for current node
        self.action_in_progress = select_action_from_dist(self.get_probs_for_selection(), deterministic)

        # Add selected action to path_of_actions
        path_of_actions.append(self.action_in_progress)

        # Get child node that belongs to the selected action
        node_of_selected_action = self.child_nodes[self.action_in_progress]

        # Add child node to path_of_nodes
        path_of_nodes.append(node_of_selected_action)

        # Traverse the path until hitting a leaf node
        while None not in path_of_nodes[-1].child_nodes:
            # Select action for last node
            path_of_nodes[-1].action_in_progress = select_action_from_dist(path_of_nodes[-1].get_probs_for_selection(), deterministic)

            # Add child node of the last node that belongs to the selected action to path_of_nodes
            path_of_nodes.append(path_of_nodes[-1].child_nodes[path_of_nodes[-1].action_in_progress])

            # Add action selected for last node to path_of_actions
            path_of_actions.append(path_of_nodes[-1].action_in_progress)

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

        # Increment exploration count of each action
        self.exploration_counts_of_actions += 1.0

        # Assign a child node for each possible action
        for i in range(cfg.action_dim):
            self.child_nodes[i] = Node(state=pred_next_states[i], model=self.model, C=self.C, using_prior_for_exploration=self.using_prior_for_exploration)

        if self.verbose:
            print(f"Expanded Node-{self.node_id} |", "Total free energy:", self.total_free_energy, "Exploration count:", self.exploration_counts_of_actions)

    # @tf.function
    def backpropagate(self, path, G):
        """
        Updates G values for all nodes in the traversed path
        """

        if self.verbose:
            print("Back-propagate:", [p.node_id for p in path], G.numpy())

        for i in range(len(path)):
            current_action = path[i].action_in_progress
            if current_action < 0:
                exit("Back-propagation error: " + str(path) + " " + str(i))

            path[i].total_free_energy[current_action] -= G
            path[i].exploration_counts_of_actions[current_action] += 1
            path[i].action_in_progress = -2  # just to remember it's been examined

            if self.verbose:
                print("Propagating to node", path[i].node_id, "with N:", path[i].exploration_counts_of_actions)

    # @tf.function
    def action_selection(self, deterministic=True):
        """
        Traverses the generated path and at each step selects the most frequently explored action.
        """

        # ============ Phase A - Build path of most frequently explored actions
        path_of_actions = []

        # First append the most frequently explored action
        action_0 = select_action_from_dist(self.exploration_counts_of_actions, deterministic)
        path_of_actions.append(action_0)

        # Current node is the one belonging to the most frequently explored action
        selected_child_node = self.child_nodes[action_0]

        # Traverse to the leaf node at the end of the path
        while None not in selected_child_node.child_nodes:
            action_of_node = select_action_from_dist(selected_child_node.exploration_counts_of_actions, deterministic)
            path_of_actions.append(action_of_node)

            if self.verbose:
                print(f"Traversed Node-{selected_child_node.node_id}. Total length of path: {len(path_of_actions)}")

            selected_child_node = selected_child_node.child_nodes[action_of_node]

        # TODO: refactor code to reduce jitter
        # ============ Phase B - remove subsequent actions that are canceling each other out
        # trimmed_path = []
        # i = 0
        # while i < len(path_of_actions) - 1:
        #     current_action = path_of_actions[i]
        #     next_action = path_of_actions[i]

        #     if cfg.action_dim == 4:
        #         if (
        #             (current_action == 0 and next_action == 1)
        #             or (current_action == 1 and next_action == 0)
        #             or (current_action == 2 and next_action == 3)
        #             or (current_action == 3 and next_action == 2)
        #         ):
        #             i += 2
        #         else:
        #             trimmed_path.append(current_action)
        #             i += 1
        #     elif cfg.action_dim == 3:
        #         if (current_action == 1 and next_action == 2) or (current_action == 2 and next_action == 1):
        #             i += 2
        #         else:
        #             trimmed_path.append(current_action)
        #             i += 1
        #     else:
        #         exit("Error: Unknown number of pi_dim " + str(cfg.action_dim))

        # if self.verbose:
        #     print("Action selection:", path_of_actions, "trimmed path:", trimmed_path)

        return path_of_actions
