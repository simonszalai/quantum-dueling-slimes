import numpy as np
import tensorflow as tf

import src.train.config as cfg
from src.model.mcts_node import Node
from src.model.mcts_utils import calc_action_threshold, normalize_distribution

node_id = 0


class MCTS:
    def __init__(self, model, C=0.1, threshold=0.5, repeats=150, simulation_repeats=1, simulation_depth=2, use_habit=False, using_prior_for_exploration=True):
        self.model = model
        self.C = C  # Higher value increases probability of choosing less explored actions
        self.threshold = threshold
        self.repeats = repeats
        self.simulation_repeats = simulation_repeats
        self.simulation_depth = simulation_depth
        self.use_habit = use_habit
        self.using_prior_for_exploration = using_prior_for_exploration
        self.verbose = True

    def active_inference_mcts(self, obs):
        states_explored_count = 0

        # For debugging
        all_paths = []
        all_paths_G = []

        # If there is no observation, do nothing
        if obs == []:
            return [0]

        # Predict current state from observation
        _, state_0_mean, _ = self.model.encoder_net.encode(obs)

        # Important to use the mean here as we repeat it cfg.action_dim times
        root_node = Node(state=state_0_mean[0], model=self.model, C=self.C, using_prior_for_exploration=self.using_prior_for_exploration)

        # ============= Phase A: Habitual Network =============
        # Action will be selected in this phase if the habitual network is more confident in one action than the threshold
        # Predict probabilities for each action given the current state using the habitual network
        P_action = self.model.habitual_net.predict_action(state_0_mean).numpy()

        # Remove list nesting
        root_node.P_action = np.squeeze(P_action)

        if self.use_habit:
            habitual_threshold = calc_action_threshold(root_node.P_action, axis=0)
            if habitual_threshold > self.threshold:
                if self.verbose:
                    print("Action selected in Phase A |", "P_action:", P_action, "habitual_threshold:", habitual_threshold)

                choosen_action = np.random.choice(cfg.action_dim, p=root_node.P_action)
                return choosen_action
        # ============= /Phase A =============

        # Initialize child nodes for each possible action
        root_node.expand()

        # ============= Phase B: Exploration Count =============
        path_of_nodes = []
        for repeat in range(self.repeats):
            norm_exp_counts_of_actions = normalize_distribution(root_node.exploration_counts_of_actions)
            exp_count_threshold = calc_action_threshold(norm_exp_counts_of_actions, axis=0)

            if exp_count_threshold > self.threshold:
                final_path = root_node.action_selection(deterministic=True)
                if self.verbose:
                    self.print_action_selected(root_node, len(path_of_nodes), repeats=repeat, phase="B")

                return final_path[0]

            # Create path by selecting actions based on the average G of nodes
            path_of_nodes, path_of_actions = root_node.traverse_path_to_leaf(deterministic=True)

            # Expand the leaf node at the end of the path (add child nodes for each action)
            path_of_nodes[-1].expand()

            start_state = path_of_nodes[-1].node_state[0]  # Same state is saved actions_dim times, so just take the first

            # Predict action probabilities of the current node using the habitual net
            P_action_of_node = self.model.habitual_net.predict_action(start_state.reshape(1, -1))
            path_of_nodes[-1].P_action = tf.squeeze(P_action_of_node).numpy()

            # Get the mean of Gs of 'self.simulation_steps' actions executed based on the agent's internal model
            simulation_G_mean, states_explored_in_sim = self.get_G_of_internal_model(start_state)
            states_explored_count += states_explored_in_sim

            # Get full path of nodes in the tree (including the root node)
            full_path_of_nodes = [root_node, *path_of_nodes[:-1]]

            # Traverse back the tree and subtract the mean G from 'total_free_energy' of each node
            path_of_nodes[-1].backpropagate(full_path_of_nodes, simulation_G_mean)

            # Append paths to debug registers
            all_paths.append(path_of_actions)
            all_paths_G.append(simulation_G_mean)

        # ============= Phase C: Action Selection =============
        final_path = root_node.action_selection(deterministic=True)
        if self.verbose:
            self.print_action_selected(root_node, len(path_of_nodes), repeats=repeat, phase="C")

        return final_path[0]

    def print_action_selected(self, node, path_length, repeats, phase):
        probs = [str(p).ljust(4, "0") for p in np.round(node.get_probs_for_selection(), 2)]
        counts = [str(int(c)).rjust(4) for c in node.exploration_counts_of_actions]
        print(f"Action selected in Phase {phase} - depth: {path_length} - repeats: {repeats}")
        print(f"  Probs:  {' | '.join(probs)}")
        print(f"  Counts: {' | '.join(counts)}")

    def get_G_of_internal_model(self, start_state):
        """
        Executes 'self.simulation_depth' steps by selecting actions using the habitual net then predicting next states
        using the transition net, then averages the G values of each step
        """

        states_explored_in_sim = 0
        # Repeat and average G over 'simulation_repeats' times
        simulation_G_values = np.zeros(self.simulation_repeats)
        for sim_repeat in range(self.simulation_repeats):
            states_explored_in_sim += self.simulation_depth

            # Get the mean of Gs for each step down in the tree to simulation_depth (based on the agent's internal model)
            G = self.model.mcts_step_simulate(start_state, self.simulation_depth)
            simulation_G_values[sim_repeat] = G

        simulation_G_mean = simulation_G_values.mean()

        return simulation_G_mean, states_explored_in_sim
