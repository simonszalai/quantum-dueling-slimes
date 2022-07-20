import numpy as np

from src.model.mcts_node import Node
from src.model.mcts_utils import calc_action_threshold, normalize_distribution

node_id = 0


class MCTS:
    def __init__(self, C=1.0, threshold=0.5, repeats=300, simulation_repeats=1, simulation_depth=1, use_habit=False):
        self.C = C  # Higher value increases probability of choosing less explored actions
        self.threshold = threshold
        self.repeats = repeats
        self.simulation_repeats = simulation_repeats
        self.simulation_depth = simulation_depth
        self.use_habit = use_habit
        self.verbose = False
        self.using_prior_for_exploration = False

    def active_inference_mcts(self, model, obs):
        states_explored_count = 0
        all_paths = []  # For debugging.
        all_paths_G = []  # For debugging.
        if obs == []:
            return [0], 0, states_explored_count, all_paths, all_paths_G

        # Predict current state from observation
        state_0_mean, _ = model.encoder_net.encode(obs)

        # Important to use the mean here as we repeat it cfg.action_dim times
        root_node = Node(state=state_0_mean[0], model=model, C=self.C, using_prior_for_exploration=self.using_prior_for_exploration)

        # ============= Phase A: Habitual Network =============
        # Action will be selected in this phase if the habitual network is more confident in one action than the threshold
        # Predict probabilities for each action given the current state using the habitual network
        Q_action = model.habitual_net.predict_action(state_0_mean).numpy()

        # Remove list nesting
        root_node.Q_action = np.squeeze(Q_action)

        if self.use_habit:
            habitual_threshold = calc_action_threshold(root_node.Q_action, axis=0)
            if habitual_threshold > self.threshold:
                if self.verbose:
                    print("Action selected in Phase A |", "Q_action:", Q_action, "habitual_threshold:", habitual_threshold)

                choosen_action = np.random.choice(model.action_dim, p=root_node.Q_action)
                return [choosen_action], 0, states_explored_count, all_paths, all_paths_G
        # ============= /Phase A =============

        # Initialize child nodes for each possible action
        root_node.expand()

        # ============= Phase B: Exploration Count =============
        for repeat in range(self.repeats):
            norm_exp_counts_of_actions = normalize_distribution(root_node.exploration_counts_of_actions)
            exp_count_threshold = calc_action_threshold(norm_exp_counts_of_actions, axis=0)

            if exp_count_threshold > self.threshold:
                MCTS_choices = norm_exp_counts_of_actions
                if self.verbose:
                    print(
                        "Decision in phase B",
                        np.round(root_node.probs_for_selection(), 2),
                        np.round(MCTS_choices, 2),
                        calc_action_threshold(MCTS_choices, axis=0),
                        "N:",
                        root_node.exploration_counts_of_actions,
                    )
                final_path = root_node.action_selection(deterministic=True)
                return final_path, repeat, states_explored_count, all_paths, all_paths_G

            # Traverse path to find leaf node on the end
            path_of_nodes, path_of_actions = root_node.traverse_path_to_leaf(deterministic=True)
            leaf_node_of_path = path_of_nodes[-1]

            # Expand the leaf node
            leaf_node_of_path.expand()

            all_av_G = np.zeros(self.simulation_repeats)
            for sim_repeat in range(self.simulation_repeats):
                states_explored_count += self.simulation_depth
                all_av_G[sim_repeat], pi0, path_of_nodes[-1].Q_action = model.mcts_step_simulate(path_of_nodes[-1].node_state[0], self.simulation_depth)

            path_of_nodes[-1].backpropagate([root_node] + path_of_nodes[:-1], all_av_G.mean())
            all_paths.append(path_of_actions)
            all_paths_G.append(all_av_G.mean())

        final_path = root_node.action_selection(deterministic=True)
        if self.verbose:
            print(
                "Decision in phase C",
                root_node.exploration_counts_of_actions,
                len(all_paths),
                np.round(root_node.Q() - np.min(root_node.Q()), 2),
                np.round(
                    root_node.C * root_node.Q_action * np.sqrt(root_node.exploration_counts_of_actions.sum()) / (1.0 + root_node.exploration_counts_of_actions),
                    2,
                ),
                "Q_action:",
                np.round(root_node.Q_action, 2),
            )
        return final_path, self.repeats, states_explored_count, all_paths, all_paths_G
