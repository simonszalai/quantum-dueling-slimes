import numpy as np
import tensorflow as tf

import src.train.config as cfg


node_id = 0


class Node:
    def __init__(self, s, model, C, verbose=False, using_prior_for_exploration=False):
        global node_id

        # The latent state that corresponds to this node
        self.node_state = np.stack((s,) * cfg.action_dim, axis=0)  # NOTE: It's saved cfg.action_dim times to simplify calculations
        self.model = model
        self.verbose = verbose
        self.using_prior_for_exploration = using_prior_for_exploration
        self.visited = False  # For visualization
        self.node_id = node_id
        node_id += 1

        if self.verbose:
            print("NEW NODE:", np.shape(self.node_state), self.node_id)

        self.total_free_energy = np.zeros(cfg.action_dim)  # The total value of G so far in the next state given an edge
        self.exploration_times = np.zeros(cfg.action_dim)  # The total number of times an action was explored from here
        self.Qpi = np.zeros(cfg.action_dim)  # Prior probability distribution for actions
        self.children_nodes = [None for _ in range(cfg.action_dim)]
        self.C = C  # 100.0 # 1.0 # A constant in AlphaGo Zero. [it was 1.0 in that paper]
        # This takes the index of the child node that is currently under investigation
        # It's important for the back-propagation of G, to remember which action was taken
        self.in_progress = -1

    def Q(self):
        return self.total_free_energy / self.exploration_times

    def probs_for_selection(self):
        # Initially normalize Q() to make it a distribution.
        Qnormed = self.Q()
        Qnormed -= Qnormed.min()
        Qnormed = Qnormed / Qnormed.sum()
        if self.using_prior_for_exploration:
            return Qnormed + self.C * self.Qpi * 1.0 / (self.exploration_times)
        else:
            return Qnormed + self.C * 1.0 / (self.exploration_times)

    def select(self, deterministic=True):
        path = []
        actions_path = []
        if deterministic:
            self.in_progress = np.argmax(self.probs_for_selection())
        else:
            self.in_progress = np.random.choice(cfg.action_dim, p=self.probs_for_selection())
        actions_path.append(self.in_progress)
        path.append(self.children_nodes[self.in_progress])
        while None not in path[-1].children_nodes:  # If not leaf!
            if deterministic:
                path[-1].in_progress = np.argmax(path[-1].probs_for_selection())
            else:
                path[-1].in_progress = np.random.choice(cfg.action_dim, p=path[-1].probs_for_selection())
            actions_path.append(path[-1].in_progress)
            path.append(path[-1].children_nodes[path[-1].in_progress])
        if self.verbose:
            print("Select:", [p.node_id for p in path], actions_path)
        return path, actions_path

    def expand(self, samples=1):
        """
        Note: It works if 'self' is a leaf.
        Expanding assigning a state and an initial expected free energy in all
        children nodes of the current leaf!
        """
        dummy_action_onehot = tf.eye(cfg.action_dim, dtype=cfg.np_precision)

        G, pred_state_1, _ = self.model.calculate_G(self.node_state, dummy_action_onehot, average_G_over_N_samples=samples)
        self.total_free_energy -= G.numpy()  # Note: Negative expected free energy to be used as a Q value in RL applications.
        self.exploration_times += 1.0
        for i in range(cfg.action_dim):
            self.children_nodes[i] = Node(s=pred_state_1[i], model=self.model, C=self.C, using_prior_for_exploration=self.using_prior_for_exploration)
        if self.verbose:
            print("Expand:", G.numpy(), self.total_free_energy, self.exploration_times)

    # @tf.function
    def backpropagate(self, path, G):
        if self.verbose:
            print("Back-propagate:", [p.node_id for p in path], G.numpy())
        for i in range(len(path)):
            if path[i].in_progress < 0:
                exit("Back-propagation error: " + str(path) + " " + str(i))
            path[i].total_free_energy[path[i].in_progress] -= G
            path[i].exploration_times[path[i].in_progress] += 1
            path[i].in_progress = -2  # just to remember it's been examined..
            if self.verbose:
                print("Propagating to node", path[i].node_id, "with N:", path[i].exploration_times)

    # @tf.function
    def action_selection(self, deterministic=True):
        path = []
        if deterministic:
            path.append(np.argmax(self.exploration_times))
        else:
            path.append(np.random.choice(cfg.action_dim, p=self.normalization(self.exploration_times)))
        node = self.children_nodes[path[-1]]
        if self.verbose:
            print(len(path), node.node_id)
        while None not in node.children_nodes:
            if deterministic:
                path.append(np.argmax(node.exploration_times))
            else:
                path.append(np.random.choice(cfg.action_dim, p=self.normalization(node.exploration_times)))
            node = node.children_nodes[path[-1]]
            if self.verbose:
                print(len(path), node.node_id)

        # trimmed_path = []
        # i = 0
        # while i < len(path) - 1:
        #     if cfg.action_dim == 4:
        #         if (
        #             (path[i] == 0 and path[i + 1] == 1)
        #             or (path[i] == 1 and path[i + 1] == 0)
        #             or (path[i] == 2 and path[i + 1] == 3)
        #             or (path[i] == 3 and path[i + 1] == 2)
        #         ):
        #             i += 2
        #         else:
        #             trimmed_path.append(path[i])
        #             i += 1
        #     elif cfg.action_dim == 3:
        #         if (path[i] == 1 and path[i + 1] == 2) or (path[i] == 2 and path[i + 1] == 1):
        #             i += 2
        #         else:
        #             trimmed_path.append(path[i])
        #             i += 1
        #     else:
        #         exit("Error: Unknown number of pi_dim " + str(cfg.action_dim))
        # if self.verbose:
        #     print("Action selection:", path, "trimmed path:", trimmed_path)
        return path


def calc_threshold(P, axis):
    return np.max(P, axis=axis) - np.mean(P, axis=axis)


def normalization(x, tau=1):
    """Compute softmax values for each sets of scores in x."""
    return x / x.sum(axis=0)


class MCTS:
    def __init__(self, C=1.0, threshold=0.5, repeats=300, simulation_repeats=1, simulation_depth=1, use_habit=False):
        self.C = C
        self.threshold = threshold
        self.repeats = repeats
        self.simulation_repeats = simulation_repeats
        self.simulation_depth = simulation_depth
        self.use_habit = use_habit
        self.verbose = False
        self.method = "ai"
        self.using_prior_for_exploration = False

    def active_inference_mcts(self, model, obs):
        states_explored = 0
        all_paths = []  # For debugging.
        all_paths_G = []  # For debugging.
        if obs == []:
            return [0], 0, states_explored, all_paths, all_paths_G

        # Calculate current s_t
        qs0_mean, qs0_logvar = model.encoder_net.encode(obs)

        # Important to be the mean here as we repeat it model.pi_dim times!
        root = Node(s=qs0_mean[0], model=model, C=self.C, using_prior_for_exploration=self.using_prior_for_exploration)

        # Habit.
        root.Qpi = model.habitual_net.predict_action(qs0_mean).numpy()[0]

        if self.use_habit:
            if calc_threshold(root.Qpi, axis=0) > self.threshold:
                if self.verbose:
                    print("Decision in phase A Qpi:", root.Qpi, calc_threshold(root.Qpi, axis=0))
                MCTS_choices = root.Qpi
                return [np.random.choice(model.action_dim, p=root.Qpi)], 0, states_explored, all_paths, all_paths_G

        root.expand()

        for repeat in range(self.repeats):

            if calc_threshold(normalization(root.exploration_times), axis=0) > self.threshold:
                MCTS_choices = normalization(root.exploration_times)
                if self.verbose:
                    print(
                        "Decision in phase B",
                        np.round(root.probs_for_selection(), 2),
                        np.round(MCTS_choices, 2),
                        calc_threshold(MCTS_choices, axis=0),
                        "N:",
                        root.exploration_times,
                    )
                final_path = root.action_selection(deterministic=True)
                return final_path, repeat, states_explored, all_paths, all_paths_G

            path, actions_path = root.select(deterministic=True)  # Path[-1] is a leaf node!
            path[-1].expand()
            all_av_G = np.zeros(self.simulation_repeats)
            for sim_repeat in range(self.simulation_repeats):
                states_explored += self.simulation_depth
                all_av_G[sim_repeat], pi0, path[-1].Qpi = model.mcts_step_simulate(path[-1].node_state[0], self.simulation_depth)
            path[-1].backpropagate([root] + path[:-1], all_av_G.mean())
            all_paths.append(actions_path)
            all_paths_G.append(all_av_G.mean())

        final_path = root.action_selection(deterministic=True)
        if self.verbose:
            print(
                "Decision in phase C",
                root.exploration_times,
                len(all_paths),
                np.round(root.Q() - np.min(root.Q()), 2),
                np.round(root.C * root.Qpi * np.sqrt(root.exploration_times.sum()) / (1.0 + root.exploration_times), 2),
                "Qpi:",
                np.round(root.Qpi, 2),
            )
        return final_path, self.repeats, states_explored, all_paths, all_paths_G
