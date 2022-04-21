from copy import deepcopy
from BayesBeta import BayesBeta
import numpy as np
from scipy.stats import bernoulli
from itertools import product


class MCTSNode:
    def __init__(self, actions_history, parent=None):
        self.children = {}
        self.actions_history = actions_history
        self.parent = parent
        self.N = 0
        # TODO: is 1e-3 ok instead of zero?
        self.N_per_action = np.ones((2, 2)) * 1e-3  # 2 arms, 2 query inds
        self.Q_per_action = np.ones((2, 2)) * 1e-3

    def expand(self):
        for action, query_ind in product((0, 1), (0, 1)):
            new_history = list(self.actions_history)
            new_history.append((action, query_ind))
            new_node = MCTSNode(tuple(new_history), parent=self)
            self.children[(action, query_ind)] = new_node

    def get_argmax(self):
        action, query_ind = np.unravel_index(np.argmax(self.Q_per_action, axis=None), self.Q_per_action.shape)
        return action, query_ind

    def get_action_uct(self, exploration_const):
        uct_matrix = self.Q_per_action + exploration_const * np.sqrt(np.log(self.N) / self.N_per_action)
        return np.unravel_index(np.argmax(uct_matrix, axis=None), uct_matrix.shape)


class MCTree:
    def __init__(self, actions_history: tuple, learning_rate, discount_factor, query_cost, exploration_const):
        """
        :param history: A tuple (in order to be hashable) of triplets (action, query_ind, reward)
        """
        root = MCTSNode(actions_history)
        self.nodes = {actions_history: root}
        self.rewards_history = []
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.Q = np.random.randn(2, 2)  # 2 arms and 2 queries indicators
        self.query_cost = query_cost
        self.exploration_const = exploration_const

    def get_arm_dict(self, arm):
        d = {'succ': 0, 'fails': 0}
        for action, reward in self.rewards_history:
            if action == arm:
                if reward == 1:
                    d['succ'] += 1
                elif reward == 0:
                    d['fails'] += 1
        return d

    def init_dist_params(self):
        P_bernoullis = []
        for i in range(2):  # theta for each of the two arms
            d = self.get_arm_dict(i)
            a_i = d['succ']
            b_i = d['fails']
            theta = BayesBeta(arm=i, a=a_i, b=b_i).sample()
            P_bernoullis.append(bernoulli(theta))
        return P_bernoullis

    @staticmethod
    def pi_rollout(Q_pi):
        """
        :param Q_pi: (2, 2) array actions X query_inds
        :return: action, query_ind indices
        """
        probs = np.exp(Q_pi[:, 0]) / np.sum(np.exp(Q_pi[:, 0]))  # Choosing only arm without quering
        a_q = np.random.multinomial(1, probs)  # action = flattened array of 0's and 1 in desired action index
        return np.unravel_index(np.argmax(a_q, axis=None), a_q.shape)[0]

    def tree_search(self, Q_M, actions_history, max_depth, max_simulations: int = 1):
        root = self.nodes[actions_history]
        for _ in range(max_simulations):
            Q_pi = deepcopy(Q_M)
            P_bernoullis = self.init_dist_params()
            _ = self.simulate(root, Q_pi, P_bernoullis, max_depth, curr_d=0)

        action, query_ind = root.get_argmax()

        return action, query_ind

    def simulate(self, node: MCTSNode, Q_pi, P_bernoullis, max_depth, curr_d):
        if curr_d == max_depth:
            return 0

        # Rollout
        if node.N == 0:
            node.expand()
            # Update MCTree's nodes
            for child in node.children.values():
                self.nodes[child.actions_history] = child

            query_ind = 0
            action = self.pi_rollout(Q_pi)
            r = P_bernoullis[action].rvs()  # P(;|a,theta)
            R = r + self.rollout(Q_pi, P_bernoullis, max_depth, curr_d + 1)  # TODO: multiply by discount factor?

            node.N = 1
            node.N_per_action[action, query_ind] = 1
            node.Q_per_action[action, query_ind] = R
            return R

        # Simulation
        action, query_ind = node.get_action_uct(self.exploration_const)
        r = P_bernoullis[action].rvs()  # P(;|a,theta)
        next_node = node.children[action, query_ind]

        if query_ind == 0:
            R = r + self.simulate(next_node, Q_pi, P_bernoullis, max_depth, curr_d + 1)
            # Update rewards history for bayesian update
            self.rewards_history.append((action, None))
        else:
            R = r + self.simulate(next_node, Q_pi, P_bernoullis, max_depth, curr_d + 1) - self.query_cost
            # Update Q values for rollout policy
            self.q_update(Q_pi, action, query_ind, r)
            # Update rewards history for bayesian update
            self.rewards_history.append((action, r))

        node.N += 1
        node.N_per_action[action, query_ind] += 1
        node.Q_per_action[action, query_ind] += (R - node.Q_per_action[action, query_ind]) / node.N_per_action[action, query_ind]
        return R

    def rollout(self, Q_pi, P_bernoullis, max_depth, curr_d):
        if curr_d == max_depth:
            return 0
        action = self.pi_rollout(Q_pi)
        r = P_bernoullis[action].rvs()  # P(;|a,theta)
        R = r + self.rollout(Q_pi, P_bernoullis, max_depth, curr_d + 1)
        return R

    def q_update(self, Q_pi, action, query_ind, reward):
        assert action in (0, 1)
        assert reward in (0, 1, None)
        old_val = Q_pi[action, query_ind]
        Q_pi[action, query_ind] += self.learning_rate * (reward - old_val - self.query_cost + self.discount_factor * max(Q_pi[action, :]))
        # TODO: need to update according to the original BAMCP papaer
