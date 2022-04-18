from copy import deepcopy
from BayesBeta import BayesBeta
from QLearner import QLearner
import numpy as np
from scipy.stats import bernoulli
from History import History
from itertools import product


class MCTSNode:
    def __init__(self, history: History, parent=None):
        self.children = {}
        self.history = history  # Including action and rewards of this node
        self.parent = parent
        self.N = 0
        self.R = 0
        # TODO: is it ones? zeros? other value?
        self.N_per_action = np.ones((2, 2))  # 2 arms, 2 query inds

    def expand(self, action, query_ind, r):
        if self.children.get((action, query_ind)) is None:
            if query_ind:
                observed_r = r
            else:
                observed_r = None
            new_history = self.history.update(action, observed_r)
            new_node = MCTSNode(new_history, parent=self)
            self.children[(action, query_ind)] = new_node
        else:
            new_node = self.children[(action, query_ind)]

        return new_node


class MCTree:
    def __init__(self, history, learning_rate, discount_factor, query_cost, exploration_const):
        self.root = MCTSNode(history)
        self.nodes = {history: self.root}
        self.qlearner = QLearner(learning_rate, discount_factor, query_cost)
        self.exploration_const = exploration_const

    @staticmethod
    def init_dist_params(history):
        P_bernoullis = []
        for i in range(2):  # theta for each of the two arms
            d = history.get_arm_dicts(i)
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

    def get_action_uct(self, node):
        Q = self.qlearner.Q.copy()
        N_per_action = node.N_per_action.copy()
        uct_matrix = Q + self.exploration_const * np.sqrt(np.log(node.N) / N_per_action)
        return np.unravel_index(np.argmax(uct_matrix, axis=None), uct_matrix.shape)

    def tree_search(self, Q_M, history, max_depth, max_simulations: int = 1):
        assert max_simulations is not None
        root = self.nodes[history]  # TODO: should be an ongoing tree or new tree each search?
        for _ in range(max_simulations):
            Q_pi = deepcopy(Q_M)
            P_bernoullis = self.init_dist_params(history)
            R = self.simulate(root, Q_pi, P_bernoullis, max_depth, curr_d=0)

        action, query_ind = np.unravel_index(np.argmax(self.qlearner.Q, axis=None), self.qlearner.Q.shape)

        # Update MCTree's nodes
        if query_ind:
            for t_r in (0, 1):
                t_h = root.history.update(action, t_r)
                if self.nodes.get(t_h, None) is None:
                    self.nodes[t_h] = MCTSNode(t_h, parent=root)
        else:
            t_h = root.history.update(action, reward=None)
            self.nodes[t_h] = MCTSNode(t_h, parent=root)

        return action, query_ind

    def simulate(self, node: MCTSNode, Q_pi, P_bernoullis, max_depth, curr_d):
        if curr_d == max_depth:
            return 0

        # Rollout
        if node.N == 0:
            query_ind = 0
            action = self.pi_rollout(Q_pi)
            r = P_bernoullis[action].rvs()  # P(;|a,theta)
            R = r + self.rollout(Q_pi, P_bernoullis, max_depth, curr_d + 1)

            node.N += 1
            node.R += R
            node.N_per_action[action, query_ind] += 1
            return R

        # Simulation
        action, query_ind = self.get_action_uct(node)
        r = P_bernoullis[action].rvs()  # P(;|a,theta)
        next_node = node.expand(action, query_ind, r)

        # Update MCTree's nodes
        if self.nodes.get(next_node.history, None) is None:
            self.nodes[next_node.history] = next_node

        if query_ind == 0:
            R = r + self.simulate(next_node, Q_pi, P_bernoullis, max_depth, curr_d + 1)
        else:
            R = r + self.simulate(next_node, Q_pi, P_bernoullis, max_depth, curr_d + 1) - self.qlearner.query_cost
            self.qlearner.q_update(action, query_ind,
                                   r)  # TODO: Doesn't it make more sense to update Q values before doing further simulations?

        node.N += 1
        node.R += R
        node.N_per_action[action, query_ind] += 1
        return R

    def rollout(self, Q_pi, P_bernoullis, max_depth, curr_d):
        if curr_d == max_depth:
            return 0
        action = self.pi_rollout(Q_pi)
        r = P_bernoullis[action].rvs()  # P(;|a,theta)
        R = r + self.rollout(Q_pi, P_bernoullis, max_depth, curr_d + 1)
        return R
