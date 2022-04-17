from copy import deepcopy
from BayesBeta import BayesBeta
from QLearner import QLearner
import numpy as np
from scipy.stats import bernoulli


class MCTSNode:
    def __init__(self, history, parent=None):
        self.childs = []
        self.history = history  # Including action and rewards of this node
        self.parent = parent
        self.N = 0
        self.R = 0

    def expand(self):
        pass


class MCTRee:
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
        probs = np.exp(Q_pi[:, 0])/np.sum(np.exp(Q_pi[:, 0]))  # Choosing only arm without quering
        a_q = np.random.multinomial(1, probs)  # action = flattened array of 0's and 1 in desired action index
        return np.unravel_index(np.argmax(a_q, axis=None), a_q.shape)[0]

    def get_action_uct(self, node):
        # TODO: Implement uct function
        pass

    def tree_search(self, Q_M, history, max_depth,
                    max_simulations: int = 1):
        assert max_simulations is not None
        root = self.nodes[history]  # TODO: should be an ongoing tree or new tree each search?
        for _ in range(max_simulations):
            Q_pi = deepcopy(Q_M)
            P_bernoullis = self.init_dist_params(history)
            R = self.simulate(root, Q_pi, P_bernoullis, max_depth, curr_d=0)

        return np.unravel_index(np.argmax(self.qlearner.Q, axis=None), self.qlearner.Q.shape)

    def simulate(self, node: MCTSNode, Q_pi, P_bernoullis, max_depth, curr_d):
        if curr_d == max_depth:
            return 0
        # Rollout
        if node.N == 0:
            query_ind = 0
            action = self.pi_rollout(Q_pi)
            r = P_bernoullis[action].rvs()  # P(;|a,theta)
            R = r + self.rollout(Q_pi, P_bernoullis, max_depth, curr_d + 1)
            # TODO: update R and current node (<s,h>)?
            return R

        # Simulation
        # TODO: choose i,a using uct
        R = None
        return R

    def rollout(self, Q_pi, P_bernoullis, max_depth, curr_d):
        if curr_d == max_depth:
            return 0
        query_ind = 0
        action = self.pi_rollout(Q_pi)
        r = P_bernoullis[action].rvs()  # P(;|a,theta)
        R = r + self.rollout(Q_pi, P_bernoullis, max_depth, curr_d + 1)
        # TODO: update R and current node (<s,h>)?
        return R