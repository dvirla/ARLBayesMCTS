from copy import deepcopy
from BayesBeta import BayesBeta
import numpy as np
from scipy.stats import bernoulli
from itertools import product

np.seterr(divide='ignore', invalid='ignore')


class MCTSNode:
    def __init__(self, actions_history, parent=None):
        self.children = {}
        self.actions_history = actions_history
        self.parent = parent
        self.N = 0
        self.N_per_action = np.zeros((2, 2))  # 2 arms, 2 query inds
        self.Q_per_action = np.zeros((2, 2))

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
        uct_matrix[np.isnan(uct_matrix)] = np.inf
        return np.unravel_index(np.argmax(uct_matrix, axis=None), uct_matrix.shape)


class MCTree:
    def __init__(self, actions_history, learning_rate, discount_factor, base_query_cost, increase_factor,
                 decrease_factor, exploration_const):
        """
        :param history: A tuple (in order to be hashable) of triplets (action, query_ind, reward)
        """
        self.root = MCTSNode(actions_history)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.Q = np.random.randn(2, 2)  # 2 arms and 2 queries indicators
        self.base_query_cost = base_query_cost
        self.query_cost = base_query_cost
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.exploration_const = exploration_const
        self.arms_dicts = [{'succ': 0, 'fails': 0}, {'succ': 0, 'fails': 0}]

    def get_arm_dict(self, arm):
        d = self.arms_dicts[arm]
        return d

    def update_arm_dict(self, arm, r):
        """
        :param arm: which arm was chosen
        :param r: only calling the function when querying so r in (0, 1)
        """
        assert r == 1 or r == 0
        if r == 1:
            self.arms_dicts[arm]['succ'] += 1
        else:
            self.arms_dicts[arm]['fails'] += 1

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

    def get_query_cost(self, query_ind, update=False, query_cost=None):
        if query_cost is None:
            query_cost = self.query_cost
        query_cost = query_ind * query_cost * self.increase_factor + \
                     (1 - query_ind) * max(query_cost * self.decrease_factor, self.base_query_cost)
        if update:
            self.query_cost = query_cost
        return query_cost

    def tree_search(self, Q_M, max_depth, root, max_simulations=1):
        if root is None:
            root = self.root
        for _ in range(max_simulations):
            Q_pi = deepcopy(Q_M)
            P_bernoullis = self.init_dist_params()
            _ = self.simulate(root, Q_pi, P_bernoullis, max_depth, curr_d=0)

        action, query_ind = root.get_argmax()
        _ = self.get_query_cost(query_ind, update=True)

        return action, query_ind, root.children[(action, query_ind)]

    def simulate(self, node, Q_pi, P_bernoullis, max_depth, curr_d, query_cost=None):
        if curr_d == max_depth:
            return 0

        if query_cost is None:
            query_cost = self.query_cost

        # Rollout
        if node.N == 0:
            node.expand()
            query_ind = 0
            action = self.pi_rollout(Q_pi)
            r = P_bernoullis[action].rvs()  # P(;|a,theta)
            R = r + self.rollout(Q_pi, P_bernoullis, max_depth, curr_d + 1)

            node.N = 1
            node.N_per_action[action, query_ind] = 1
            node.Q_per_action[action, query_ind] = R
            return R

        # Simulation
        action, query_ind = node.get_action_uct(self.exploration_const)
        r = P_bernoullis[action].rvs()  # P(;|a,theta)
        next_node = node.children[(action, query_ind)]

        query_cost = self.get_query_cost(query_ind, False, query_cost)

        if query_ind == 0:
            R = r + self.simulate(next_node, Q_pi, P_bernoullis, max_depth, curr_d + 1, query_cost)
        else:
            R = r + self.simulate(next_node, Q_pi, P_bernoullis, max_depth, curr_d + 1, query_cost) - query_cost
            # Update Q values for rollout policy
            self.q_update(Q_pi, action, query_ind, r)
            # Update rewards history for bayesian update
            self.update_arm_dict(action, r)

        node.N += 1
        node.N_per_action[action, query_ind] += 1
        node.Q_per_action[action, query_ind] += (R - node.Q_per_action[action, query_ind]) / node.N_per_action[
            action, query_ind]
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
        Q_pi[action, query_ind] += self.learning_rate * (
                reward - old_val - self.query_cost + self.discount_factor * max(Q_pi[action, :]))
