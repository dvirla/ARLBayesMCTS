from copy import deepcopy
from BayesBeta import BayesBeta
from QLearner import QLearner
import numpy as np


class MCTSNode:
    def __init__(self, parent=None):
        self.childs = []
        self.parent = parent
        self.N = 0

    def expand(self):
        pass


def init_dist_params(history):
    thetas = []
    for i in range(2):  # theta for each of the two arms
        d = history.get_arm_dicts(i)
        a_i = d['succ']
        b_i = d['fails']
        theta = BayesBeta(arm=i, a=a_i, b=b_i).sample()
        thetas.append(theta)
    return thetas


def tree_search(Q_M, learning_rate, discount_factor, query_cost, history, max_depth, max_simulations: int = 1):
    assert max_simulations is not None
    qlearner = QLearner(learning_rate, discount_factor, query_cost)
    root = MCTSNode()  # TODO: should be an ongoing tree or new tree each search?
    for _ in range(max_simulations):
        Q_pi = deepcopy(Q_M)
        thetas = init_dist_params(history)
        simulate(history, Q_pi, thetas, max_depth)

    return np.unravel_index(np.argmax(qlearner.Q, axis=None), qlearner.Q.shape)


def simulate(history, Q_pi, thetas, max_depth, curr_d=0):
    if curr_d == max_depth:
        return 0



def rollout():
    pass
