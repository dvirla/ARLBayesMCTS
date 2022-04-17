import numpy as np


class QLearner:
    def __init__(self, learning_rate, discount_factor, query_cost):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.Q = np.random.randn(2, 2)  # 2 arms and 2 queries indicators
        # self.reward_to_idx = {0: 0, 1: 1, None: 2}
        self.query_cost = query_cost

    def q_update(self, action, query_ind, reward):
        assert action in (0, 1)
        assert reward in (0, 1, None)
        # reward_idx = self.reward_to_idx[reward]
        old_val = self.Q[action, query_ind]
        self.Q[action, query_ind] += self.learning_rate * (reward - old_val - self.query_cost + self.discount_factor * max(self.Q[action, :]))