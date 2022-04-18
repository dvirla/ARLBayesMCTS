from scipy.stats import beta
from History import History


class BayesBeta:
    def __init__(self, arm: int, a=0.5, b=0.5):
        assert arm == 0 or arm == 1
        self.arm = arm
        self.a = a
        self.b = b
        if a == 0:
            self.a = 0.5
        if b == 0:
            self.b = 0.5

    def update_a_b(self, history: History):
        arm_dict = history.get_arm_dicts(self.arm)
        self.a += arm_dict['succ']
        self.b += arm_dict['fails']

    def pdf(self, x):
        return beta.pdf(x, self.a, self.b)

    def cdf(self, x):
        return beta.cdf(x, self.a, self.b)

    def sample(self):
        return beta.rvs(self.a, self.b)