from scipy.stats import beta


class BayesBeta:
    def __init__(self, arm, a=0.5, b=0.5):
        assert arm == 0 or arm == 1
        self.arm = arm
        self.a = a + 0.5
        self.b = b + 0.5
        if a == 0:
            self.a = 0.5
        if b == 0:
            self.b = 0.5

    def pdf(self, x):
        return beta.pdf(x, self.a, self.b)

    def cdf(self, x):
        return beta.cdf(x, self.a, self.b)

    def sample(self):
        return beta.rvs(self.a, self.b)