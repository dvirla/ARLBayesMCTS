"""
Library for active Multi-armed bandits

Written by Jan Leike 2016


Conventions:

T: number of times pulled each arm
s: number of successes on each arm
n: horizon
t: current time step
"""

import random
from math import sqrt, ceil, log
import numpy as np
from scipy.stats import beta, norm
from scipy.integrate import dblquad
from tqdm import tqdm


class Bandit(object):
    """
    Bernoulli bandit problem

    mus: list of Bernoulli parameters
    n: horizon
    cost: the query cost
    """

    def __init__(self, mus, n, cost, increase_factor, decrease_factor):
        self.mus = mus
        self.n = n
        self.cost = cost
        self.base_query_cost = cost
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor

    def num_arms(self):
        return len(self.mus)

    def best_mu(self):
        return max(self.mus)

    def best_arm(self):
        return np.argmax(self.mus)

    def pull_arm(self, j):
        return int(self.mus[j] > random.random())

    def cost_fn(self, query_ind, num_queries=1, curr_cost=None):
        if curr_cost is None:
            curr_cost = self.cost
        new_cost = query_ind * curr_cost * self.increase_factor + \
                   (1 - query_ind) * max(curr_cost * self.decrease_factor, self.base_query_cost)
        if num_queries <= 1:
            return new_cost
        else:
            return new_cost + self.cost_fn(query_ind=1, num_queries=num_queries - 1, curr_cost=new_cost)

    def update_cost(self, query_ind):
        self.cost = self.cost_fn(query_ind, num_queries=1)

    def __str__(self):
        return ("{1}-armed active Bernoulli bandit " +
                "with means {0.mus}, horizon {0.n}, and cost {0.cost}").format(
            self, self.num_arms())

    def __eq__(self, other):
        return self.mus == other.mus and self.n == other.n and \
               self.cost == other.cost


def bestArm(T, s):
    """Return the best arm (greedy policy)"""
    k = len(T)
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    m = max(mu_hats)
    return random.choice([i for i in range(k) if mu_hats[i] == m])


def EpsGreedyPolicy(T, s, n, t, d=0.1, c=0.05):
    k = len(T)  # number of arms
    eps_t = c * k / float(d ** 2 * t)  # from the theory, d is the minimum gap in the arms
    #    eps_t = 1/t
    if random.random() > eps_t:
        # greedy policy
        return bestArm(T, s)
    else:
        # pick an action uniformly at random
        return random.randint(0, k - 1)


def ThompsonPolicy(T, s, n, t):
    """policy based on Thompson sampling"""
    mu_hats = beta.rvs(np.add(s, 1), np.subtract(np.add(T, 1), s))
    return np.argmax(mu_hats)


def UCBPolicy(T, s, n, t, c=0.5):
    """policy based on the UCB algorithm"""
    if 0 in T:
        return T.index(0)
    else:
        ucb = np.array(s) / np.array(T) + np.sqrt(c * np.log(t) / np.array(T))
        return max(ucb, key=lambda x: (x, random.random()))


def OCUCBPolicy(T, s, n, t, c=0.5):
    """Policy based on the optimally confident UCB algorithm
    by Lattimore (2015)"""
    if 0 in T:
        return T.index(0)
    else:
        ucb = np.array(s) / np.array(T) + \
              np.sqrt(c * np.log(n / float(t)) / np.array(T))
        return max(ucb, key=lambda x: (x, random.random()))


def BayesUCBPolicy(T, s, n, t):
    """Bayes-UCB policy with quartile 1/t"""
    if 0 in T:
        return T.index(0)
    else:
        a = np.add(s, 1)
        b = np.subtract(np.add(T, 1), s)
        quantiles = beta.isf(1 / float(t), a, b)
        return max(quantiles, key=lambda x: (x, random.random()))


def Arm1Policy(T, s, n, t, cost=0):
    """policy that always pulls arm 1"""
    return 0, False


def RoundRobinPolicy(T, s, n, t, cost=0):
    """policy that alternates between all arms"""
    return t % len(T)


def expectedRegret(T, s, n, t, arm, tol=1e-3):
    """Bayes-expected regret when committing to the arm 'arm'"""
    k = len(T)
    a = np.add(s, 1)
    b = np.subtract(np.add(T, 1), s)

    def f(x, j):
        """integrant for E [ theta_j - theta_arm | j is best arm ]"""
        assert j != arm
        y = x * beta.cdf(x, a[arm], b[arm])
        y -= beta.expect(lambda z: z, (a[arm], b[arm]), lb=0, ub=x, epsrel=tol)
        for s in range(k):
            if s != arm and s != j:
                y *= beta.cdf(x, a[s], b[s])
        return y

    x = 0
    for j in range(k):
        if j != arm:
            x += beta.expect(lambda z: f(z, j), (a[j], b[j]), epsabs=tol)
    return x * (n - t)


def probBestArm(T, s, arm, tol=1e-3):
    """posterior probability that "arm" is the best arm"""
    k = len(T)
    a = np.add(s, 1)
    b = np.subtract(np.add(T, 1), s)

    def f(x):
        y = 1
        for i in range(k):
            if i != arm:
                y *= beta.cdf(x, a[i], b[i])
        return y

    return beta.expect(f, (a[arm], b[arm]), epsabs=tol)


def EntBestArm(T, s, base=2, tol=1e-3):
    """The entropy of the posterior about the best arm"""
    k = len(T)
    x = 0
    for i in range(k):
        p = probBestArm(T, s, i, tol)
        if p > 0:
            x -= p * log(p, base)
    return x


def minExpectedRegret(T, s, n, t, tol=1e-3):
    """Bayes-expected regret when committing to the best arm"""
    return expectedRegret(T, s, n, t, bestArm(T, s), tol)


def FixedQueryPolicy(T, s, n, t, cost, query_for=float('inf'),
                     alg=OCUCBPolicy):
    """use standard bandit algorithm and query the query_for steps"""
    if t <= query_for:
        return alg(T, s, n, t), True
    else:
        return bestArm(T, s), False


def EpsQueryPolicy(T, s, n, t, cost, alg=OCUCBPolicy):
    """use OCUCB and query with probability 1/t"""
    if random.random() < 1 / float(t + 1):
        return alg(T, s, n, t), True
    else:
        return bestArm(T, s), False


def ExpQueryPolicy(T, s, n, t, cost, alg=OCUCBPolicy):
    """query whenever doing an exploration action"""
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    j = alg(T, s, n, t)
    if mu_hats[j] == max(mu_hats):
        return j, T[j] < max(T)
    else:
        return j, True


def querySteps(T, s):
    """
    the number of steps you expect to need to
    bring the two arms with the highest means together
    """
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    m = max(mu_hats)
    j = np.argmax(mu_hats)
    gap_steps = np.add(T, 1) * np.subtract(m, mu_hats)
    del list(gap_steps)[j]
    return 2 * ceil(min(gap_steps) + 0.01) ** 2


def querySteps4(T, s):
    """
    the number of steps you expect to need to
    bring the two arms with the highest means together
    """
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    m = max(mu_hats)
    j = np.argmax(mu_hats)
    gap_steps = np.add(T, 1) * np.subtract(m, mu_hats)
    del list(gap_steps)[j]
    return max(1, ceil(2 * min(gap_steps) ** 2))


def querySteps3(T, s):
    """
    the number of steps you expect to need to
    bring the two arms with the highest means together
    note: this code is horribly inefficient, but that should't matter ^_^
    """

    def mu_hats(T, s):
        return np.add(s, 1) / np.add(T, 2).astype(float)

    mu_hats_ = list(mu_hats(T, s))
    j = np.argmax(mu_hats_)
    del mu_hats_[j]
    i = np.argmax(mu_hats_)
    mu_hats_ = mu_hats(T, s)
    i += 1 if i >= j else 0
    assert i != j
    z = int(ceil(sqrt(2 * (min(T[i], T[j]) + 2))))  # upper bound
    l = [(xi, xj) for xi in range(z + 1) for xj in range(z + 1)]
    l.sort(key=lambda x: x[1] ** 2 + x[2] ** 2)
    for (xi, xj) in l:
        if mu_hats(T[i] + 2 * xi ** 2, s[i] + xi + 2 * xi ** 2 * mu_hats_[i]) >= \
                mu_hats(T[j] + 2 * xj ** 2, s[j] - xj + 2 * xj ** 2 * mu_hats_[j]):
            return 2 * (xi ** 2 + xj ** 2)
    return float('inf')


def DMED(T, s, n, t):
    """DMED bandit policy"""
    k = len(T)
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    m = max(mu_hats)
    if 0 in T:
        arms = range(k)  # prevent log(0) errors
    else:
        Jp = np.multiply(T, np.subtract(m, mu_hats)) - np.log(n) + np.log(T)
        arms = []
        for i in range(k):
            if Jp[i] <= 0:
                arms.append(i)
    return random.choice(arms)


def DMEDPolicy(T, s, n, t, cost):
    # Honda, Junya, and Akimichi Takemura.
    # An Asymptotically Optimal Bandit Algorithm for Bounded Support Models.
    # COLT 2010.
    # here we use (theta - theta^*) instead of KL(B(theta), B(theta^*))
    k = len(T)
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    m = max(mu_hats)
    if 0 in T:
        arms = range(k)  # prevent log(0) errors
    else:
        Jp = np.multiply(T, np.subtract(m, mu_hats)) - np.log(n) + np.log(T)
        arms = []
        for i in range(k):
            if Jp[i] <= 0:
                arms.append(i)
    return random.choice(arms), len(arms) > 1


def parameterizedRegretQuery(T, s, n, t, cost, banditpolicy=DMED, alpha=0.35):
    """
    execute bandit policy until
    cost to move posterior < alpha * expected regret
    with parameter alpha \in (0, 1)
    """
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    best_arm = np.argmax(mu_hats)
    query_steps = querySteps4(T, s)
    if t + query_steps >= n:
        # instant commitment because the time frame is too long
        return best_arm, False
    query = cost * query_steps < alpha * minExpectedRegret(T, s, n, t)
    if query:
        return banditpolicy(T, s, n, t), True
    else:
        return best_arm, False


def parameterizedRegretQueryChangingCost(T, s, n, t, cost_fn, banditpolicy=DMED, alpha=0.35):
    """
    execute bandit policy until
    cost to move posterior < alpha * expected regret
    with parameter alpha \in (0, 1)
    """
    # mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    # best_arm = np.argmax(mu_hats)
    best_arm = bestArm(T, s)
    query_steps = querySteps4(T, s)
    if t + query_steps >= n:
        # instant commitment because the time frame is too long
        return best_arm, 0
    # query = np.sum(costs) < alpha * minExpectedRegret(T, s, n, t)

    cum_cost = cost_fn(query_ind=1, num_queries=query_steps)
    query = cum_cost < alpha * expectedRegret(T, s, n, t, best_arm, tol=1e-3)
    if query:
        return banditpolicy(T, s, n, t), 1
    else:
        return best_arm, 0


def minGap(T, s):
    """Return the minimal gap between two distince arms"""
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    m = max(mu_hats)
    j = np.argmax(mu_hats)
    del mu_hats[j]
    return m - max(mu_hats)


def knowledgeGradient(T, s, n, t, queries, arm):
    """Estimate the knowledge gradient"""
    if queries == 0:
        return 0
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    m = max(mu_hats)
    var_post = (np.add(s, 1) * np.add(np.subtract(T, s), 1)) / \
               (np.add(T, 2) ** 2 * np.add(T, 3)).astype(float)
    var_data = mu_hats * np.subtract(1, mu_hats)

    def std_cond_change(l, i):
        return sqrt(var_post[i] - 1 / (1 / var_post[i] + l / var_data[i]))

    def f(x):
        return x * norm.cdf(x) + norm.pdf(x)

    return (std_cond_change(queries, arm) *
            f((mu_hats[arm] - m) / std_cond_change(queries, arm)))


def knowledgeGradientPolicy(T, s, n, t, cost):
    """See Section 5.2 in Warren Powell and Ilya Ryzhov.
    Optimal Learning. John Wiley & Sons, 2012."""
    k = len(T)
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    m = 0
    j = None
    for i in range(k):
        if i != np.argmax(mu_hats):
            m_ = max([knowledgeGradient(T, s, n, t, l, i) * (n - t) - cost * l
                      for l in range(int(sqrt(n)))])
            if m_ > m:
                m = m_
                j = i
    if m > 0 and j is not None:
        return j, True
    else:
        return bestArm(T, s), False


def knowledgeGradientChangingCostPolicy(T, s, n, t, cost_fn):
    """See Section 5.2 in Warren Powell and Ilya Ryzhov.
    Optimal Learning. John Wiley & Sons, 2012."""
    k = len(T)
    # mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    best_arm = bestArm(T, s)
    m = 0
    j = None
    for i in range(k):
        if i != best_arm:
            m_ = max([knowledgeGradient(T, s, n, t, l, i)*(n-t) - cost_fn(query_ind=l>0, num_queries=l) for l in range(int(sqrt(n)))])
            if m_ > m:
                m = m_
                j = i
    if m > 0 and j is not None:
        return j, 1
    else:
        return best_arm, 0


def TorsPolicy(T, s, n, t, cost, c=0.5):
    """Tor's policy 2016-11-15, c is the OCUCB parameter"""
    if 0 in T:
        return T.index(0), True
    k = len(T)
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    cb = np.sqrt(c * np.log(n / float(t)) / np.array(T))
    ucb = np.array(s) / np.array(T) + cb
    lcb = np.array(s) / np.array(T) - cb
    #    print(lcb, ucb)
    arms = [i for i in range(k) if ucb[i] >= max(lcb)]  # non-eliminated arms
    assert len(arms) > 0
    #    print(arms)
    delta_hat = max(max(mu_hats) - mu_hats[arms])
    #    delta_hat = max(ucb[arms]) - min(lcb[arms])  # estimated gap
    #    print(delta_hat)
    if cost <= (n - t) * delta_hat ** 3 / 2 / k / log(n):
        return random.choice(arms), True
    else:
        return bestArm(T, s), False


def activeBanditPolicy3(T, s, n, t, cost, tol=0.05):
    k = len(T)
    j = bestArm(T, s)
    a = np.add(s, 1)
    b = np.subtract(np.add(T, 1), s)
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)

    def _fun(theta_j, theta_i, i):
        if theta_i > mu_hats[j]:
            gain = (theta_i - theta_j) * (n - t)
            steps = (mu_hats[j] * (T[i] + 2) - s[i] - 1) / (theta_i - mu_hats[j])
            return max(theta_i - theta_j - cost, gain - cost * steps)
        else:
            return theta_i - theta_j - cost

    def fun(theta_j, theta_i, i):
        x = _fun(theta_j, theta_i, i)
        return x * beta.pdf(theta_j, a[j], b[j]) * beta.pdf(theta_i, a[i], b[i])

    def _funj(theta_j, theta_i, i):
        if mu_hats[i] > theta_j:
            gain = (theta_i - theta_j) * (n - t)
            steps = (mu_hats[i] * (T[j] + 2) - s[j] - 1) / (mu_hats[i] - theta_j)
            return max(theta_j - theta_i - cost, gain - cost * steps)
        else:
            return theta_j - theta_i - cost

    def funj(theta_j, theta_i, i):
        x = _funj(theta_j, theta_i, i)
        return x * beta.pdf(theta_j, a[j], b[j]) * beta.pdf(theta_i, a[i], b[i])

    # Figure out which arm to query
    VoI = [-float('inf')] * k
    for i in range(k):
        if tol < 1e-2:
            VoI[i], err = dblquad(fun if i != j else funj, 0, 1, lambda x: 0,
                                  lambda x: 1, args=(i,),
                                  epsabs=tol, epsrel=tol)
        else:
            num = int(tol ** (-2))  # number of MC samples
            f = _fun if i != j else _funj
            VoI[i] = np.average([f(beta.rvs(a[j], b[j]),
                                   beta.rvs(a[i], b[i]), i)
                                 for _ in range(num)])

    # If the VoI is positive, query, otherwise commit
    print(mu_hats)
    print(VoI)
    if max(VoI) > 0:
        return np.argmax(VoI), True
    else:
        return j, False


def playBernoulli(bandit, policy, assume_commitment=True, **kwargs):
    """Play a game of bernoulli arms"""
    k = bandit.num_arms()
    T = [0] * k  # number of times pulled each arm
    s = [0] * k  # number of successes on each arm
    # regret = 0  # cumulative undiscounted regret
    # cregret = [0]
    query = True
    last_query_step = 0

    timesteps = list(range(bandit.n))
    query_costs = []
    chosen_arms = []
    query_inds = []
    rewards = []

    for t in range(bandit.n):
        # old_query = query
        j, query = policy(T, s, bandit.n, t, bandit.cost_fn, **kwargs)
        r = bandit.pull_arm(j)

        bandit.update_cost(query)
        rewards.append(r)
        query_costs.append(bandit.cost)
        chosen_arms.append(j)
        query_inds.append(query)

        # if query:
        #     T[j] += 1
        #     s[j] += r
        #     regret += bandit.cost
        #     last_query_step = t
    #     elif old_query:
    #         #            print('stopping at t = %d. Commited to arm %d' % (t, j + 1))
    #         if assume_commitment:
    #             d = bandit.best_mu() - bandit.mus[j]
    #             cregret.extend([regret + d * i
    #                             for i in range(1, bandit.n - t + 1)])
    #             regret += d * (bandit.n - t)
    #             break
    #
    #     regret += bandit.best_mu() - bandit.mus[j]
    #     cregret.append(regret)
    # #    print('regret = %.2f' % regret)

    # return {'cregret': cregret, 'last query step': last_query_step}
    return timesteps, rewards, query_costs, chosen_arms, query_inds


def runExperiment(mus, n, cost, increase_factor, decrease_factor, policy, N, assume_commitment=False, progressbar=False,
                  **kwargs):
    # if progressbar:
    # from IPython.html.widgets import FloatProgressWidget
    # from IPython.display import display
    # f = FloatProgressWidget(min=0, max=N)
    # display(f)
    runs = []
    timesteps = []
    arms_thetas = [mus] * n * N
    base_query_cost = [cost] * n * N
    query_costs = []
    horizon = [n] * n * N
    chosen_arms = []
    query_inds = []
    rewards = []
    # results = []
    for i in tqdm(range(N)):
        #        random.shuffle(mus)
        np.random.seed(i)
        random.seed(i)

        bandit = Bandit(mus, n, cost, increase_factor, decrease_factor)
        temp_timesteps, temp_rewards, temp_query_costs, temp_chosen_arms, temp_query_inds = playBernoulli(bandit,
                                                                                                          policy,
                                                                                                          assume_commitment,
                                                                                                          **kwargs)
        runs.extend([i] * n)
        timesteps.extend(temp_timesteps)
        query_costs.extend(temp_query_costs)
        chosen_arms.extend(temp_chosen_arms)
        query_inds.extend(temp_query_inds)
        rewards.extend(temp_rewards)
        # if progressbar:
        #     f.value = i
    # return bandit, policy, results
    return runs, timesteps, arms_thetas, base_query_cost, query_costs, horizon, chosen_arms, query_inds, rewards

# example = Bandit([0.6, 0.5, 0.4, 0.4], 10000, 2)
