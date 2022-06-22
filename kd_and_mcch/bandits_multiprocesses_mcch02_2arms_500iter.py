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
import concurrent.futures
import pandas as pd
import time
import json
import csv
from datetime import datetime

class Bandit(object):
    """
    Bernoulli bandit problem

    mus: list of Bernoulli parameters
    n: horizon
    cost: the query cost
    """

    def __init__(self, mus, n, cost, increase_by, decrease_by):
        self.mus = mus
        self.n = n
        self.cost = cost
        self.increase_by = increase_by
        self.decrease_by = decrease_by
        self.decrease_by_rounded = round(decrease_by, 2)

    def num_arms(self):
        return len(self.mus)

    def best_mu(self):
        return max(self.mus)

    def best_arm(self):
        return np.argmax(self.mus)

    def pull_arm(self, j):
        return int(self.mus[j] > random.random())

    def __str__(self):
        return ("{1}-armed active Bernoulli bandit " +
                "with means {0.mus}, horizon {0.n}, cost {0.cost}, increase by {0.increase_by}, and decrease by {0.decrease_by_rounded}").format(
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

# def dynamicCostFunction (c_b, inc, dec, c_h, q_h, t):
#     if t==0:
#         c_t = c_b
#     else:
#         if q_h[t-1] == 0:
#             c_t = c_h[t-1]
#         else:
#             c_t = q_h[t]*(c_h[t-1]*inc) + (1-q_h[t]) * max((c_h[t-1]*dec), c_b)
#     return c_t
#
# def naiveCumCost (c_b, c_c,q_h, inc,steps, t):
#     cum_cost_arr = []
#     x = np.ones(steps)
#     if t==0:
#         cum_cost_arr.append(c_b)
#     else:
#         if (q_h[t-1] == 1):
#             cum_cost_arr.append(c_c[t-1]*inc)
#         else:
#             cum_cost_arr.append(c_c[t - 1])
#
#     for i in range(steps-1):
#         cum_cost_arr.append(cum_cost_arr[i]*inc)
#
#     return np.sum(cum_cost_arr)


""" 
    dynamicCostFunction: Calculate the current cost considering the query and cost history. 

    Input: 
    c_b: basic cost
    inc: increase factor
    dec: decrease factor
    c_h: cost history
    q_h: query history
    t: time step. 
    
    Output: 
    c_t: current query cost. 
         
 """
def dynamicCostFunction (c_b, inc, dec, c_h, q_h, t):
    if t==0:
        c_t = c_b
    else:
        if q_h[t-1] == 0:
            c_t = c_h[t-1]
        else:
            c_t = q_h[t]* min((c_h[t-1]*inc), 4) + (1-q_h[t]) * max((c_h[t-1]*dec), c_b)
    return c_t

"""
    naiveCumCost: Calculate the cost for querying number of steps consecutively.
    
    Input:
    c_b: basic cost
    inc: increase factor
    c_c: current query cost
    q_h: query history
    steps: number of timesteps of querying. 
    t: time step 
    
    Output: total cost of querying 'steps' timesteps consecutively.
    
"""
def naiveCumCost (c_b, c_c, q_h, inc, steps, t):
    cum_cost_arr = []
    x = np.ones(steps)
    if t==0:
        cum_cost_arr.append(c_b)
    else:
        if (q_h[t-1] == 1):
            cum_cost_arr.append( min(c_c[t-1]*inc, 4) )
        else:
            cum_cost_arr.append(c_c[t - 1])

    for i in range(steps-1):
        cum_cost_arr.append( min(cum_cost_arr[i]*inc, 4) )

    return np.sum(cum_cost_arr)

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



def minExpectedRegret(T, s, n, t, tol=1e-3):
    """Bayes-expected regret when committing to the best arm"""
    return expectedRegret(T, s, n, t, bestArm(T, s), tol)



def querySteps4(T, s):
    """
    the number of steps you expect to need to
    bring the two arms with the highest means together
    """
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    m = max(mu_hats)
    j = bestArm(T,s)
    # j = np.argmax(mu_hats)
    gap_steps = np.add(T, 1) * np.subtract(m, mu_hats)
    gap_steps_list =list(gap_steps)
    del gap_steps_list[j]
    return max(1, ceil(2*min(gap_steps_list)**2))



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

#TODO: notice if want to use chnge the cost to dynamic cost
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


def parameterizedRegretQuery(T, s, bandit, t, c_c,q_h, banditpolicy=DMED, alpha=0.35):
    """
    execute bandit policy until
    cost to move posterior < alpha * expected regret
    with parameter alpha \in (0, 1)
    """
    n = bandit.n
    inc = bandit.increase_by
    c_b = bandit.cost
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    # best_arm = np.argmax(mu_hats)
    best_arm = bestArm(T,s)
    query_steps = querySteps4(T, s)
    # print("query steps", query_steps)

    if t + query_steps >= n:
        # instant commitment because the time frame is too long
        return best_arm, False
    cum_cost = naiveCumCost(c_b, c_c, q_h, inc, query_steps, t)
    # print("cum cost", cum_cost)

    # query = cum_cost < alpha * minExpectedRegret(T, s, n, t)

    query = cum_cost < alpha * expectedRegret(T, s, n, t, best_arm, tol = 1e-3)

    if query:
        return banditpolicy(T, s, n, t), True
    else:
        return best_arm, False


def knowledgeGradient(T, s, n, t, queries, arm):
    """Estimate the knowledge gradient, based on Optimal Learning book cp. 5"""
    if queries == 0:
        return 0
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float) #identical to tetha_n
    m = max(mu_hats)
    # our prior beliefe equivalent to beta_x^n equation 5.7
    var_post = (np.add(s, 1) * np.add(np.subtract(T, s), 1)) / \
               (np.add(T, 2)**2 * np.add(T, 3)).astype(float)
    var_data = mu_hats * np.subtract(1, mu_hats) #caculate beta_W
    #caculate the conditional std after l steps change
    def std_cond_change(l, i):
        return sqrt(var_post[i] - 1/(1/var_post[i] + l/var_data[i]))
    #equtaion 5.9
    def f(x):
        return x*norm.cdf(x) + norm.pdf(x)

    return (std_cond_change(queries, arm) *
            f((mu_hats[arm] - m) / std_cond_change(queries, arm))) #equation 5.8


def knowledgeGradientPolicy(T, s, bandit, t, c_c, q_h):
    """See Section 5.2 in Warren Powell and Ilya Ryzhov.
    Optimal Learning. John Wiley & Sons, 2012."""
    n = bandit.n
    k = len(T)
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    m = 0
    j = None
    best_arm = bestArm(T,s)
    # if (np.argmax(mu_hats) != (bestArm(T,s))):
    #     print("NOT EQUAL")
        # print("argmax is", np.argmax(mu_hats))
        # print("bestArm is", bestArm(T,s))
    for i in range(k):
        if i != best_arm:
        # if i != np.argmax(mu_hats):
            # m_ = []
            # for l in range(int(sqrt(n))):
            #     KG = knowledgeGradient(T, s, n, t, l, i)*(n-t)
            #     cumCost = naiveCumCost(bandit.cost, c_c, q_h, bandit.increase_by, l, t)
            #     m_.append(KG - cumCost)
            # m_max = max(m_)
            m_ = max([knowledgeGradient(T, s, n, t, l, i)*(n-t) - naiveCumCost(bandit.cost, c_c, q_h, bandit.increase_by, l, t)
                      for l in range(int(sqrt(n)))])
            # if (m_>1):
            #     print (m_)
            # m_ = max([knowledgeGradient(T, s, n, t, l, i) * (n - t) - cost*l
            #           for l in range(int(sqrt(n)))]
            if m_ > m:
                m = m_
                j = i
    if m > 0 and j is not None:
        return j, True
    else:
        return best_arm, False


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
    if cost <= (n - t)*delta_hat**3 / 2 / k / log(n):
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
            steps = (mu_hats[j]*(T[i] + 2) - s[i] - 1)/(theta_i - mu_hats[j])
            return max(theta_i - theta_j - cost, gain - cost*steps)
        else:
            return theta_i - theta_j - cost

    def fun(theta_j, theta_i, i):
        x = _fun(theta_j, theta_i, i)
        return x*beta.pdf(theta_j, a[j], b[j])*beta.pdf(theta_i, a[i], b[i])

    def _funj(theta_j, theta_i, i):
        if mu_hats[i] > theta_j:
            gain = (theta_i - theta_j) * (n - t)
            steps = (mu_hats[i]*(T[j] + 2) - s[j] - 1)/(mu_hats[i] - theta_j)
            return max(theta_j - theta_i - cost, gain - cost*steps)
        else:
            return theta_j - theta_i - cost

    def funj(theta_j, theta_i, i):
        x = _funj(theta_j, theta_i, i)
        return x*beta.pdf(theta_j, a[j], b[j])*beta.pdf(theta_i, a[i], b[i])

    # Figure out which arm to query
    VoI = [-float('inf')] * k
    for i in range(k):
        if tol < 1e-2:
            VoI[i], err = dblquad(fun if i != j else funj, 0, 1, lambda x: 0,
                                  lambda x: 1, args=(i,),
                                  epsabs=tol, epsrel=tol)
        else:
            num = int(tol**(-2))  # number of MC samples
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


def playBernoulli(run_n, df, writer, sum_writer, bandit, policy, assume_commitment=True, **kwargs):
    """Play a game of bernoulli arms"""
    policy_name = str(policy.__name__ if policy.__name__ != "parameterizedRegretQuery" else "MCCH")
    details = '{}'.format(bandit) + ', ' + str(policy.__name__)
    if (policy_name == 'MCCH'):
        alpha_MCCH = kwargs['alpha']
    k = bandit.num_arms()
    T = [0]*k  # number of times pulled each arm
    s = [0]*k  # number of successes on each arm
    regret = 0  # cumulative undiscounted regret
    cregret = [0]
    query = True
    last_query_step = 0
    q_h = []
    array_cost = []
    arm_counts = np.zeros(bandit.num_arms())
    # committed_arm = 100
    # data = []
    for t in range(bandit.n):
        # line = []
        # line.append(run_n, t)
        old_query = query
        j, query = policy(T, s, bandit, t, array_cost, q_h, **kwargs)
        q_h.append(1 if query else 0)
        # line.append(query)
        arm_counts[j] += 1
        r = bandit.pull_arm(j)
        if query:
            T[j] += 1
            s[j] += r
            array_cost.append(dynamicCostFunction(bandit.cost, bandit.increase_by, bandit.decrease_by, array_cost, q_h, t))
            regret += array_cost[t]
            last_query_step = t
        elif (t-last_query_step) <= 10:
            array_cost.append(dynamicCostFunction(bandit.cost, bandit.increase_by, bandit.decrease_by, array_cost, q_h, t))
        else:
            # print("last else t- last query is %d" %(t-last_query_step))
            if assume_commitment:
                remaining_steps = bandit.n - t - 1
                # committed_arm = j
                # print('stopping at t = %d. Commited to arm %d' % (t, j + 1))
                d = bandit.best_mu() - bandit.mus[j]
                cregret.extend([regret + d*i
                                for i in range(1, remaining_steps +2)])
                # print("len of query hist%d" %len(q_h))
                q_h.extend([0] * remaining_steps)
                arm_counts[j] += remaining_steps
                for i in range (remaining_steps+1):
                    regret += d
                    # cregret = d*i
                    # row = [run_n, i+t, policy_name, bandit.mus, bandit.cost, bandit.increase_by, bandit.n, regret, j, 1 if query else 0, last_query_step, sum(q_h)]
                    # writer.writerow(row)
                    if i == remaining_steps:
                        row = [run_n, i + t, policy_name, bandit.mus, bandit.cost, bandit.increase_by, bandit.n, regret,
                               j, 1 if query else 0, last_query_step, sum(q_h)]
                        writer.writerow(row)
                        writer_sum.writerow(row)
                break
            else:
                array_cost.append(dynamicCostFunction(bandit.cost, bandit.increase_by, bandit.decrease_by, array_cost, q_h, t))

        regret += bandit.best_mu() - bandit.mus[j]
        # print (t)
        cregret.append(regret)
        row = [run_n, t, policy_name , bandit.mus, bandit.cost, bandit.increase_by, bandit.n, regret, j, 1 if query else 0, last_query_step, sum(q_h)]
        writer.writerow(row)
        if (t==(bandit.n -1)):
            # row.append()
           sum_writer.writerow(row)

        # row = {'run': run_n, 'iter': t, 'policy': policy_name ,'mus': bandit.mus, 'init_cost':bandit.cost, 'increase_by': bandit.increase_by, 'horizon': bandit.n, 'regret': regret, 'arm':j, 'query_history':query, 'last_query_step': last_query_step}
        # df = df.append(row, ignore_index=True)

        # print(array_cost)
        # print(q_h)

#    print('regret = %.2f' % regret)
#     print(type(committed_arm))
#     print("###")
#     print(type(last_query_step))
#     data = {'run': [run_n]*bandit.n, 'iter': [i for i in range(bandit.n)], 'cregret': cregret, 'last query step': last_query_step, 'committed to arm': int(j), 'query history': q_h, 'arm count': arm_counts.tolist(), 'details':details}
    return {'cregret': cregret, 'last query step': last_query_step, 'committed to arm': int(j), 'query history': q_h, 'arm count': arm_counts.tolist(), 'details':details}

    # return df
    # return df

"runExperiment: multiprocess - does not work"
def runExperiment(mus, n, cost, inc, dec, policy, N, df, w, sum_w, progressbar=False, multiprocess =True, **kwargs):
    # if progressbar:
    #     from IPython.html.widgets import FloatProgressWidget
    #     from IPython.display import display
    #     f = FloatProgressWidget(min=0, max=N)
    #     display(f)
    bandit = Bandit(mus, n, cost, inc, dec)
    print(bandit.mus, bandit.cost)
    results = []
    filename = '{}'.format(bandit) + ', ' + str(policy.__name__ if policy.__name__ != "parameterizedRegretQuery" else "MCCH")
    arr_file_name = [filename]
    # print(policy.__name__)
    if multiprocess:
        print('start runs with multiprocess ' + str(policy.__name__ if policy.__name__ != "parameterizedRegretQuery" else "MCCH"))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor_res = [executor.submit(playBernoulli, _, df, writer, writer_sum, bandit, policy, True, **kwargs) for _ in range(N)]
            for f in concurrent.futures.as_completed(executor_res):
                print("Im here")
                # results.append(f.result())
                results.append(f.result())
                # results.append(filename)
                # with open(filename, 'w') as f:
                #     json.dump(results[0], f)
                    # json.dump(arr_file_name, f)
    else:
        print("start runs with no multiprocess")
        print(policy.__name__)
        for i in range(N):
            print('round %d out of %d' % (i, N))
            #        random.shuffle(mus)
            bandit = Bandit(mus, n, cost, inc, dec)
            # results.append(playBernoulli(i, df, bandit, policy, True, **kwargs))
            # df = df.append(playBernoulli(i, df, bandit, policy, True, **kwargs))
            df = playBernoulli(i, df, w, sum_w, bandit, policy, True, **kwargs)
            # with open(filename, 'w') as f:
            #     json.dump(results, f)
                # json.dump(arr_file_name, f)
    #     with open('df_as_csv.csv', 'a') as csvfile:
    #         df.to_csv(csvfile, header = False)
    #
    # # f.close()
    # csvfile.close()
    # return bandit, policy, results
    return bandit, policy, df

if __name__ == '__main__':
    start = time.perf_counter()

    cols = ['run', 'iter', 'policy', 'mus', 'init_cost', 'increase_by', 'horizon', 'regret', 'arm', 'query_history',
            'last_query_step','cum_number_of_queries']
    df = pd.DataFrame(columns=cols)

    # mus_arr = [[0.0, 1.0], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.2, 0.3], [0.8, 0.7]] #, [0.7, 0.5, 0.4, 0.4],[0.6, 0.2, 0.4, 0.9], [0.3, 0.1, 0.5, 0.2], [0.1, 0.5, 0.9], [0.2, 0.4, 0.6], [0.3, 0.4, 0.7]]
    # costs = [0.5, 1, 2, 5, 10]

    # run settings:
    mus_arr = [[1, 0], [0.2, 0.8]]
    costs = [1]
    n = 500
    runs = 100
    inc = 2
    dec = 1.0/2.0
    a_MCCH_param = 0.1

    filename = '500iter_runs_alpha01_2arms_02_08_results_new_' + datetime.now().strftime("%Y_%m_%d-%I_%M_%S") +'.csv'
    filename_summary = '500iter_runs_alpha01_2arms_02_08_summary_new_' + datetime.now().strftime("%Y_%m_%d-%I_%M_%S") +'.csv'

    # filename = '200runs_1000iter_2arms_alpha01' + datetime.now().strftime("%Y_%m_%d-%I_%M_%S") + '.csv'
    # filename_summary = '200runs_1000iter_2arms_alpha01_summary' + datetime.now().strftime("%Y_%m_%d-%I_%M_%S") + '.csv'

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(cols)
        with open (filename_summary, 'w', newline='') as csvfile2:
            writer_sum = csv.writer(csvfile2)
            writer_sum.writerow(cols)
    # with open('res_addition.csv', 'w',newline='') as csvfile:

        #
        # regret_KG = runExperiment(mus, n, 1, inc, dec, knowledgeGradientPolicy, runs, df, writer, progressbar=True,
        #                           multiprocess=False)[2]
            for i in mus_arr:
                for j in costs:
                    # for a in a_MCCH_param:
                    # print(j, i)
                    regret_KG = runExperiment(i, n, j, inc, dec, knowledgeGradientPolicy, runs, df, writer, writer_sum, progressbar = True, multiprocess=False)[2]
                    regret_PRQ_DMED = runExperiment(i, n, j, inc, dec, parameterizedRegretQuery, runs, df,writer, writer_sum, progressbar = True, banditpolicy=DMED, multiprocess=False, alpha=a_MCCH_param)[2]

            csvfile2.close()
        csvfile.close()
    # with

    # for j in costs:
    #     for i in mus_arr:
    #         regret_KG = runExperiment(mus, n, cost, inc, dec, knowledgeGradientPolicy, runs, df, progressbar=True, multiprocess=False)[2]
    #         regret_PRQ_DMED = runExperiment(mus, n, cost, inc, dec, parameterizedRegretQuery, runs, df, progressbar=True, banditpolicy=DMED,
    #                       multiprocess=False, alpha=0.35)[2]

    # start = time.perf_counter()
    # regret_KG = runExperiment(mus, n, cost, inc, dec, knowledgeGradientPolicy, runs, df, progressbar=True, multiprocess=False)[2]
    # print(regret_KG[0])
    # start = time.perf_counter()
    # regret_PRQ_DMED = runExperiment(mus, n, cost, inc, dec, parameterizedRegretQuery, runs, df, progressbar=True, banditpolicy=DMED, multiprocess=False, alpha=0.35)[2]
    # print(regret_PRQ_DMED[0])
    #
    finish = time.perf_counter()
    # finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} seconds (s)')