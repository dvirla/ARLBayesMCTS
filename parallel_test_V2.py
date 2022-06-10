import argparse
import csv
import pickle
import threading
import multiprocessing as mp
from MCTS_V2 import MCTree
import numpy as np
from scipy.stats import bernoulli, beta
from multiprocessing import Pool
import random
import pandas as pd
from datetime import datetime
from copy import deepcopy


# create the lock
csv_writer_lock = threading.Lock()


def calc_mean_bernouli_param(a, b):
    samples = beta.rvs(a, b, size=10000)
    samples_mean = np.mean(samples)
    return samples_mean


def get_a_b(arms_dict):
    params = [None] * 4
    for i in range(2):  # theta for each of the two arms
        d = arms_dict[i]
        params[i*2] = d['succ'] + 0.5
        params[i*2 + 1] = d['fails'] + 0.5
    return params


def update_arms_conf(arms_p_confidences, new_0_mean, new_1_mean):
    arms_p_confidences[0] = np.linalg.norm(arms_p_confidences[0] - new_0_mean) if arms_p_confidences[0] != float("inf") else new_0_mean
    arms_p_confidences[1] = np.linalg.norm(arms_p_confidences[1] - new_1_mean) if arms_p_confidences[1] != float("inf") else new_1_mean

    return arms_p_confidences


def parallel_write(writer_path, run, t, arms_thetas, base_query_cost, query_cost, T, regret, action, query_ind, r,
                   seed):
    with open(writer_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(
            {'run': run, 'timestep': t, 'mus': arms_thetas, 'base_query_cost': base_query_cost,
             'query_cost': query_cost, 'horizon': T, 'regret': regret,
             'chosen_arm': action, 'query_ind': query_ind, 'reward': r, 'seed': seed})


def BAMCP_PP(writer_func, writer_path, run, T, learning_rate, discount_factor, base_query_cost, exploration_const,
             max_simulations, arms_thetas: tuple, delayed_tree_expansion, seed, increase_factor, decrease_factor):
    """
    :param exploration_const: exploration constant for choosing next action during simulation via UCB.
    :param decrease_factor: decrease factor of the query cost
    :param increase_factor: increase factor of the query cost
    :param delayed_tree_expansion: maximum depth allowed in each search before backpropagation
    :param run: number of run (out of total number runs)
    :param writer_path: path for csv recorder
    :param writer_func: function for writing to csv
    :param max_simulations: maximum simulation per search call
    :param arms_thetas:probabilities for bernouli functions
    :param T: Horizon
    :param learning_rate: for Q values
    :param discount_factor: for Q values
    :param query_cost: base query cost
    """
    random.seed(seed)
    np.random.seed(seed)

    actions_history = tuple([])
    regret = 0
    Q = np.random.randn(2, 2)

    mctree = MCTree(actions_history, learning_rate, discount_factor, base_query_cost, increase_factor, decrease_factor,
                    exploration_const)
    arms_p_confidences = [float("inf"), float("inf")]
    arms_p_confidences_history = {"arm_0": [], "arm_1": []}

    node = None
    for t in range(T):
        action, query_ind, node = mctree.tree_search(Q.copy(), max_depth=delayed_tree_expansion, root=node,
                                                     max_simulations=max_simulations, t=t, horizon=T, arms_p_confidences=arms_p_confidences)
        r = bernoulli(arms_thetas[action]).rvs()
        if query_ind:
            mctree.q_update(Q, action, query_ind, r)  # , log_dict=Q_vals_dict
            # Update rewards history for bayesian update
            mctree.update_arm_dict(action, r)
            a_0, b_0, a_1, b_1 = get_a_b(mctree.arms_dicts)
            new_0_mean = calc_mean_bernouli_param(a_0, b_0)
            new_1_mean = calc_mean_bernouli_param(a_1, b_1)
            arms_p_confidences = update_arms_conf(arms_p_confidences, new_0_mean, new_1_mean)
            arms_p_confidences_history["arm_0"].append(arms_p_confidences[0])
            arms_p_confidences_history["arm_1"].append(arms_p_confidences[1])

        new_history = list(actions_history)
        new_history.append((action, query_ind))
        actions_history = tuple(new_history)

        regret += max(arms_thetas) - r
        with csv_writer_lock:
            writer_func(writer_path, run, t, arms_thetas, base_query_cost, mctree.query_cost, T, regret, action,
                        query_ind, r, seed)

    #df = pd.DataFrame(arms_p_confidences_history)
    #df.to_csv("./conf_values/{0}_{1}".format(run, writer_path.split("records_tests/")[1]), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning_rate', type=float, default=1.,
                        metavar='')  # No learning rate according to both papers
    parser.add_argument('--discount_factor', type=float, default=0.95, metavar='')
    parser.add_argument('--base_query_cost', type=float, default=0., metavar='')
    parser.add_argument('--exploration_const', type=float, default=1, metavar='')
    parser.add_argument('--max_simulations', type=int, default=1000, metavar='')
    parser.add_argument('--horizon', type=int, default=500, metavar='')
    parser.add_argument('--arms_thetas', type=tuple, default=(0.2, 0.8), metavar='')
    parser.add_argument('--runs', type=int, default=1, metavar='')
    parser.add_argument('--delayed_tree_expansion', type=int, default=10, metavar='')
    parser.add_argument('--increase_factor', type=float, default=2., metavar='')
    parser.add_argument('--decrease_factor', type=float, default=0.5, metavar='')

    args = parser.parse_args()
    num_workers = max(mp.cpu_count() - 40, 4)
    exp_const = str(args.exploration_const).split('.')
    if len(exp_const) > 1 and exp_const[-1] == "0":
        exp_const = ''.join(exp_const[:-1])
    else:
        exp_const = ''.join(exp_const)

    now = datetime.now()
    now = now.strftime("%m%d%Y%H%M%S")
    writer_path = './records_tests/test_record_{0}_sim_{1}_exp_{2}_runs_{3}_tree_{4}.csv'.format(args.max_simulations,
                                                                                              exp_const,
                                                                                              args.runs,
                                                                                              args.delayed_tree_expansion,
                                                                                                now)
    with open(writer_path.split('.csv')[0]+'.pkl', 'wb') as f:
        pickle.dump(args, f)

    with open(writer_path, 'w', newline='') as csvfile:
        fieldnames = ['run', 'timestep', 'mus', 'base_query_cost', 'query_cost', 'horizon', 'regret', 'chosen_arm',
                      'query_ind', 'reward', 'seed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

    func_args = []
    seed = 0
    horizon = args.horizon
    base_query_cost = args.base_query_cost
    for run in range(args.runs):
        func_args.append((parallel_write, writer_path, run, horizon,
                          args.learning_rate, args.discount_factor,
                          base_query_cost,
                          args.exploration_const, args.max_simulations,
                          args.arms_thetas, args.delayed_tree_expansion, seed,
                          args.increase_factor, args.decrease_factor))
        seed += 1

    num_tasks = len(func_args)
    with Pool(num_workers) as p:
        for i, _ in enumerate(p.starmap(BAMCP_PP, func_args), 1):
            print('\rdone {0:.2%}'.format(i / num_tasks))
