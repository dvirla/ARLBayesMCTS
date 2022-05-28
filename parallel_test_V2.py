import argparse
import csv
import threading
import multiprocessing as mp
from MCTS_V2 import MCTree
import numpy as np
from scipy.stats import bernoulli
from multiprocessing import Pool
import random
import pandas as pd

# create the lock
csv_writer_lock = threading.Lock()


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
    # Q_vals_dict = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}

    mctree = MCTree(actions_history, learning_rate, discount_factor, base_query_cost, increase_factor, decrease_factor,
                    exploration_const)
    node = None
    for t in range(T):
        action, query_ind, node = mctree.tree_search(Q.copy(), max_depth=delayed_tree_expansion, root=node,
                                                     max_simulations=max_simulations)
        r = bernoulli(arms_thetas[action]).rvs()
        if query_ind:
            mctree.q_update(Q, action, query_ind, r)  # , log_dict=Q_vals_dict
            # Update rewards history for bayesian update
            mctree.update_arm_dict(action, r)

        new_history = list(actions_history)
        new_history.append((action, query_ind))
        actions_history = tuple(new_history)

        regret += max(arms_thetas) - r
        with csv_writer_lock:
            writer_func(writer_path, run, t, arms_thetas, base_query_cost, mctree.query_cost, T, regret, action,
                        query_ind, r, seed)

    # MCTS_Q_vals_df = pd.DataFrame(mctree.Q_vals_dict.values()).transpose().rename(columns={0: '(0,0)', 1: '(0,1)', 2: '(1,0)', 3: '(1,1)'}).round(4)
    # BAMCP_Q_vals_df = pd.DataFrame(Q_vals_dict.values()).transpose().rename(columns={0: '(0,0)', 1: '(0,1)', 2: '(1,0)', 3: '(1,1)'}).round(4)
    # bayes_params_df = pd.DataFrame(mctree.bayes_params_dict)

    # MCTS_Q_path = './MCTS_Q_vals/run_{0}_{1}_sim_{2}_exp_{3}_runs_{4}_tree.csv'.format(run, args.max_simulations,
    #                                                                                    ''.join(
    #                                                                                        f'{args.exploration_const}'.split(
    #                                                                                            '.')[:-1]),
    #                                                                                    args.runs,
    #                                                                                    args.delayed_tree_expansion)
    # BAMCP_Q_path = './BAMCP_Q_vals/run_{0}_{1}_sim_{2}_exp_{3}_runs_{4}_tree.csv'.format(run, args.max_simulations,
    #                                                                                      ''.join(
    #                                                                                          f'{args.exploration_const}'.split(
    #                                                                                              '.')[:-1]),
    #                                                                                      args.runs,
    #                                                                                      args.delayed_tree_expansion)
    # bayes_params_path = './bayes_params/run_{0}_{1}_sim_{2}_exp_{3}_runs_{4}_tree.csv'.format(run, args.max_simulations,
    #                                                                                           ''.join(
    #                                                                                               f'{args.exploration_const}'.split(
    #                                                                                                   '.')[:-1]),
    #                                                                                           args.runs,
    #                                                                                           args.delayed_tree_expansion)
    # MCTS_Q_vals_df.to_csv(MCTS_Q_path)
    # BAMCP_Q_vals_df.to_csv(BAMCP_Q_path)
    # bayes_params_df.to_csv(bayes_params_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning_rate', type=float, default=1.,
                        metavar='')  # No learning rate according to both papers
    parser.add_argument('--discount_factor', type=float, default=0.95, metavar='')  # According to original BAMCP paper
    parser.add_argument('--query_cost', type=float, default=0., metavar='')  # According to BAMCP++ paper
    parser.add_argument('--exploration_const', type=float, default=1, metavar='')
    parser.add_argument('--max_simulations', type=int, default=1000, metavar='')
    parser.add_argument('--horizon', type=int, default=500, metavar='')
    parser.add_argument('--arms_thetas', type=tuple, default=(0.2, 0.8), metavar='')  # According to BAMCP++ paper
    parser.add_argument('--runs', type=int, default=1, metavar='')
    parser.add_argument('--delayed_tree_expansion', type=int, default=10, metavar='')
    parser.add_argument('--increase_factor', type=float, default=2., metavar='')
    parser.add_argument('--decrease_factor', type=float, default=0.5, metavar='')

    args = parser.parse_args()
    num_workers = max(mp.cpu_count() - 18, 4)
    exp_const = str(args.exploration_const).split('.')
    if len(exp_const) > 1 and exp_const[-1] == "0":
        exp_const = ''.join(exp_const[:-1])
    else:
        exp_const = ''.join(exp_const)
    writer_path = './test_record_{0}_sim_{1}_exp_{2}_runs_{3}_tree_svetas_setting.csv'.format(args.max_simulations,
                                                                                              exp_const,
                                                                                              args.runs,
                                                                                              args.delayed_tree_expansion)
    with open(writer_path, 'w', newline='') as csvfile:
        fieldnames = ['run', 'timestep', 'mus', 'base_query_cost', 'query_cost', 'horizon', 'regret', 'chosen_arm',
                      'query_ind', 'reward', 'seed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

    func_args = []
    seed = 0
    horizon = args.horizon
    base_query_cost = 1
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
