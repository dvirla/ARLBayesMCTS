import argparse
import csv
import pickle
import threading
import multiprocessing as mp
from MCTS_V2 import MCTree
import numpy as np
from scipy.stats import bernoulli
from multiprocessing import Pool
import random
from datetime import datetime


# create the lock
csv_writer_lock = threading.Lock()
now = datetime.now()
now = now.strftime("%m%d%Y%H%M%S")


def parallel_write(writer_path, run, t, arms_thetas, base_query_cost, query_cost, T, regret, action, query_ind, r,
                   seed):
    fieldnames = ['run', 'timestep', 'mus', 'base_query_cost', 'query_cost', 'horizon', 'regret', 'chosen_arm',
                  'query_ind', 'reward', 'seed']
    with open(writer_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(
            {'run': run, 'timestep': t, 'mus': arms_thetas, 'base_query_cost': base_query_cost,
             'query_cost': query_cost, 'horizon': T, 'regret': regret,
             'chosen_arm': action, 'query_ind': query_ind, 'reward': r, 'seed': seed})


def BAMCP_PP(writer_func, writer_path, run, T, learning_rate, discount_factor, base_query_cost, exploration_const,
             max_simulations, arms_thetas: tuple, delayed_tree_expansion, seed, increase_factor, decrease_factor, use_temperature):
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

    node = None
    for t in range(T):
        if use_temperature:
            action, query_ind, node = mctree.tree_search(Q.copy(), max_depth=delayed_tree_expansion, root=node,
                                                         max_simulations=max_simulations, t=t, horizon=T)
        else:
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


def run_multi_expiriemnt(args):
    combs = [((0.2, 0.8), 1, 2, 0.5), ((1, 0), 1, 2, 0.5), ((0.5, 0.5), 1, 2, 0.5),
             ((1, 0), 0.5, 1, 1), ((0.2, 0.8), 0.5, 1, 1), ((0.2, 0.8), 0, 1, 1), ((1, 0), 0, 1, 1)]
    exp_const = str(args.exploration_const).split('.')
    if len(exp_const) > 1 and exp_const[-1] == "0":
        exp_const = ''.join(exp_const[:-1])
    else:
        exp_const = ''.join(exp_const)

    for mus, cost, increase_factor, decrease_factor in combs:
        args.increase_factor = increase_factor
        args.decrease_factor = decrease_factor
        args.base_query_cost = cost
        args.arms_thetas = mus
        if increase_factor != 1:
            folder = '/changing_cost'
            writer_cost = 'changing_cost'
        else:
            folder = '/fixed_cost'
            writer_cost = '05_cost' if cost != 0 else 'no_cost'

        writer_path = './records_tests/with_temperature/{0}/test_record_{1}_{2}_sim_{3}_exp_{4}_arms_{5}_tree_{6}.csv'.format(
            folder, writer_cost, args.max_simulations,
            exp_const,
            '_'.join([str.rstrip(''.join(str(x).split('.')), '0')
                      for x in args.arms_thetas]),
            args.delayed_tree_expansion,
            now)

        runner(writer_path, args)

        with open(writer_path.split('.csv')[0] + '.pkl', 'wb') as f:
            pickle.dump({'mus': mus, 'cost': cost, 'inc_factor': increase_factor, 'dec_factor': decrease_factor}, f)


def runner(writer_path, args, ):
    num_workers = max(mp.cpu_count() - 40, 4)
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
                          args.increase_factor, args.decrease_factor, args.use_temperature))
        seed += 1

    num_tasks = len(func_args)
    with Pool(num_workers) as p:
        for i, _ in enumerate(p.starmap(BAMCP_PP, func_args), 1):
            print('\rdone {0:.2%}'.format(i / num_tasks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning_rate', type=float, default=1.,
                        metavar='')  # No learning rate according to both papers
    parser.add_argument('--discount_factor', type=float, default=0.95, metavar='')
    parser.add_argument('--base_query_cost', type=float, default=0., metavar='')
    parser.add_argument('--exploration_const', type=float, default=1, metavar='')
    parser.add_argument('--max_simulations', type=int, default=1000, metavar='')
    parser.add_argument('--horizon', type=int, default=500, metavar='')
    parser.add_argument('--arms_thetas', nargs='+', type=float, default=[0.2, 0.8], metavar='')
    parser.add_argument('--runs', type=int, default=1, metavar='')
    parser.add_argument('--delayed_tree_expansion', type=int, default=10, metavar='')
    parser.add_argument('--increase_factor', type=float, default=2., metavar='')
    parser.add_argument('--decrease_factor', type=float, default=0.5, metavar='')
    parser.add_argument('--use_temperature', action='store_true')
    parser.add_argument('--multi_exp', action='store_true')

    args = parser.parse_args()
    if args.multi_exp:
        run_multi_expiriemnt(args)
    else:
        args.arms_thetas = tuple(args.arms_thetas)
        exp_const = str(args.exploration_const).split('.')
        if len(exp_const) > 1 and exp_const[-1] == "0":
            exp_const = ''.join(exp_const[:-1])
        else:
            exp_const = ''.join(exp_const)

        is_temp = 'with_temp' if args.use_temperature else ''
        if args.increase_factor != 1:
            w_cost = 'changing_cost'
        else:
            w_cost = '05_cost' if args.base_query_cost != 0 else 'no_cost'

        writer_path = './records_tests/test_record_{0}_{1}_{2}_sim_{3}_exp_{4}_arms_{5}_tree_{6}.csv'.format(
            is_temp, w_cost, args.max_simulations,
            exp_const,
            '_'.join([str.rstrip(''.join(str(x).split('.')), '0')
                      for x in args.arms_thetas]),
            args.delayed_tree_expansion,
            now)

        with open(writer_path.split('.csv')[0]+'.pkl', 'wb') as f:
            pickle.dump(args, f)

        runner(writer_path, args)
