import argparse
from MCTS import MCTree
from scipy.stats import bernoulli
from tqdm import tqdm
import numpy as np
import csv


def BAMCP_PP(writer, run, T, learning_rate, discount_factor, query_cost, exploration_const, max_simulations,
             arms_thetas: tuple, delayed_tree_expansion):
    """
    :param T: Horizon
    :param learning_rate: for Q values
    :param discount_factor: for Q values
    :param query_cost: fixed query cost for Q values
    """
    rewards, chosen_arms, query_inds, regrets = [], [], [], []
    actions_history = tuple([])
    regret = 0
    Q = np.random.randn(2, 2)
    mctree = MCTree(actions_history, learning_rate, discount_factor, query_cost, exploration_const)
    for t in range(T):
        action, query_ind = mctree.tree_search(Q.copy(), actions_history, max_depth=delayed_tree_expansion,
                                               max_simulations=max_simulations)
        r = bernoulli(arms_thetas[action]).rvs()
        if query_ind:
            mctree.q_update(Q, action, query_ind, r)

        new_history = list(actions_history)
        new_history.append((action, query_ind))
        actions_history = tuple(new_history)

        regret += arms_thetas[action] - r
        rewards.append(r)
        chosen_arms.append(action)
        query_inds.append(query_ind)
        regrets.append(regret)

        writer.writerow(
            {'run': run, 'timestep': t, 'mus': arms_thetas, 'query_cost': query_cost, 'horizon': T, 'regret': regret,
             'chosen_arm': action, 'query_ind': query_ind, 'reward': r})

    return chosen_arms, rewards, query_inds, regrets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning_rate', type=float, default=1., metavar='')  # No learning rate according to both papers
    parser.add_argument('--discount_factor', type=float, default=0.95, metavar='')  # According to original BAMCP paper
    parser.add_argument('--query_cost', type=float, default=0.5, metavar='')  # According to BAMCP++ paper
    parser.add_argument('--exploration_const', type=float, default=5., metavar='')  # TODO: optimize?
    parser.add_argument('--max_simulations', type=int, default=100, metavar='')  # TODO: maybe can be higher?
    parser.add_argument('--arms_thetas', type=tuple, default=(0.2, 0.8), metavar='')  # According to BAMCP++ paper
    parser.add_argument('--runs', type=int, default=100, metavar='')
    parser.add_argument('--delayed_tree_expansion', type=int, default=0, metavar='')  # TODO: optimize?

    args = parser.parse_args()

    with open('./test_record.csv', 'w', newline='') as csvfile:
        fieldnames = ['run', 'timestep', 'mus', 'query_cost', 'horizon', 'regret', 'chosen_arm', 'query_ind', 'reward']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for horizon in (10, 20, 30, 40, 50):
            for query_cost in (0, 0.3, 0.5, 1, 100):
                print(f'Horizon = {horizon} and cost = {query_cost}')
                for run in tqdm(range(args.runs)):
                    chosen_arms, rewards, query_inds, regrets = BAMCP_PP(writer, run, horizon, args.learning_rate, args.discount_factor,
                                                                         query_cost,
                                                                         args.exploration_const, args.max_simulations,
                                                                         args.arms_thetas, args.delayed_tree_expansion)
