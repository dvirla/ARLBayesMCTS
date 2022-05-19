import argparse
from backup.MCTS import MCTree
from scipy.stats import bernoulli
from tqdm import tqdm
import numpy as np
import wandb


def BAMCP_PP(T, learning_rate, discount_factor, query_cost, exploration_const, max_simulations, arms_thetas, delayed_tree_expansion):
    """
    :param T: Horizon
    :param learning_rate: for Q values
    :param discount_factor: for Q values
    :param query_cost: fixed query cost for Q values
    """
    total_reward = 0
    actions_history = tuple([])
    regret = 0
    Q = np.random.randn(2, 2)
    mctree = MCTree(actions_history, learning_rate, discount_factor, query_cost, exploration_const)
    for t in range(T):
        max_depth = min(T - t, delayed_tree_expansion)  # Preventing expansion over the horizon
        if max_depth == 0:
            max_depth = T - t
        action, query_ind = mctree.tree_search(Q.copy(), actions_history, max_depth=max_depth,
                                               max_simulations=max_simulations)
        r = bernoulli(arms_thetas[action]).rvs()
        if query_ind:
            mctree.q_update(Q, action, query_ind, r)

        new_history = list(actions_history)
        new_history.append((action, query_ind))
        actions_history = tuple(new_history)

        total_reward += r
        regret += arms_thetas[action] - r

        # writer.writerow(
        #     {'run': run, 'timestep': t, 'mus': arms_thetas, 'query_cost': query_cost, 'horizon': T, 'regret': regret,
        #      'chosen_arm': action, 'query_ind': query_ind, 'reward': r})

    return total_reward / T, regret / T


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning_rate', type=float, default=1.,
                        metavar='')  # No learning rate according to both papers
    parser.add_argument('--discount_factor', type=float, default=0.95, metavar='')  # According to original BAMCP paper
    parser.add_argument('--query_cost', type=float, default=0.5, metavar='')  # According to BAMCP++ paper
    parser.add_argument('--exploration_const', type=float, default=5., metavar='')  # TODO: optimize?
    parser.add_argument('--max_simulations', type=int, default=200, metavar='')  # TODO: maybe can be higher?
    parser.add_argument('--arms_thetas', type=tuple, default=(0.2, 0.8), metavar='')  # According to BAMCP++ paper
    parser.add_argument('--runs', type=int, default=100, metavar='')
    parser.add_argument('--delayed_tree_expansion', type=int, default=0, metavar='')  # TODO: optimize?

    args = parser.parse_args()

    wandb.init(project="ARL", entity="dvirlafer")
    wandb.config.update(
        {'exploration_const': args.exploration_const, 'delayed_tree_expansion': args.delayed_tree_expansion})

    # with open('./expiriment_record.csv', 'w', newline='') as csvfile:
    #     fieldnames = ['run', 'timestep', 'mus', 'query_cost', 'horizon', 'regret', 'chosen_arm', 'query_ind', 'reward']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # writer.writeheader()
    for horizon in [10, 20, 30, 40, 50]:
        pbar = tqdm(range(args.runs))
        pbar.set_description('Horizon = ', horizon)
        total_reward_per_run, total_regret_per_run = 0, 0
        for run in pbar:
            avg_reward, avg_regret = BAMCP_PP(horizon, args.learning_rate, args.discount_factor,
                                              args.query_cost,
                                              args.exploration_const, args.max_simulations,
                                              args.arms_thetas, args.delayed_tree_expansion)
            total_reward_per_run += avg_reward
            total_regret_per_run += avg_regret

        avg_reward, avg_regret = total_reward_per_run / args.runs, total_regret_per_run / args.runs
        wandb.log({'Average_Reward': avg_reward, 'Average_Regret': avg_regret})
