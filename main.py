import argparse
from QLearner import QLearner
from MCTS import MCTree
from scipy.stats import bernoulli
import pandas as pd
from tqdm import tqdm


def BAMCP_PP(T, learning_rate, discount_factor, query_cost, exploration_const, max_simulations,
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
    qlearner = QLearner(learning_rate, discount_factor, query_cost)
    mctree = MCTree(actions_history, learning_rate, discount_factor, query_cost, exploration_const)
    for t in range(T):
        action, query_ind = mctree.tree_search(qlearner.Q.copy(), actions_history, max_depth=T - t - delayed_tree_expansion,
                                               max_simulations=max_simulations)
        r = bernoulli(arms_thetas[action]).rvs()
        if query_ind:
            qlearner.q_update(action, query_ind, r)

        new_history = list(actions_history)
        new_history.append((action, query_ind))
        actions_history = tuple(new_history)

        regret += arms_thetas[action] - r
        rewards.append(r)
        chosen_arms.append(action)
        query_inds.append(query_ind)
        regrets.append(regret)

    return chosen_arms, rewards, query_inds, regrets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning_rate', type=float, default=1., metavar='')  # No learning rate according to both papers
    parser.add_argument('--discount_factor', type=float, default=0.95, metavar='')  # According to original BAMCP paper
    parser.add_argument('--query_cost', type=float, default=0.5, metavar='')  # According to BAMCP++ paper
    parser.add_argument('--exploration_const', type=float, default=5., metavar='')  # TODO: optimize?
    parser.add_argument('--max_simulations', type=int, default=100, metavar='')  # TODO: maybe can be higher?
    parser.add_argument('--arms_thetas', type=tuple, default=(0.2, 0.8), metavar='')  # According to BAMCP++ paper
    parser.add_argument('--horizon', type=int, default=30, metavar='')
    parser.add_argument('--runs', type=int, default=100, metavar='')
    parser.add_argument('--delayed_tree_expansion', type=int, default=0, metavar='')  # TODO: optimize?

    args = parser.parse_args()
    assert args.horizon > args.delayed_tree_expansion + 1

    for i, query_cost in enumerate((0, 0.3, 0.5, 1, 100)):
        recorder = {'run': [], 'query_cost': [], 'chosen_arm': [], 'reward': [], 'query_ind': [], 'regret': []}
        print(f'Starting cost = {query_cost}')
        for run in tqdm(range(args.runs)):
            chosen_arms, rewards, query_inds, regrets = BAMCP_PP(args.horizon, args.learning_rate, args.discount_factor,
                                                                 query_cost,
                                                                 args.exploration_const, args.max_simulations,
                                                                 args.arms_thetas, args.delayed_tree_expansion)
            recorder['run'].extend([run] * args.horizon)
            recorder['query_cost'].extend([query_cost] * args.horizon)
            recorder['chosen_arm'].extend(chosen_arms)
            recorder['reward'].extend(rewards)
            recorder['query_ind'].extend(query_inds)
            recorder['regret'].extend(regrets)

        df = pd.DataFrame(recorder)
        df.to_csv(f'./records/record_q_per_node_{i}.csv')
