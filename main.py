import argparse
from QLearner import QLearner
from MCTS import MCTree
from History import History
from scipy.stats import bernoulli
import pandas as pd
from tqdm import tqdm


def BAMCP_PP(T, learning_rate, discount_factor, query_cost, exploration_const, max_simulations,
             arms_thetas: tuple):
    """
    :param run: idx of the current run, fixed per run
    :param T: Horizon
    :param learning_rate: for Q values
    :param discount_factor: for Q values
    :param query_cost: fixed query cost for Q values
    """
    rewards, chosen_arms, query_inds, regrets = [], [], [], []
    history = History()
    qlearner = QLearner(learning_rate, discount_factor, query_cost)
    mctree = MCTree(history, learning_rate, discount_factor, query_cost, exploration_const)
    for t in range(T):
        action, query_ind = mctree.tree_search(qlearner.Q.copy(), history, max_depth=T - t,
                                               max_simulations=max_simulations)
        r = bernoulli(arms_thetas[action]).rvs()
        if query_ind:
            observed_reward = r
            qlearner.q_update(action, query_ind, r)
        else:
            observed_reward = None
        history = history.update(action, observed_reward)

        regret = arms_thetas[action] - r
        rewards.append(r)
        chosen_arms.append(action)
        query_inds.append(query_ind)
        regrets.append(regret)

    return chosen_arms, rewards, query_inds, regrets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning_rate', type=float, default=1.,
                        metavar='')  # No learning rate according to both papers
    parser.add_argument('--discount_factor', type=float, default=0.95, metavar='')  # According to original BAMCP paper
    parser.add_argument('--query_cost', type=float, default=0.5, metavar='')  # According to BAMCP++ paper
    parser.add_argument('--exploration_const', type=float, default=3., metavar='')  # According to original BAMCP paper
    parser.add_argument('--max_simulations', type=int, default=100, metavar='')  # TODO: maybe can be higher
    parser.add_argument('--arms_thetas', type=tuple, default=(0.2, 0.8), metavar='')  # According to BAMCP++ paper
    parser.add_argument('--horizon', type=int, default=50, metavar='')  # According to original BAMCP paper
    parser.add_argument('--runs', type=int, default=100, metavar='')  # According to original BAMCP paper

    args = parser.parse_args()

    recorder = {'run': [], 'chosen_arm': [], 'reward': [], 'query_ind': [], 'regret': []}

    for run in tqdm(range(args.runs)):
        chosen_arms, rewards, query_inds, regrets = BAMCP_PP(args.horizon, args.learning_rate, args.discount_factor,
                                                             args.query_cost,
                                                             args.exploration_const, args.max_simulations,
                                                             args.arms_thetas)
        recorder['run'].extend([run] * args.horizon)
        recorder['chosen_arm'].extend(chosen_arms)
        recorder['reward'].extend(rewards)
        recorder['query_ind'].extend(query_inds)
        recorder['regret'].extend(regrets)

    df = pd.DataFrame(recorder)
    df.to_csv('./record.csv')
