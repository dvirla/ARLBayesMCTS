import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_heatmap(data, cost, cost_idx, arm_dist_title):
    ax = sns.heatmap(data, linewidth=0.5, cmap='coolwarm', vmin=0, vmax=1)
    plt.title(f"Cost = {cost}\nQuery Indicator Heatmap\n{arm_dist_title}")
    plt.savefig(f'./Images/fixed_cost_heatmap_{cost_idx}.jpg')


def plot_queries(arr, title, ylabel, path, horizon):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(list(range(1, horizon + 1)), arr)
    ax.set_xticks(list(range(1, horizon + 1, 4)))
    ax.set_title(title)
    ax.set_xlabel('Timestamp')
    ax.set_ylabel(ylabel)
    plt.savefig(path)


if __name__ == "__main__":
    horizon = 30
    runs = 100
    arm_dist_title = 'Arms Distribution = {0.2, 0.8}'
    for i, query_cost in enumerate((0, 0.3, 0.5, 1, 100)):
        k = i
        df = pd.read_csv(f'./records/record_q_per_node_{k}.csv')

        queries_map = np.zeros((runs, horizon))
        for j, row in df.iterrows():
            run = int(row['run'])
            t = j % horizon
            queries_map[run][t] = row['query_ind']

        plot_heatmap(queries_map, query_cost, k, arm_dist_title)

        df['timestamp'] = [k % horizon for k in range(len(df))]
        means_df = df.groupby('timestamp').mean()[['reward', 'query_ind', 'regret', 'chosen_arm']]

        plot_queries(means_df['reward'], f'Average Reward at Cost = {query_cost}\n{arm_dist_title}', 'Avg. Reward', f'./Images/average_reward_cost_{k}.jpg', horizon)
        plot_queries(means_df['query_ind'], f'Query Probability at Cost = {query_cost}\n{arm_dist_title}', 'Query Probability', f'./Images/query_prob_cost_{k}.jpg', horizon)
        plot_queries(means_df['regret'], f'Average Regret at Cost = {query_cost}\n{arm_dist_title}', 'Avg. Regret', f'./Images/average_regret_cost_{k}.jpg', horizon)
        plot_queries(means_df['chosen_arm'], f'Average Chosen Arm at Cost = {query_cost}\n{arm_dist_title}', 'Avg. Arm', f'./Images/average_chosen_arm_cost_{k}.jpg', horizon)
