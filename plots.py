import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_heatmap(data, cost):
    ax = sns.heatmap(data, linewidth=0.5, cmap='coolwarm')
    plt.title(f"Cost = {cost}\nQuery Indicator Heatmap")
    plt.savefig(f'./fixed_cost_heatmap.jpg')


if __name__ == "__main__":
    df = pd.read_csv('./record.csv')
    queries_map = np.zeros((100, 50))  # runs, horizon
    for i, row in df.iterrows():
        run = int(row['run'])
        t = i%50
        queries_map[run][t] = row['query_ind']
    plot_heatmap(queries_map, 0.5)