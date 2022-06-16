from bandits import knowledgeGradientChangingCostPolicy, runExperiment
import argparse
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--base_query_cost', type=float, default=1., metavar='')
    parser.add_argument('--horizon', type=int, default=500, metavar='')
    parser.add_argument('--arms_thetas', nargs='+', type=float, default=[0.2, 0.8], metavar='')
    parser.add_argument('--runs', type=int, default=100, metavar='')
    parser.add_argument('--increase_factor', type=float, default=2., metavar='')
    parser.add_argument('--decrease_factor', type=float, default=0.5, metavar='')

    args = parser.parse_args()
    mus = args.arms_thetas
    horizon = args.horizon
    runs = args.runs
    cost = args.base_query_cost
    increase_factor = args.increase_factor
    decrease_factor = args.decrease_factor

    runs_list, timesteps, arms_thetas, base_query_cost, query_costs, horizon, chosen_arms, query_inds, rewards = runExperiment(
        mus, horizon, cost, increase_factor,
        decrease_factor,
        knowledgeGradientChangingCostPolicy,
        runs, progressbar=True)

    to_df = {'run': runs, 'timestep': timesteps, 'mus': arms_thetas, 'base_query_cost': base_query_cost,
             'query_cost': query_costs, 'horizon': horizon, 'chosen_arm': chosen_arms, 'query_ind': query_inds,
             'reward': rewards}
    df = pd.DataFrame(to_df)
    df.to_csv('./kd_50_horizon_100_runs_svetas_settings.csv', index=False)

