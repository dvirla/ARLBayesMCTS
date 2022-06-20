from bandits import knowledgeGradientChangingCostPolicy, runExperiment, parameterizedRegretQueryChangingCost
import argparse
import pandas as pd
import pickle


def run_multi_expiriemnt(horizon, runs):
    combs = [((0.2, 0.8), 1, 2, 0.5), ((1, 0), 1, 2, 0.5), ((0.5, 0.5), 1, 2, 0.5),
             ((1, 0), 0.5, 1, 1), ((0.2, 0.8), 0.5, 1, 1), ((0.2, 0.8), 0, 1, 1), ((1, 0), 0, 1, 1)]
    policies = ['mcch', 'kd']
    for mus, cost, increase_factor, decrease_factor in combs:
        for st_policy in policies:
            policy = knowledgeGradientChangingCostPolicy if st_policy == 'kd' else parameterizedRegretQueryChangingCost
            runs_list, timesteps, arms_thetas, base_query_cost, query_costs, horizons, chosen_arms, query_inds, rewards = runExperiment(
                mus, horizon, cost, increase_factor,
                decrease_factor,
                policy,
                runs, progressbar=True)

            to_df = {'run': runs_list, 'timestep': timesteps, 'mus': arms_thetas, 'base_query_cost': base_query_cost,
                     'query_cost': query_costs, 'horizon': horizons, 'chosen_arm': chosen_arms, 'query_ind': query_inds,
                     'reward': rewards}
            df = pd.DataFrame(to_df)

            writer_policy = st_policy
            if increase_factor != 1:
                folder = '/changing_cost'
                writer_cost = 'changing_cost'
            else:
                folder = '/fixed_cost'
                writer_cost = '05_cost' if cost != 0 else 'no_cost'

            writer_path = './{0}_records{1}/{2}_{3}_{4}_horizon_{5}_arms.csv'.format(writer_policy, folder, writer_policy, writer_cost,
                                                                                    horizon,
                                                                                    '_'.join([str.rstrip(
                                                                                        ''.join(str(x).split('.')), '0')
                                                                                        for x in mus]))

            df.to_csv(writer_path, index=False)
            with open(writer_path.split('.csv')[0] + '.pkl', 'wb') as f:
                pickle.dump({'mus': mus, 'cost': cost, 'inc_factor': increase_factor, 'dec_factor': decrease_factor}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--base_query_cost', type=float, default=1., metavar='')
    parser.add_argument('--horizon', type=int, default=500, metavar='')
    parser.add_argument('--arms_thetas', nargs='+', type=float, default=[0.2, 0.8], metavar='')
    parser.add_argument('--runs', type=int, default=100, metavar='')
    parser.add_argument('--increase_factor', type=float, default=2., metavar='')
    parser.add_argument('--decrease_factor', type=float, default=0.5, metavar='')
    parser.add_argument('--mcch', action='store_true')
    parser.add_argument('--multi_exp', action='store_true')

    args = parser.parse_args()
    horizon = args.horizon
    runs = args.runs

    if args.multi_exp:
        run_multi_expiriemnt(horizon, runs)

    else:
        mus = args.arms_thetas
        cost = args.base_query_cost
        increase_factor = args.increase_factor
        decrease_factor = args.decrease_factor
        policy = knowledgeGradientChangingCostPolicy
        if args.mcch:
            policy = parameterizedRegretQueryChangingCost

        runs_list, timesteps, arms_thetas, base_query_cost, query_costs, horizon, chosen_arms, query_inds, rewards = runExperiment(
            mus, horizon, cost, increase_factor,
            decrease_factor,
            policy,
            runs, progressbar=True)

        to_df = {'run': runs, 'timestep': timesteps, 'mus': arms_thetas, 'base_query_cost': base_query_cost,
                 'query_cost': query_costs, 'horizon': horizon, 'chosen_arm': chosen_arms, 'query_ind': query_inds,
                 'reward': rewards}
        df = pd.DataFrame(to_df)

        writer_policy = 'mcch' if args.mcch else 'kd'
        folder = '/changing_cost' if increase_factor != 1 else '/fixed_cost'
        writer_cost = '05_cost' if args.base_query_cost != 0 else 'no_cost'

        writer_path = './kd_records{0}/{1}_{2}_{3}_horizon_{4}_arms.csv'.format(folder, writer_policy, writer_cost,
                                                                                args.horizon,
                                                                                '_'.join([str.rstrip(
                                                                                    ''.join(str(x).split('.')), '0')
                                                                                          for x in args.arms_thetas]))

        df.to_csv(writer_path, index=False)
        with open(writer_path.split('.csv')[0] + '.pkl', 'wb') as f:
            pickle.dump(args, f)
