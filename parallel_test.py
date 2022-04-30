import argparse
from tqdm import tqdm
import csv
import threading
from test import BAMCP_PP
import multiprocessing as mp

# create the lock
csv_writer_lock = threading.Lock()


def parallel_write(writer, run, t, arms_thetas, query_cost, T, regret, action, query_ind, r):
    with csv_writer_lock:
        writer.writerow(
            {'run': run, 'timestep': t, 'mus': arms_thetas, 'query_cost': query_cost, 'horizon': T, 'regret': regret,
             'chosen_arm': action, 'query_ind': query_ind, 'reward': r})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning_rate', type=float, default=1.,
                        metavar='')  # No learning rate according to both papers
    parser.add_argument('--discount_factor', type=float, default=0.95, metavar='')  # According to original BAMCP paper
    parser.add_argument('--query_cost', type=float, default=0.5, metavar='')  # According to BAMCP++ paper
    parser.add_argument('--exploration_const', type=float, default=5., metavar='')  # TODO: optimize?
    parser.add_argument('--max_simulations', type=int, default=100, metavar='')  # TODO: maybe can be higher?
    parser.add_argument('--arms_thetas', type=tuple, default=(0.2, 0.8), metavar='')  # According to BAMCP++ paper
    parser.add_argument('--runs', type=int, default=100, metavar='')
    parser.add_argument('--delayed_tree_expansion', type=int, default=5, metavar='')  # TODO: optimize?

    args = parser.parse_args()
    num_workers = mp.cpu_count()  # 48

    with open('./test_record.csv', 'w', newline='') as csvfile:
        fieldnames = ['run', 'timestep', 'mus', 'query_cost', 'horizon', 'regret', 'chosen_arm', 'query_ind', 'reward']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        func_args = []
        for horizon in (10, 20, 30, 40, 50):
            for query_cost in [0.5]:  #(0, 0.3, 0.5, 1, 100):
                for run in range(args.runs):
                    func_args.append((parallel_write, writer, run, horizon,
                                      args.learning_rate, args.discount_factor,
                                      query_cost,
                                      args.exploration_const, args.max_simulations,
                                      args.arms_thetas, args.delayed_tree_expansion))

        num_batches = len(func_args) // num_workers
        for batch_idx in tqdm(range(num_batches + 1)):
            threads = []
            for f_args in func_args[batch_idx * num_workers:(batch_idx + 1) * num_workers]:
                x = threading.Thread(target=BAMCP_PP, args=f_args)
                threads.append(x)
                x.start()
            for thread in threads:
                thread.join()
