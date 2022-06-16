## Active Reinforcement Learning with Monte-Carlo Tree Search for bandits

This project is based on the papers:
* **[Active Reinforcement Learning with Monte-Carlo Tree Search](https://arxiv.org/pdf/1803.04926.pdf)** 
* **[Efficient Bayes-Adaptive Reinforcement Learning using Sample-Based Search](https://papers.nips.cc/paper/2012/file/35051070e572e47d2c26c241ab88307f-Paper.pdf)**

### Main objective:
* Attempting to find (sub)optimal policy in a two arm bandit configuration where rewards are only visible at a cost of querying, and transition function's parameters are unknown.
* The algorithm from the mentioned paper was compared to knowledge gradient algorithm which was adjusted to the querying setting.
* After discovering that BAMCPP (the algorithm from the articles) did not outperform KD, I adjusted it to have a temperature element in its MCTS' argmax stage.

### How to use:
1. Running BAMCPP without temperature element:\
``python3 parallel_test_V2.py --base_query_cost=1 --increase_factor=2 --decrease_factor=0.5 --runs=100 --horizon=500 --exploration_const=0.0000001 --delayed_tree_expansion=10 --max_simulations=50``
2. Running BAMCPP with temperature:\
``python3 parallel_test_V2.py --base_query_cost=1 --increase_factor=2 --decrease_factor=0.5 --runs=100 --horizon=500 --exploration_const=0.0000001 --delayed_tree_expansion=10 --max_simulations=50 --use_temperature``
3. Running KD:\
``python3 parallel_test_V2.py --base_query_cost=1 --increase_factor=2 --decrease_factor=0.5 --runs=100 --horizon=500 --exploration_const=0.0000001 --delayed_tree_expansion=10 --max_simulations=50``
4. If you wish to change to cost settings, in both `parallel_test_V2.py` and `run_kd.py` there are parameters called:, `base_query_cost`, `increase_factor`, `decrease_factor`.\
   Those parameters will define the cost following this rule:\
   ``query_cost = query_ind * query_cost * increase_factor + (1 - query_ind) * max(query_cost * decrease_factor, base_query_cost)``
<br><br>