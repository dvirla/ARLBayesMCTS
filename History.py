class History:
    def __init__(self):
        self.actions = []
        self.rewards = []
        self.arm_0 = {'succ': 0, 'fails': 0}
        self.arm_1 = {'succ': 0, 'fails': 0}

    def update(self, action, reward):
        """
        :param action: which arm is chosen {0, 1}
        :param reward: the reward if observed or None else. reward is one of {0, 1, None} in active two-armed bandits
        """
        assert action == 0 or action == 1
        assert reward == 0 or reward == 1 or reward is None
        if action == 0:
            if reward == 1:
                self.arm_0['succ'] += 1
            elif reward == 0:
                self.arm_0['fails'] += 1
        else:
            if reward == 1:
                self.arm_1['succ'] += 1
            elif reward == 0:
                self.arm_1['fails'] += 1
        self.actions.append(action)
        self.rewards.append(reward)

    def get_arm_dicts(self, arm):
        """
        :param arm: on of {0, 1} in two-armed bandits
        :return: respective dict
        """
        if arm:
            return self.arm_1
        return self.arm_0

    def __getitem__(self, idx):
        return self.actions[idx], self.rewards[idx]

    def __len__(self):
        return len(self.actions)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __key(self):
        return tuple(zip(self.actions, self.rewards))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, History):
            return self.__key() == other.__key()
        return NotImplemented