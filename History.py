class History:
    def __init__(self, actions=None, rewards=None, arm_0=None, arm_1=None):
        if actions is None and rewards is None and arm_0 is None and arm_1 is None:
            self.actions = []
            self.rewards = []
            self.arm_0 = {'succ': 0, 'fails': 0}
            self.arm_1 = {'succ': 0, 'fails': 0}
        elif actions is not None and rewards is not None and arm_0 is not None and arm_1 is not None:
            self.actions = actions
            self.rewards = rewards
            self.arm_0 = arm_0
            self.arm_1 = arm_1
        else:
            raise Exception("History init got unhandled inputs")

    def update(self, action, reward):
        """
        :param action: which arm is chosen {0, 1}
        :param reward: the reward if observed or None else. reward is one of {0, 1, None} in active two-armed bandits
        """
        assert action == 0 or action == 1
        assert reward == 0 or reward == 1 or reward is None
        actions, rewards, arm_0, arm_1 = self.actions.copy(), self.rewards.copy(), self.arm_0.copy(), self.arm_1.copy()
        if action == 0:
            if reward == 1:
                arm_0['succ'] += 1
            elif reward == 0:
                arm_0['fails'] += 1
        else:
            if reward == 1:
                arm_1['succ'] += 1
            elif reward == 0:
                arm_1['fails'] += 1
        actions.append(action)
        rewards.append(reward)
        return History(actions, rewards, arm_0, arm_1)

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