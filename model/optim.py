import numpy as np


class ScheduledAdam():
    def __init__(self, optimzer, hidden_dim, warm_steps):
        self.init_lr = np.power(hidden_dim, -0.5)
        self.optimzer = optimzer
        self.current_steps = 0
        self.warm_steps = warm_steps

    def step(self):
        self.current_steps += 1

        for p in self.optimzer.param_gropus:
            p['lr'] = lr

        self.optimzer.step()

    def zero_grad(self):
        self.optimzer.zero_grad()

    def get_scale(self):
        return np.min([
            np.power(self.current_steps, -0.5),
            self.current_steps * np.power(self.warm_steps, -0.5)
        ])