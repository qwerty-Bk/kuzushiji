import os
import wandb


class Logger:
    def __init__(self, log_type='wandb'):
        super(Logger, self).__init__()
        self.log_type = log_type
        if log_type == 'wandb':
            if os.path.exists('wandb.txt'):
                with open('wandb.txt', 'r') as f:
                    for line in f:
                        key = line
                        break
                wandb.login(key=key)
            wandb.init(project='dl_final')
        else:
            raise NotImplementedError(f'Log type {log_type} does not exist')

    def log(self, values, step=None):
        if self.log_type == 'wandb':
            wandb.log(values)
