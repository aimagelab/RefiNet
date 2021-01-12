import os
import torch
import torch.nn as nn

class ModuleUtilizer(object):
    """Module utility class

    Attributes:
        configer (Configer): Configer object, contains procedure configuration.

    """
    def __init__(self, configer):
        """Class constructor for Module utility"""
        self.configer = configer

    def update_optimizer(self, net, iters):
        """Load optimizer and adjust learning rate during training, if using SGD.

                Args:
                    net (torch.nn.Module): Module in use
                    iters (int): current iteration number

                Returns:
                    optimizer (torch.optim.optimizer): PyTorch Optimizer
                    lr (float): Learning rate for training procedure

        """
        optim = self.configer.get('solver', 'type')
        decay = self.configer.get('solver', 'weight_decay')

        if optim == "Adam":
            print("Using Adam.")
            lr = self.configer.get('solver', 'base_lr')
            # Watch weight decay, it's proved that with Adam in reality is L1_Regularization -> arxiv:1711.05101
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,
                                         weight_decay=decay)
        elif optim == "SGD":
            print("Using SGD")
            policy = self.configer.get('solver', 'lr_policy')

            if policy == 'fixed':
                lr = self.configer.get('solver', 'base_lr')

            elif policy == 'step':
                gamma = self.configer.get('solver', 'gamma')
                ratio = gamma ** (iters // self.configer.get('solver', 'step_size'))
                lr = self.configer.get('solver', 'base_lr') * ratio

            elif policy == 'exp':
                lr = self.configer.get('solver', 'base_lr') * (self.configer.get('solver', 'gamma') ** iters)

            elif policy == 'inv':
                power = -self.configer.get('solver', 'power')
                ratio = (1 + self.configer.get('solver', 'gamma') * iters) ** power
                lr = self.configer.get('solver', 'base_lr') * ratio

            elif policy == 'multistep':
                lr = self.configer.get('solver', 'base_lr')
                for step_value in self.configer.get('solver', 'stepvalue'):
                    if iters >= step_value:
                        lr *= self.configer.get('solver', 'gamma')
                    else:
                        break
            else:
                raise NotImplementedError('Policy:{} is not valid.'.format(policy))

            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr = lr,
                                        momentum=self.configer.get('solver', 'momentum'), weight_decay=decay)

        else:
            raise NotImplementedError('Optimizer: {} is not valid.'.format(optim))

        return optimizer, lr

    def load_net(self, net):
        """Loading net method. If resume is True load from provided checkpoint, if False load new DataParallel

                Args:
                    net (torch.nn.Module): Module in use

                Returns:
                    net (torch.nn.DataParallel): Loaded Network module
                    iters (int): Loaded current iteration number, 0 if Resume is False
                    epoch (int): Loaded current epoch number, 0 if Resume is False
                    optimizer (torch.nn.optimizer): Loaded optimizer state, None if Resume is False

        """
        net = nn.DataParallel(net, device_ids=self.configer.get('gpu')).cuda()
        iters = 0
        epoch = 0
        optimizer = None
        if self.configer.get('resume') is not None:
            checkpoint_dict = torch.load(self.configer.get('resume'))
            load_dict = dict()
            for key, value in checkpoint_dict['state_dict'].items():
                if key.split('.')[0] == 'module':
                    # Load older weight version
                    if key.split('.')[1] == 'CPM_fe':
                        new_k = 'module.model0.CPM.{}.{}'.format(key.split('.')[-2], key.split('.')[-1])
                        load_dict[new_k] = checkpoint_dict['state_dict'][key]
                    else:
                        load_dict[key] = checkpoint_dict['state_dict'][key]
                else:
                    load_dict['module.{}'.format(key)] = checkpoint_dict['state_dict'][key]
            net.load_state_dict(load_dict, strict=False)
            iters = checkpoint_dict['iter']
            optimizer = checkpoint_dict['optimizer'] if 'optimizer' in checkpoint_dict else None
            epoch = checkpoint_dict['epoch'] if 'epoch' in checkpoint_dict else None
        return net, iters, epoch, optimizer

    def save_net(self, net, optimizer, iters, epoch):
        """Saving net state method.

                Args:
                    net (torch.nn.Module): Module in use
                    optimizer (torch.nn.optimizer): Optimizer state to save
                    iters (int): Current iteration number to save
                    epoch (int): Current epoch number to save

        """
        state = {
            'iter': iters,
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        checkpoints_dir = self.configer.get('checkpoints', 'save_dir')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        if self.configer.get("checkpoints", "best") is True:
            latest_name = 'best_{}.pth'.format(self.configer.get('checkpoints', 'save_name'))
        else:
            latest_name = '{}_{}.pth'.format(self.configer.get('checkpoints', 'save_name'), epoch)
        torch.save(state, os.path.join(checkpoints_dir, latest_name))
