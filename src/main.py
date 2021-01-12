import argparse

from src.train import PoseRefine as Refine
from src.test import PoseTest as Test
from src.test_baracca import PoseTest as Test_baracca
from src.utils.configer import Configer

import random
import numpy as np
import torch

SEED = 2931389
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.enabled = True  # Enables cudnn
    torch.backends.cudnn.benchmark = True  # It should improve runtime performances when batch shape is fixed. See https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    torch.backends.cudnn.deterministic = True  # To have ~deterministic results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypes', default=None, type=str,
                        dest='hypes', help='The file of the hyper parameters.')
    parser.add_argument('--phase', default='train', type=str,
                        dest='phase', help='The phase of module.')
    parser.add_argument('--gpu', default=[0, ], nargs='+', type=int,
                        dest='gpu', help='The gpu used.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='resume', help='The path of pretrained model.')
    parser.add_argument('--base_lr', default=None, type=float,
                        dest='base_lr', help='The learning rate.')
    parser.add_argument('--save_pkl', default=None, action="store_true",
                        dest='save_pkl', help='Save pkl.')
    parser.add_argument('--disable_from_gt', default=None, action="store_false",
                        dest='from_gt', help='Disable from_gt.')
    parser.add_argument('--fill_depth', default=None, type=bool,
                        dest='fill_depth', help='Filling depth zeros.')
    parser.add_argument('--data_aug_2D', default=None, type=bool,
                        dest='data_aug_2D', help='Add data aug on 2D before process z value.')
    parser.add_argument('--overlap_prob', default=None, type=float,
                        dest='overlap_prob', help='Overlapping probability.')
    parser.add_argument('--swap_prob', default=None, type=float,
                        dest='swap_prob', help='Swapping probability.')
    parser.add_argument('--attention', default=None, action="store_true",
                        dest='attention', help='Enable attention.')
    parser.add_argument('--bigskip', default=None, action="store_true",
                        dest='bigskip', help='Enable bigskip.')

    # print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])

    args = parser.parse_args()
    torch.autograd.set_detect_anomaly(True)
    configer = Configer(args)
    if configer.get('phase') == 'train':
        model = Refine(configer)
        model.init_model()
        model.train()
    elif configer.get('phase') == 'test':
        if configer["dataset"].lower() == "baracca":
            model = Test_baracca(configer)
            model.init_model()
            model.test()
        else:
            model = Test(configer)
            model.init_model()
            model.test()
