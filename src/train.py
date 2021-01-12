import math
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision

from torch.utils.data import DataLoader

# Import Datasets
from src.Datasets.ITOP import ITOP
from src.Datasets.transforms import Compose, GaussianNoise

# Import Model
from src.models.refinement import LinearModel
from src.models.refine_patch_2d import Patch_2D_Model
from src.models.pointnet import PointPatch
from src.models.module_utilizer import ModuleUtilizer
from torch.optim.lr_scheduler import MultiStepLR

# Import loss
from src.utils.loss import vitruvian_loss, mse_masked, forcing_loss

# Import Utils
import h5py
import scipy.io
from tensorboardX import SummaryWriter
from src.utils.average_meter import AverageMeter
from tqdm import tqdm
from src.utils.normalization import MEAN_itop, STD_itop, MEAN_patch_depth, STD_patch_depth
from src.utils.utils_3d import depth_to_world, world_to_depth
from src.utils.visualization import point_on_image

# Import Metrics
from src.utils.metrics import OKS, Metric_ITOP


# Setting seeds
def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)

class PoseRefine(object):
    """Human Pose Estimation Refine Train class

    Attributes:
        configer (Configer): Configer object, contains procedure configuration.
        train_loader (torch.utils.data.DataLoader): Train data loader variable
        val_loader (torch.utils.data.DataLoader): Val data loader variable
        test_loader (torch.utils.data.DataLoader): Test data loader variable
        net (torch.nn.Module): Network used for the current procedure
        lr (int): Learning rate value
        optimizer (torch.nn.optim.optimizer): Optimizer for training procedure
        iters (int): Starting iteration number, not zero if resuming training
        epoch (int): Starting epoch number, not zero if resuming training
        scheduler (torch.optim.lr_scheduler): Scheduler to utilize during training

    """

    def __init__(self, configer):
        self.configer = configer

        # Input and output size
        self.input_size = configer.get('data', 'input_size')    #: int: Size of the input
        self.output_size = configer.get('data', 'output_size')  #: int: Size of the output

        self.data_path = configer.get("train_dir")              #: str: Path to data directory

        # Losses
        self.train_losses = AverageMeter()                      #: Train loss avg meter
        self.val_losses = AverageMeter()                        #: Val loss avg meter
        self.test_losses = AverageMeter()                       #: Test loss avg meter
        self.v_loss_train = AverageMeter()                      #: loss avg meter
        self.v_loss_val = AverageMeter()                        #: loss avg meter
        self.v_loss_test = AverageMeter()                       #: loss avg meter
        self.forcing_loss_train = AverageMeter()                #: loss avg meter
        self.forcing_loss_val = AverageMeter()                  #: loss avg meter
        self.forcing_loss_test = AverageMeter()                 #: loss avg meter

        # mAP
        self.train_map = AverageMeter()                         #: Train mAP avg meter
        self.val_map = AverageMeter()                           #: Val mAP avg meter
        self.test_map = AverageMeter()                          #: Test mAP avg meter

        # DataLoaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Module load and save utility
        self.model_utility = ModuleUtilizer(self.configer)      #: Model utility for load, save and update optimizer
        self.net = None
        self.lr = None

        # Training procedure
        self.optimizer = None
        self.iters = None
        self.epoch = 0

        # Tensorboard and Metrics
        self.loss_summary = SummaryWriter("{}/{}".format(
            configer.get('checkpoints', 'tb_path'),
            configer.get('checkpoints', 'save_name')))          #: Summary Writer to plot data with TensorboardX
        self.loss_summary.add_text('parameters', str(self.configer).replace("\n", "\n\n"))
        self.save_iters = self.configer.get('checkpoints', 'save_iters')    #: int: Saving ratio
        self.bestmAP = 0.0      #: Best mAP achieved on the validation for current epoch

        # Other useful data
        self.side = self.configer["side"]                       #: str: View for choosing side-view or top-view in ITOP
        self.type = self.configer.get("data", "type")           #: str: Type of data (2D, 3D, zaxis)
        self.dataset = self.configer.get("dataset").lower()     #: str: Type of dataset
        self.mean = 0.0                                         #: int: Mean value for normalization purpose
        self.std = 0.0                                          #: int: Std value for normalization purpose
        self.ids_train = None                                   #: list of str: List of ids for the train set
        self.ids_test = None                                    #: list of str: List of ids for the test set
        self.scheduler = None

    def init_model(self):
        """Initialize model and other data for procedure"""
        # Selecting activation function
        act = self.configer.get('network', 'activation')
        if act == 'ReLU' or act == 'Relu' or act == 'relu' or act == 'ReLu':
            activation = torch.nn.ReLU
        elif act == 'Tanh' or act == 'tanh' or act == 'tan':
            activation = torch.nn.Tanh
        elif act == 'PReLU' or act == 'PRelu' or act == 'prelu' or act == 'PReLu' or act == 'Prelu':
            activation = torch.nn.PReLU
        elif act == 'MySigmoid' or act == 'mysigmoid':
            activation = MySigmoid
        else:
            raise NotImplementedError(f"Wrong activation function: {act}")

        # Selecting correct model and normalization variable based on type variable
        if self.type == "base":
            # Linear model for base type
            self.net = LinearModel(
                self.input_size[0] * self.input_size[1],
                self.output_size[0] * self.output_size[1],
                self.configer.get('network', 'linear_size'),
                self.configer.get('network', 'dropout'),
                self.configer.get('network', 'batch_norm'),
                self.configer.get('network', 'residual'),
                activation,
                self.configer.get('network', 'attention') if self.configer.get('network', 'attention') is not None else False,
                self.configer.get('network', 'bigskip') if self.configer.get('network', 'bigskip') is not None else False
            )
        elif self.type == "depth":
            self.net = Patch_2D_Model(self.configer.get("data", "output_size"), activation)
        elif self.type == "pcloud":
            self.net = PointPatch(True, True, 0.2)
        else:
            raise NotImplementedError("Type: {} not implemented yet".format(self.type))

        # Initializing training
        self.iters = 0
        self.epoch = None
        phase = self.configer.get('phase')

        # Starting or resuming procedure
        if phase == 'train':
            self.net, self.iters, self.epoch, optim_dict = self.model_utility.load_net(self.net)
        else:
            raise ValueError('Phase: {} is not valid.'.format(phase))

        if self.epoch is None:
            self.epoch = 0

        # ToDo Restore optimizer and scheduler from checkpoint
        self.optimizer, self.lr = self.model_utility.update_optimizer(self.net, self.iters)
        self.scheduler = MultiStepLR(self.optimizer, self.configer["solver", "decay_steps"], gamma=0.1)

        #  Resuming training, restoring optimizer value
        if optim_dict is not None:
            print("Resuming training from epoch {}.".format(self.epoch))
            self.optimizer.load_state_dict(optim_dict)

        # Selecting Dataset and DataLoader
        if self.dataset == "itop":
            Dataset = ITOP
            if self.type in ("base", "depth") and self.configer["metrics", "kpts_type"] != "rwc":
                if self.configer.get("data", "from_gt") is not True:
                    transform = Compose([GaussianNoise(self.configer.get("data", "image_size"),
                                                       sigma=self.configer.get("data_aug", "sigma")), ])
                else:
                    transform = None
            else:
                transform = None

            # Selecting correct normalization values
            if self.type == "base":
                self.mean = MEAN_itop
                self.std = STD_itop
                self.alpha = self.configer["network", "alpha"]
                self.beta = self.configer["network", "beta"]
                self.gamma = self.configer["network", "gamma"]
            elif self.type == "depth":
                self.mean = MEAN_patch_depth
                self.std = STD_patch_depth
            elif self.type == "pcloud":
                self.mean = 0
                self.std = 1
            else:
                raise NotImplementedError("Type: {} not implemented yet".format(self.type))

            # If itop and 2D I need ids to retrieve real world coordinates for metric evaluation
            self.ids_train = h5py.File(f"{self.data_path}ITOP/ITOP_{self.side}_train_labels.h5", 'r')['id']
            self.ids_test = h5py.File(f"{self.data_path}ITOP/ITOP_{self.side}_test_labels.h5", 'r')['id']
            self.ids_train = [str(el) for el in self.ids_train]
            self.ids_test = [str(el) for el in self.ids_test]
        else:
            raise NotImplementedError(f"Dataset not supported: {self.configer.get('dataset')}")

        # Setting Dataloaders
        self.train_loader = DataLoader(
            Dataset(self.configer, split="train", base_transform=transform),
            self.configer.get('data', 'batch_size'), shuffle=True, drop_last=True,
            num_workers=self.configer.get('solver', 'workers'), pin_memory=True, worker_init_fn=worker_init_fn)
        self.val_loader = DataLoader(
            Dataset(self.configer, split="val"),
            self.configer.get('data', 'batch_size'), shuffle=False,
            num_workers=self.configer.get('solver', 'workers'), pin_memory=True, worker_init_fn=worker_init_fn)
        self.test_loader = DataLoader(
            Dataset(self.configer, split="test"),
            self.configer.get('data', 'batch_size'), shuffle=False,
            num_workers=self.configer.get('solver', 'workers'), pin_memory=True, worker_init_fn=worker_init_fn)

    def __train(self):
        """Train function for every epoch."""
        oks = OKS(n_joints=self.configer.get("data", "num_keypoints"),
                  sigmas=self.configer.get("metrics", "sigmas"))
        itop_met = Metric_ITOP(n_joints=self.configer.get("data", "num_keypoints"),
                               thresh=self.configer.get("metrics", "dist_thresh"))

        mini_batch_size = self.configer.get("solver", "max_joints_batch_size")

        self.net.train()
        for data_tuple in tqdm(self.train_loader):

            inputs = data_tuple[0].cuda()
            gt = data_tuple[1].cuda()

            # I not always need mask
            mask = data_tuple[2].cuda()
            kpts = data_tuple[3].cuda()

            self.optimizer.zero_grad()

            output, loss = self.__forward_backward(inputs, gt, mask, kpts, self.train_losses,
                                                   self.v_loss_train, self.forcing_loss_train)

            if self.type == "base":
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)
            self.optimizer.step()

            # Update iters and tensorboard graph
            self.iters += 1
            if self.iters % self.save_iters == 0:
                self.loss_summary.add_scalar('loss_training', self.train_losses.avg, self.iters)
                self.train_losses.reset()
                if self.type == "base":
                    self.loss_summary.add_scalar('loss_v_training', self.v_loss_train.avg, self.iters)
                    self.loss_summary.add_scalar('loss_f_training', self.forcing_loss_train.avg, self.iters)
                    self.v_loss_train.reset()
                    self.forcing_loss_train.reset()

                # Preparing and De-normilize data only when plotting results, not every iteration
                visible = data_tuple[2].cpu().numpy()
                ids = data_tuple[4].cpu().numpy()
                if self.type.lower() == "depth" or self.type.lower() == "pcloud" \
                        or (self.type.lower() == "base" and self.configer.get("offset") is True):
                    kpts = data_tuple[3].cpu().numpy()
                    kpts_off = output.cpu().detach().numpy().astype(np.float32)
                    gt_off = gt.cpu().detach().numpy().astype(np.float32)
                    kpts_gt = kpts - gt_off
                    kpts_pred = kpts - kpts_off
                else:
                    kpts_pred = output.cpu().detach().numpy().astype(np.float32)
                    kpts_gt = gt.cpu().detach().numpy().astype(np.float32)

                self.update_metrics(kpts_pred, kpts_gt, visible, "train", ids, oks, itop_met)

        mAP, cm_dist = itop_met.get_values()
        _, mAP_oks = oks.get_metrics()
        self.loss_summary.add_scalar('accuracy_itop_training', mAP, self.iters)
        self.loss_summary.add_scalar('cm_training', cm_dist,
                                     self.iters)
        self.loss_summary.add_scalar('mAP_oks_training', mAP_oks, self.iters)

    def __val(self):
        """Validation function."""
        self.net.eval()
        # Taking OKS and ITOP metrics for evaluation
        oks = OKS(n_joints=self.configer.get("data", "num_keypoints"),
                  sigmas=self.configer.get("metrics", "sigmas"))
        itop_met = Metric_ITOP(n_joints=self.configer.get("data", "num_keypoints"),
                               thresh=self.configer.get("metrics", "dist_thresh"))

        mini_batch_size = self.configer.get("solver", "max_joints_batch_size")

        with torch.no_grad():
            for i, data_tuple in enumerate(tqdm(self.val_loader)):
                """
                input, gt, visible
                """
                inputs = data_tuple[0].cuda()
                gt = data_tuple[1].cuda()

                # I not always need mask
                mask = data_tuple[2].cuda()
                kpts = data_tuple[3].cuda()

                output, loss = self.__forward(inputs, gt, mask, kpts, self.val_losses, self.v_loss_val, self.forcing_loss_val)

                # Preparing and De-normilize data only when plotting results, not every iteration
                visible = data_tuple[2].cpu().numpy()
                kpts = data_tuple[3].cpu().numpy()
                ids = data_tuple[4].cpu().numpy()
                if self.type.lower() == "depth" or self.type.lower() == "pcloud" \
                        or (self.type.lower() == "base" and self.configer.get("offset") is True):
                    kpts = data_tuple[3].cpu().numpy()
                    kpts_off = output.cpu().detach().numpy().astype(np.float32)
                    gt_off = gt.cpu().detach().numpy().astype(np.float32)
                    kpts_gt = kpts - gt_off
                    kpts_pred = kpts - kpts_off
                else:
                    kpts = data_tuple[3].cpu().numpy()
                    kpts_pred = output.cpu().detach().numpy().astype(np.float32)
                    kpts_gt = gt.cpu().detach().numpy().astype(np.float32)
                self.update_metrics(kpts_pred, kpts_gt, visible, "val", ids, oks, itop_met)

                # Plot data on tensorboard only for the first batch
                if i == 0:
                    self.save_images(ids, kpts_pred, kpts_gt, kpts, "val", self.epoch)

            # Printing Metrics
            print(itop_met)
            print(oks)

            mAP, cm_dist = itop_met.get_values()
            _, mAP_oks = oks.get_metrics()
            self.loss_summary.add_scalar('accuracy_itop_validation', mAP, self.epoch + 1)
            self.loss_summary.add_scalar('cm_validation', cm_dist,
                                         self.epoch + 1)
            self.loss_summary.add_scalar('mAP_oks_validation', mAP_oks, self.epoch + 1)

            self.loss_summary.add_scalar('loss_validation', self.val_losses.avg, self.iters)
            self.val_losses.reset()
            if self.type == "base":
                self.loss_summary.add_scalar('loss_v_validation', self.v_loss_val.avg, self.iters)
                self.loss_summary.add_scalar('loss_f_validation', self.forcing_loss_val.avg, self.iters)
                self.v_loss_val.reset()
                self.forcing_loss_val.reset()

            if mAP > self.bestmAP:
                # Saving net and testing only if the mAP calculated on the current validation is the best until now
                self.model_utility.save_net(self.net, self.optimizer, self.iters, self.epoch + 1)
                self.bestmAP = mAP
                self.__test()

    def __test(self):
        """Test function."""
        self.net.eval()
        # Taking OKS and ITOP metrics for evaluation
        oks = OKS(n_joints=self.configer.get("data", "num_keypoints"),
                  sigmas=self.configer.get("metrics", "sigmas"))
        itop_met = Metric_ITOP(n_joints=self.configer.get("data", "num_keypoints"),
                               thresh=self.configer.get("metrics", "dist_thresh"))
        itop_met2 = Metric_ITOP(n_joints=self.configer.get("data", "num_keypoints"),
                               thresh=self.configer.get("metrics", "dist_thresh") // 2)

        mini_batch_size = self.configer.get("solver", "max_joints_batch_size")

        with torch.no_grad():
            for i, data_tuple in enumerate(tqdm(self.test_loader)):
                """
                input, gt, visible
                """
                inputs = data_tuple[0].cuda()
                gt = data_tuple[1].cuda()

                # I not always need mask
                mask = data_tuple[2].cuda()
                kpts = data_tuple[3].cuda()

                output, loss = self.__forward(inputs, gt, mask, kpts, self.test_losses, self.v_loss_test, self.forcing_loss_test)

                # Preparing and De-normalize data only when plotting results, not every iteration
                visible = data_tuple[2].cpu().numpy()
                kpts = data_tuple[3].cpu().numpy()
                ids = data_tuple[4].cpu().numpy()
                if self.type.lower() in ("depth", "pcloud") \
                        or (self.type.lower() == "base" and self.configer.get("offset") is True):
                    kpts = data_tuple[3].cpu().numpy()
                    kpts_off = output.cpu().detach().numpy().astype(np.float32)
                    gt_off = gt.cpu().detach().numpy().astype(np.float32)
                    kpts_gt = kpts - gt_off
                    kpts_pred = kpts - kpts_off
                else:
                    kpts_pred = output.cpu().detach().numpy().astype(np.float32)
                    kpts_gt = gt.cpu().detach().numpy().astype(np.float32)
                self.update_metrics(kpts_pred, kpts_gt, visible, "test", ids, oks, itop_met, itop_met2)

                # Plot data on tensorboard only for the first batch
                if i == 0:
                    self.save_images(ids, kpts_pred, kpts_gt, kpts, "test", self.epoch)

            # Printing Metrics
            print(itop_met)
            print(oks)
            print("Metric on 5cm thresh")
            print(itop_met2)

            mAP, cm_dist = itop_met.get_values()
            _, mAP_oks = oks.get_metrics()
            self.loss_summary.add_scalar('accuracy_itop_test', mAP, self.epoch + 1)
            self.loss_summary.add_scalar('cm_test', cm_dist,
                                         self.epoch + 1)
            self.loss_summary.add_scalar('mAP_oks_test', mAP_oks, self.epoch + 1)

            self.loss_summary.add_scalar('loss_test', self.test_losses.avg, self.iters)
            self.test_losses.reset()
            if self.type == "base":
                self.loss_summary.add_scalar('loss_v_test', self.v_loss_test.avg, self.iters)
                self.loss_summary.add_scalar('loss_f_test', self.forcing_loss_test.avg, self.iters)
                self.v_loss_test.reset()
                self.forcing_loss_test.reset()

    def __forward(self, inputs, gt, mask, kpts, avgm_loss, avgm_v_loss=None, avgm_f_loss=None):
        """Execute forward pass through the network

        Args:
            inputs (torch.Tensor): Input data
            gt (torch.Tensor): Ground truth data
            mask (torch.Tensor): Binary visibility mask
            kpts (torch.Tensor): Input kpts data
            avgm_loss (AverageMeter): MSE loss accumulator
            avgm_v_loss (AverageMeter, optional): VITRUVIAN loss accumulator
            avgm_f_loss (AverageMeter, optional): FORCING loss accumulator

        Returns:
            output (torch.Tensor): Output data
            loss (torch.Tensor): Output loss

        """
        # Forward pass.
        output = self.net(inputs)

        # Update loss
        if self.type == "base":
            loss_mse = mse_masked(output, gt, mask) * self.alpha
            if self.configer.get("offset") is True:
                kpts_gt = kpts - gt
                kpts_pred = kpts - output
            else:
                kpts_gt = gt
                kpts_pred = output
            loss_v = vitruvian_loss(kpts_pred, mask, self.dataset) * self.beta
            loss_forced = forcing_loss(kpts_pred, kpts_gt, mask, self.dataset) * self.gamma
            avgm_v_loss.update(loss_v.item(), inputs.size(0))
            avgm_f_loss.update(loss_forced.item(), inputs.size(0))
            loss = loss_mse + loss_v + loss_forced
        else:
            loss = mse_masked(output, gt, mask)

        # Update logs
        avgm_loss.update(loss.item(), inputs.size(0))

        return output, loss

    def __forward_backward(self, inputs, gt, mask, kpts, avgm_loss, avgm_v_loss=None, avgm_f_loss=None):
        """Call _forward to execute forward, then execute backward on returned values

        Returns:
            output (torch.Tensor): Output data
            loss (torch.Tensor): Output loss

        """
        output, loss = self.__forward(inputs, gt, mask, kpts, avgm_loss, avgm_v_loss, avgm_f_loss)

        # Backward
        loss.backward()
        return output, loss

    def train(self):
        """Training procedure, execute a training phase followed by a validation.
        Increasing epoch and scheduler step
        """
        cudnn.benchmark = True
        epoch = self.configer.get("epochs")
        while self.epoch < epoch + 1:
            print("Starting epoch: {}".format(self.epoch + 1))
            self.__train()
            self.__val()
            self.scheduler.step()
            print("Ending epoch {}".format(self.epoch + 1))
            self.epoch += 1

    def update_metrics(self, kpts, kpts_gt=None, visible=None, split: str = "",
                       ids=None, oks: OKS = None, accuracy: Metric_ITOP = None, additional = None):
        """Funtion to update metrics

        Args:
            kpts (np.ndarray): Current data to update
            kpts_gt (np.ndarray): Ground truth relative to current update
            visible (np.ndarray, optional): Visible binary mask
            split (np.ndarray, optional): Split type, train test or val
            ids (list of str, optional): ids of the current kpts to update
            oks (OKS, optional): oks metric variable
            accuracy (Metric_ITOP, optional): accuracy metric variable

        """
        # Update metrics with new inference
        if self.configer.get("metrics", "kpts_type").lower() == "3d":
            if self.dataset == "itop":
                kpts_2d = world_to_depth(kpts)
                kpts_gt_2d = world_to_depth(kpts_gt)
            kpts_3d = kpts
            kpts_gt_3d = kpts_gt
        elif self.configer.get("metrics", "kpts_type").lower() == "2d":
            kpts_2d = kpts
            kpts_gt_2d = kpts_gt
            if self.dataset == "itop":
                kpts_3d = np.zeros((kpts.shape[0], 15, 3))
                kpts_gt_3d = np.zeros((kpts_gt.shape[0], 15, 3))
                if split == "train":
                    name = [self.train_loader.dataset.ids[el] for el in ids]
                    ids_label = self.ids_train
                    imgs = h5py.File(f"{self.data_path}ITOP/ITOP_{self.side}_train_depth_map.h5", 'r')['data']
                elif split == "val":
                    name = [self.val_loader.dataset.ids[el] for el in ids]
                    ids_label = self.ids_train
                    imgs = h5py.File(f"{self.data_path}ITOP/ITOP_{self.side}_train_depth_map.h5", 'r')['data']
                elif split == "test":
                    name = [self.test_loader.dataset.ids[el] for el in ids]
                    ids_label = self.ids_test
                    imgs = h5py.File(f"{self.data_path}ITOP/ITOP_{self.side}_test_depth_map.h5", 'r')['data']
                else:
                    raise ValueError("Split: {} not recognized".format(split))
                for i, name in enumerate(name):
                    index = int(np.where(np.array(ids_label) == name)[0])
                    depth = imgs[index] * 1000
                    kpts_3d[i] = depth_to_world(kpts[i], depth)
                    kpts_gt_3d[i] = depth_to_world(kpts_gt[i], depth)
            else:
                raise NotImplementedError("Got 2D kpts metrics not in itop, need to be implemented!")
        elif self.configer.get("metrics", "kpts_type").lower() == "rwc":
            kpts_2d = kpts[..., :2]
            kpts_gt_2d = kpts_gt[..., :2]
            kpts_3d = np.zeros((kpts.shape[0], 15, 3))
            kpts_gt_3d = np.zeros((kpts_gt.shape[0], 15, 3))
            kpts_3d[..., 2] = kpts[..., 2]
            kpts_gt_3d[..., 2] = kpts_gt[..., 2]
            kpts_3d[..., 0] = (kpts[..., 0] - 160) * 0.0035 * kpts[..., 2]
            kpts_3d[..., 1] = -(kpts[..., 1] - 120) * 0.0035 * kpts[..., 2]

            kpts_gt_3d[..., 0] = (kpts_gt[..., 0] - 160) * 0.0035 * kpts_gt[..., 2]
            kpts_gt_3d[..., 1] = -(kpts_gt[..., 1] - 120) * 0.0035 * kpts_gt[..., 2]
        else:
            raise NotImplementedError(
                "Not implemented metric type: {}".format(self.configer.get("metrics", "kpts_type")))
        oks.eval(kpts_2d, kpts_gt_2d)
        accuracy.eval(kpts_3d, kpts_gt_3d, visible)
        if additional is not None:
            additional.eval(kpts_3d, kpts_gt_3d, visible)

    def save_images(self, ids, kpts, kpts_gt, noisy_kpts, split, step: int = 0):
        """
        Creates a grid of images with gt joints and a grid with predicted joints.
        This is a basic function for debugging purposes only.
        The grid will be written in the SummaryWriter with name "{prefix}_images" and
        "{prefix}_predictions".
        Args:
            ids (np.array): a list of images ids to be loaded.
            kpts (torch.Tensor): a tensor of predicted joints with shape (batch x joints x 2).
            kpts_gt (torch.Tensor): a tensor of gt joints with shape (batch x joints x 2).
            split (str): split type, train, val or test
            step (int): summary_writer step.
                Default: 0
        Returns:
            A pair of images which are built from torchvision.utils.make_grid
        """
        # We only need to print first 4 images
        if ids.shape[0] > 4:
            ids = ids[:4]
            kpts = kpts[:4]
            kpts_gt = kpts_gt[:4]

        # Retrieve Image from dataset
        if self.dataset == "itop":
            if split == "train":
                name = [self.train_loader.dataset.ids[el] for el in ids]
                ids_label = self.ids_train
                imgs_data = h5py.File(f"/nas/DriverMonitoring/Datasets/ITOP/ITOP_{self.side}_train_depth_map.h5", 'r')['data']
            elif split == "val":
                name = [self.val_loader.dataset.ids[el] for el in ids]
                ids_label = self.ids_train
                imgs_data = h5py.File(f"/nas/DriverMonitoring/Datasets/ITOP/ITOP_{self.side}_train_depth_map.h5", 'r')['data']
            elif split == "test":
                name = [self.test_loader.dataset.ids[el] for el in ids]
                ids_label = self.ids_test
                imgs_data = h5py.File(f"/nas/DriverMonitoring/Datasets/ITOP/ITOP_{self.side}_test_depth_map.h5", 'r')['data']
            else:
                raise ValueError("Split: {} not recognized".format(split))

            # Convert 3D annotations to 2D
            if self.configer.get("metrics", "kpts_type").lower() == "3d":
                kpts = world_to_depth(kpts)
                kpts_gt = world_to_depth(kpts_gt)

            imgs = list()
            for i, name in enumerate(name):
                index = int(np.where(np.array(ids_label) == name)[0])
                imgs.append(imgs_data[index])
        else:
            raise NotImplementedError("Dataset: {} not implemented".format(self.dataset))

        imgs_detection = np.array([point_on_image(k, el) for k, el in zip(kpts, imgs)])
        imgs_gt = np.array([point_on_image(k, el) for k, el in zip(kpts_gt, imgs)])

        grid_pred = torchvision.utils.make_grid(torch.from_numpy(imgs_detection).permute(0, 3, 1, 2).float() / 255,
                                                nrow=int(imgs_detection.shape[0] ** 0.5), padding=2,
                                                normalize=False)
        grid_gt = torchvision.utils.make_grid(torch.from_numpy(imgs_gt).permute(0, 3, 1, 2).float() / 255,
                                              nrow=int(imgs_gt.shape[0] ** 0.5), padding=2,
                                              normalize=False)

        split = "validation" if split == "val" else split
        self.loss_summary.add_image(split + '_prediction', grid_pred, global_step=step)
        self.loss_summary.add_image(split + '_gt', grid_gt, global_step=step)

        if noisy_kpts is not None:
            if self.dataset == "itop":
                if noisy_kpts.shape[-1] == 3:
                    noisy_kpts = world_to_depth(noisy_kpts / 1000)
            imgs_noise = np.array([point_on_image(k, el) for k, el in zip(noisy_kpts, imgs)])
            grid_noise = torchvision.utils.make_grid(torch.from_numpy(imgs_noise).permute(0, 3, 1, 2).float() / 255,
                                                nrow=int(imgs_detection.shape[0] ** 0.5), padding=2,
                                                normalize=False)
        self.loss_summary.add_image(split + '_input', grid_noise, global_step=step)

        return grid_gt, grid_pred
