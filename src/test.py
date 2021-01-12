import os
import scipy.io
import numpy as np
import pickle
import cv2
import torch
import time

from torch.utils.data import DataLoader

# Import Datasets
from src.Datasets.ITOP import ITOP

# Import Model
from src.models.refinement import LinearModel
from src.models.refine_patch_2d import Patch_2D_Model_V1, Patch_2D_Model_V3
from src.models.pointnet import PointPatch_channel, PointPatch_batch
from src.models.module_utilizer import ModuleUtilizer

# Import Utils
from tqdm import tqdm
import h5py
from src.utils.normalization import MEAN_itop, STD_itop, MEAN_patch_depth, STD_patch_depth
from src.utils.utils_3d import depth_to_world, world_to_depth
from src.utils.visualization import point_on_image

# Import Metrics
from src.utils.metrics import OKS, Metric_ITOP
from src.utils.visualization import plot_2D_3D

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setting seeds
def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)


class PoseTest(object):
    """Human Pose Estimation Refine Test class

    Attributes:
        configer (Configer): Configer object, contains procedure configuration.
        data_loader (torch.utils.data.DataLoader): Data Loader variable
        net (torch.nn.Module): Network used for the current procedure

    """
    def __init__(self, configer):
        """Constructor function for HPE Test class

            Args:
                configer (Configer): Configer object, contains procedure configuration.

        """
        self.configer = configer

        # DataLoader
        self.data_loader = None

        # Input and output size
        self.input_size = configer.get('data', 'input_size')    #: int: Size of the input
        self.output_size = configer.get('data', 'output_size')  #: int: Size of the output

        self.data_path = configer["train_dir"]                  #: str: Path to data directory

        # Module load and save utility
        self.model_utility = ModuleUtilizer(self.configer)      #: Model utility for load, save and update optimizer
        self.net = None

        # Other useful data
        self.side = self.configer["side"]                       #: str: View for choosing side-view or top-view in ITOP
        self.result_dir = self.configer.get("data", "result_dir")  #: str: Path to result dir
        self.type = self.configer.get("data", "type")           #: str: Type of data (2D, 3D, zaxis)
        self.dataset = self.configer.get("dataset").lower()     #: str: Type of dataset
        self.mean = 0.0                                         #: int: Mean value for normalization purpose
        self.std = 0.0                                          #: int: Std value for normalization purpose
        self.ids_train = None                                   #: list of str: List of ids for the train set
        self.ids_test = None                                    #: list of str: List of ids for the test set
        self.img_saved = 0                                      #: int: Number of images saved

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
        else:
            raise NotImplementedError(f"Wrong activation function: {act}")

        if self.type == "base":
            # Linear model for base type
            self.net = LinearModel(self.input_size[0] * self.input_size[1],
                                   self.output_size[0] * self.output_size[1],
                                   self.configer.get('network', 'linear_size'),
                                   self.configer.get('network', 'dropout'),
                                   self.configer.get('network', 'batch_norm'),
                                   self.configer.get('network', 'residual'),
                                   activation)
        elif self.type == "depth":
            # 2D Depth patch model, choice based over model_name version
            if self.configer.get("network", "model_name").lower() == "v1":
                self.net = Patch_2D_Model_V1(self.configer.get("data", "output_size"), activation)
            elif self.configer.get("network", "model_name").lower() == "v3":
                self.net = Patch_2D_Model_V3(self.configer.get("data", "output_size"), activation)
            else:
                raise NotImplementedError(
                    "Model version: {} is not implemented".format(self.configer.get("network", "model_name")))
        elif self.type == "pcloud":
            if self.configer.get("network", "model_name").lower() == "v1":
                self.net = PointPatch_channel(True, True, 0.2)
            elif self.configer.get("network", "model_name").lower() == "v2":
                self.net = PointPatch_batch(True, True, 0.2)
            else:
                raise NotImplementedError(
                    "Model version: {} is not implemented".format(self.configer.get("network", "model_name")))
        else:
            raise NotImplementedError("Type: {} not implemented yet".format(self.type))

        if self.configer.get('resume') is not None:
            print("Resuming checkpoints at: {}".format(self.configer.get('resume')))
        else:
            print("Warning! You're not resuming any known checkpoints for test operations!")
        self.net, _, _, _ = self.model_utility.load_net(self.net)

        # Selecting Dataset and DataLoader
        if self.dataset == "itop":
            Dataset = ITOP

            # Selecting correct normalization values
            if self.type == "base":
                self.mean = MEAN_itop
                self.std = STD_itop
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
        self.data_loader = DataLoader(
            Dataset(self.configer, split="test"),
            1, shuffle=False,
            num_workers=self.configer.get('solver', 'workers'), pin_memory=True, worker_init_fn=worker_init_fn)

    def __test(self):
        """Test function on multiple images."""
        self.net.eval()
        # Taking OKS and ITOP metrics for evaluation
        oks = OKS(n_joints=self.configer.get("data", "num_keypoints"),
                  sigmas=self.configer.get("metrics", "sigmas"))
        itop_met = Metric_ITOP(n_joints=self.configer.get("data", "num_keypoints"),
                               thresh=self.configer.get("metrics", "dist_thresh"))
        input_met = Metric_ITOP(n_joints=self.configer.get("data", "num_keypoints"),
                                thresh=self.configer.get("metrics", "dist_thresh"))
        input_oks = OKS(n_joints=self.configer.get("data", "num_keypoints"),
                        sigmas=self.configer.get("metrics", "sigmas"))

        # Save inference time
        tot_time = 0.0
        plt.figure()

        with torch.no_grad():
            for i, data_tuple in enumerate(tqdm(self.data_loader)):
                """
                    input, gt, visible, img_index
                """
                inputs = data_tuple[0].cuda()
                gt = data_tuple[1].cuda()
                start = time.time()

                # Inference
                output = self.net(inputs)

                # Saving inference time values
                end = time.time()
                tot_time += end - start

                # Preparing and De-normalize data only when plotting results, not every iteration
                visible = data_tuple[2].cpu().numpy()
                kpts = data_tuple[3].cpu().numpy()
                ids = data_tuple[4].cpu().numpy()
                if self.type.lower() in ("depth", "pcloud") \
                        or (self.type.lower() == "base" and self.configer.get("offset") is True):
                    kpts_off = output.cpu().detach().numpy().astype(np.float32)
                    gt_off = gt.cpu().detach().numpy().astype(np.float32)
                    kpts_gt = kpts - gt_off
                    input_data = kpts.copy()
                    kpts_pred = kpts - kpts_off
                else:
                    input_data = data_tuple[0].cpu().detach().numpy().astype(np.float32)
                    input_data *= self.std
                    input_data += self.mean
                    kpts_pred = output.cpu().detach().numpy().astype(np.float32)
                    kpts_gt = gt.cpu().detach().numpy().astype(np.float32)
                self.update_metrics(kpts_pred, kpts_gt, visible, ids, oks, itop_met)
                self.update_metrics(input_data, kpts_gt, visible, ids, input_oks, input_met)

                if self.configer.get("save_pkl"):
                    for el, kpt in zip(ids, kpts_pred):
                        if self.dataset == "itop":
                            name = self.data_loader.dataset.ids[el]
                            index = self.data_loader.dataset.ids_str.index(name)
                            self.kpts_dict[index] = kpt
                        else:
                            name = self.data_loader.dataset.joint_names[el]
                            self.kpts_dict[name] = kpt

                # Saving images
                if self.configer.get("save_img") is True:
                    self.save_images(ids, kpts_pred)
                    # self.save_images(ids, input_data, visible)
                    self.img_saved += ids.shape[0]

        # Printing Metrics
        print("Resumed checkpoint:", self.configer.get("resume"))
        print("Input metrics")
        print(input_met)
        print(input_oks)
        print(f"Upper body -> {input_met.tot_up / input_met.counter_up}")
        print(f"Lower body -> {input_met.tot_down / input_met.counter_down}")
        print()
        print("Output metrics")
        print(itop_met)
        print(oks)
        print()
        print(f"Run at: [{round(tot_time / len(self.data_loader) * 1000, 4)}]ms, "
              f"or at [{(1 / (tot_time / len(self.data_loader)))}]fps")
        print(f"Upper body -> {itop_met.tot_up / itop_met.counter_up}")
        print(f"Lower body -> {itop_met.tot_down / itop_met.counter_down}")

    def test(self):
        """Testing procedure, if needed, saving kpts with output as pickle"""
        print("Starting test procedure.")
        start = time.time()
        self.kpts_dict = dict()
        self.__test()
        if self.configer.get("save_pkl") is True and self.configer.get("from_gt") is False:
            save_name = self.configer.get("checkpoints", "save_name")
            with open(os.path.join("predictions", f"output_{save_name}.pkl"), "wb") as outfile:
                pickle.dump(self.kpts_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done in {}s".format(time.time() - start))

    def update_metrics(self, kpts, kpts_gt, visible=None,
                       ids=None, oks: OKS = None, accuracy: Metric_ITOP = None):
        """Funtion to update metrics

            Args:
                kpts (np.ndarray): Current data to update
                kpts_gt (np.ndarray): Ground truth relative to current update
                visible (np.ndarray, optional): Visible binary mask
                ids (list of str, optional): ids of the current kpts to update
                oks (OKS, optional): oks metric variable
                accuracy (Metric_ITOP, optional): accuracy metric variable

        """
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
                name = [self.data_loader.dataset.ids[el] for el in ids]
                ids_label = self.ids_test
                imgs = h5py.File(f"{self.data_path}ITOP/ITOP_{self.side}_test_depth_map.h5", 'r')['data']
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
        if oks is not None:
            oks.eval(kpts_2d, kpts_gt_2d)  # ToDo visibleeee
        if accuracy is not None:
            accuracy.eval(np.round(kpts_3d, 3), np.round(kpts_gt_3d, 3), visible)

    def save_images(self, ids, kpts):
        """Save image to the directory specified in the configer

            Args:
                kpts (np.ndarray): Current data to update
                ids (list of str, optional): ids of the current kpts to update

        """
        if self.dataset == "itop":
            name = [self.data_loader.dataset.ids[el] for el in ids]
            ids_label = self.ids_test
            imgs_data = h5py.File(f"{self.data_path}ITOP/ITOP_{self.side}_test_depth_map.h5", 'r')['data']
            imgs = list()
            for i, n in enumerate(name):
                index = int(np.where(np.array(ids_label) == n)[0])
                imgs.append(imgs_data[index])
            # Convert 3D annotations to 2D
            if self.configer.get("metrics", "kpts_type").lower() == "3d":
                kpts_3d = kpts.copy()
                kpts = world_to_depth(kpts)
                zaxis = np.zeros((kpts[0].shape[0], 3))
                zaxis[:, :-1] = kpts[0]
                zaxis[:, -1] = kpts_3d[0][:, -1] / 1000.0
                if not os.path.exists(f"{self.configer.get('data', 'result_dir')}/images/vitruvian"):
                    os.makedirs(f"{self.configer.get('data', 'result_dir')}/images/vitruvian")
                plot_2D_3D(zaxis, imgs[0], f"{self.configer.get('data', 'result_dir')}/images/vitruvian/{ids[0]}.eps",
                           stride=1)
            elif self.configer.get("metrics", "kpts_type").lower() == "rwc":
                kpts = kpts[..., :2]
            elif self.configer.get("metrics", "kpts_type").lower() == "2d":
                kpts_3d = np.zeros((kpts.shape[0], 15, 3))
                if self.dataset == "itop":
                    for i, n in enumerate(name):
                        index = int(np.where(np.array(ids_label) == n)[0])
                        depth = imgs_data[index] * 1000
                        kpts_3d[i] = depth_to_world(kpts[i], depth)
                if not os.path.exists(f"{self.configer.get('data', 'result_dir')}/images/op_2d_3d"):
                    os.makedirs(f"{self.configer.get('data', 'result_dir')}/images/op_2d_3d")
                plot_2D_3D(kpts[0], imgs[0], f"{self.configer.get('data', 'result_dir')}/images/op_2d_3d/{ids[0]}.eps",
                           stride=1)
        else:
            raise NotImplementedError("Dataset: {} not implemented".format(self.dataset))
        return
        imgs_detection = np.array([point_on_image(k, el, v) for k, el, v in zip(kpts, imgs, visible)])

        if not os.path.exists(f"{self.configer.get('data', 'result_dir')}/images/patch2d"):
            os.makedirs(f"{self.configer.get('data', 'result_dir')}/images/patch2d")
        PATH = f"{self.configer.get('data', 'result_dir')}/images/patch2d"
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        for name, el in zip(ids, imgs_detection):
            cv2.imwrite(f"{PATH}/{str(name)}.png", el)
