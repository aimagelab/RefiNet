import os
import scipy.io
import json
import numpy as np
import pickle
import cv2
import torch
import time

from torch.utils.data import DataLoader

# Import Datasets
from src.Datasets.Baracca import Baracca

# Import Model
from src.models.refinement import LinearModel
from src.models.refine_patch_2d import Patch_2D_Model
from src.models.pointnet import PointPatch
from src.models.module_utilizer import ModuleUtilizer

# Import Utils
from tqdm import tqdm


# Setting seeds
def worker_init_fn(worker_id):
    # seed = torch.initial_seed() + worker_id
    np.random.seed(torch.initial_seed() % 2 ** 32)
    # random.seed(seed)
    # torch.manual_seed(seed)


def world_to_depth(kpt_3d, angle):
    Cx = 336.14
    Cy = 231.349
    Fx = 461.605
    Fy = 461.226
    width = 480
    height = 640
    if angle == 0 or angle == 360:
        pass
    elif angle == 90:
        tmp = Cx
        Cx = Cy
        Cy = height - tmp
        tmp = Fx
        Fx = Fy
        Fy = tmp
    elif angle == 180:
        Cx = width - Cx
        Cy = height - Cy
    elif angle == 270:
        tmp = Cy
        Cy = Cx
        Cx = width - tmp
        tmp = Fx
        Fx = Fy
        Fy = tmp

    tmp = np.zeros((15, 2), dtype=np.float32)
    # tmp[..., 0] = kpt_3d[..., 0] * Fx / 2200 + Cx
    tmp[..., 0] = kpt_3d[..., 0] * Fx / kpt_3d[..., 2] + Cx
    # tmp[..., 1] = -kpt_3d[..., 1] * Fy / 2200 + Cy
    tmp[..., 1] = -kpt_3d[..., 1] * Fy / kpt_3d[..., 2] + Cy
    return np.nan_to_num(tmp)


class PoseTest(object):
    """
      DepthPose class for test only.
    """

    def __init__(self, configer):
        self.configer = configer

        # DataLoader
        self.data_loader = None

        # Input and output size
        self.input_size = configer.get('data', 'input_size')
        self.output_size = configer.get('data', 'output_size')

        self.data_path = configer["train_dir"]

        # Module load and save utility
        self.model_utility = ModuleUtilizer(self.configer)
        self.net = None

        # Other useful data
        self.side = self.configer["side"]
        self.result_dir = self.configer.get("data", "result_dir")
        self.type = self.configer.get("data", "type")
        self.dataset = self.configer.get("dataset").lower()
        self.mean = 0.0
        self.std = 0.0
        self.ids_train = None
        self.ids_test = None
        self.img_saved = 0

    def init_model(self):
        """
            Load model function.
        """
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

        # Selecting correct model and normalization variable based on type variable
        # ToDO add new models for pcloud and voxel
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
            elif self.configer.get("network", "model_name").lower() == "v2":
                self.net = Patch_2D_Model_V2(self.configer.get("data", "output_size"), activation)
            elif self.configer.get("network", "model_name").lower() == "v3":
                self.net = Patch_2D_Model_V3(self.configer.get("data", "output_size"), activation)
            elif self.configer.get("network", "model_name").lower() == "v4":
                self.net = Patch_2D_Model_V4(self.configer.get("data", "output_size"), activation)
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
        if self.dataset == "baracca":
            Dataset = Baracca
        else:
            raise NotImplementedError(f"Dataset not supported: {self.configer.get('dataset')}")

        # Setting Dataloaders
        self.data_loader = DataLoader(
            Dataset(self.configer, split="test"),
            1, shuffle=False,
            num_workers=self.configer.get('solver', 'workers'), pin_memory=True, worker_init_fn=worker_init_fn)

    def __test(self):
        """
            Test function on multiple images.
        """
        self.net.eval()

        # Save inference time
        tot_time = 0.0

        with torch.no_grad():
            for i, data_tuple in enumerate(tqdm(self.data_loader)):
                """
                    input, gt, visible, img_index
                """
                inputs = data_tuple[0].cuda()
                kpts_in = data_tuple[1].cpu().numpy()
                visible = data_tuple[2].cpu().numpy()
                ids = data_tuple[-1].cpu().numpy()
                start = time.time()

                # Inference
                output = self.net(inputs)

                # Saving inference time values
                end = time.time()
                tot_time += end - start
                # loss = mse_masked(output, gt, data_tuple[2].cuda())

                if self.type.lower() in ("depth", "pcloud") \
                        or (self.type.lower() == "base" and self.configer.get("offset") is True):
                    kpts_off = output.cpu().detach().numpy().astype(np.float32)
                    kpts_pred = kpts_in + kpts_off
                else:
                    kpts_pred = output.cpu().detach().numpy().astype(np.float32)

                self.kpts_dict[ids.item()] = kpts_pred[0] * np.expand_dims(visible[0], axis=1)

                # Saving images
                if self.configer.get("save_img") is True:
                    name = self.configer.get("data", "type")
                    self.save_images(ids, kpts_pred, kpts_in, visible, f"/homes/adeusanio/imgs/{name}")
                    self.img_saved += ids.shape[0]

    def test(self):
        print("Starting test procedure.")
        start = time.time()
        self.kpts_dict = np.zeros((2400, 15, self.configer["data", "input_size"][-1]))
        self.__test()
        if self.configer.get("save_pkl") is True and self.configer.get("from_gt") is False:
            save_name = self.configer.get("checkpoints", "save_name")
            with open(os.path.join("predictions", f"{save_name}_new.pkl"), "wb") as outfile:
                pickle.dump(self.kpts_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done in {}s".format(time.time() - start))


    def save_images(self, ids, kpts, kpts_in, visible, out_path):
        visible = visible[0]
        kpts = kpts[0]
        if self.configer["data", "kpts_type"].lower() == "3d":
            seq = int(self.data_loader.dataset.imgs_path[ids.item()].split("/")[-2])
            kpts2d = world_to_depth(kpts, int(self.data_loader.dataset.pos[str(seq)]['orientation'][0]))
            kpts_in2d = world_to_depth(kpts_in, int(self.data_loader.dataset.pos[str(seq)]['orientation'][0]))
        path = self.data_loader.dataset.imgs_path[ids.item()]
        name = f"{ids.item()}.png"
        img = cv2.imread(path, 3)
        img = (img / img.max() * 255).astype(np.uint8)
        for i, el in enumerate(kpts2d):
            if visible[i] == 0:
                continue
            if i == 0:
                cv2.circle(img, (int(el[0]), int(el[1])), 5, (0, 0, 255), -1)
            else:
                cv2.circle(img, (int(el[0]), int(el[1])), 5, (255, 0, 0), -1)
            cv2.putText(img, f"{kpts[i][2]:.1f}", (int(el[0] - 30), int(el[1]) - 20), 1, 1, (255, 255, 255), 2)
        cv2.imwrite(f"{out_path}/{name}", img)

