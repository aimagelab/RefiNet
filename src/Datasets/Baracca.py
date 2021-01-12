import json
import numpy as np
import os
import cv2

from src.utils.normalization import MEAN_itop as mean_base, STD_itop as std_base
from src.utils import utils_v2v, utils_3d
from src.utils.utils_3d import pointcloud_normalization

POSE_TO_ITOP = [0, 1, 2, 5, 3, 6, 4, 7, 8, 9, 12, 10, 13, 11, 14]
MEAN = [
    [15.60469934, -30.83205137, 500.54442057],
    [-145.6523327, -242.46433908, 637.41408031],
    [48.08589011, 18.29300275, 963.33092631],
    [62.26005679, 113.6160867, 1463.78708049],
    [59.83684574, 112.16152143, 1947.22198239],
    [-74.58052783, 30.36067747, 552.40623454],
    [64.75511807, -30.51203683, 433.28538503],
    [6.20511586, 70.29800256, 564.33671733]]
STD = [
 [137.56794339, 154.15958852, 131.85178717],
 [121.9405036, 194.34061565, 101.25147635],
 [153.01860708, 315.90534256, 58.66701796],
 [144.45149066, 413.12329783, 70.69498393],
 [146.63532662, 419.06485418, 70.47524126],
 [130.39735316, 193.21842537, 153.30596828],
 [142.26643692, 102.91073407, 68.36512043],
 [148.650386, 170.6648991, 112.70416918]]

Cx = 336.14
Cy = 231.349
Fx = 461.605
Fy = 461.226

def zaxis_to_world(x, y, z, angle):
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

    x = (x - Cx) * z / Fx
    y = (Cy - y) * z / Fy
    return [x, y, z]

class Baracca:
    def __init__(self, configer, base_transform=None,
                 input_transform=None, label_transform=None,
                 split="train"):
        """
        :param configer: Configer variable
        :param split: split value: train, val, test
        """

        print("Loader started.")

        # Setting up useful variables
        self.configer = configer
        self.split = split
        self.side = self.configer["side"]
        self.data_path = self.configer["train_dir"]

        #ToDO setup training procedure on this dataset
        self.type = self.configer.get("data", "type")

        if self.configer["data", "kpts_path_test"].find(".npz") >= 0:
            skeleton = np.load(self.configer["data", "kpts_path_test"])["arr_0"]
            self.kpts = np.reshape(skeleton, (np.prod(skeleton.shape[:-2]), skeleton.shape[-2], skeleton.shape[-1]))
        else:
            skeleton = np.load(self.configer["data", "kpts_path_test"], allow_pickle=True)
            self.kpts = np.zeros((2400, 15, 3), dtype=np.float32)

        with open("{}/viewpoints.json".format(self.configer["train_dir"]), "r") as infile:
            self.pos = json.load(infile)

        self.visible = np.ones((np.prod(skeleton.shape[:-2]), skeleton.shape[-2]))
        if self.split == "test":
            self.imgs_path = list()
            for i in range(1, 31):
                for j in range(8):
                    for k in range(10):
                        subj = str(i).zfill(3)
                        seq = str(j).zfill(3)
                        n = str(k).zfill(3)
                        self.imgs_path.append(os.path.join(self.data_path, subj, seq, f"imgDepth_{subj}_{seq}_{n}.png"))
            if self.type.lower() == "depth":
                pass
            elif self.type.lower() in ("base", "pcloud"):
                for i, path in enumerate(self.imgs_path):
                    img = cv2.imread(path, 2)
                    kpt = skeleton.copy()
                    for num, el in enumerate(kpt[i]):
                        if (el[0] < 0 or el[1] < 0) or (el[0] == 0 and el[1] == 0):
                            x, y, z = 0, 0, 0
                        else:
                            x = el[0]
                            y = el[1]
                            x_l = (x - 1 if x - 1 >= 0 else 0) if x < img.shape[1] else img.shape[1] - 2
                            x_r = x + 2 if x + 2 < img.shape[1] else img.shape[1] - 1
                            y_l = (y - 1 if y -1 >= 0 else 0) if y < img.shape[0] else img.shape[0] - 2
                            y_r = y + 2 if y + 2 < img.shape[0] else img.shape[0] - 1
                            z = np.median(img[int(y_l):int(y_r), int(x_l):int(x_r)])
                        seq = int(self.imgs_path[i].split("/")[-2])
                        self.kpts[i, num] = zaxis_to_world(x, y, z, int(self.pos[str(seq)]['orientation'][0]))
                        if z == 0:
                            self.visible[i, num] = 0
            else:
                raise NotImplementedError('Data type not supported: {}'.format(self.type))
        self.size = self.kpts.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.kpts.shape[1] > 15:
            kpt_in = np.zeros((15, 3), dtype=np.float32)
            visible = np.zeros((15), dtype=np.float32)
            for i, el in enumerate(POSE_TO_ITOP):
                kpt_in[i] = self.kpts[idx][el]
                visible[i] = self.visible[idx][el]
        else:
            kpt_in = self.kpts[idx]
            visible = self.visible[idx]

        if self.type.lower() == "depth":
            patch_dim = self.configer.get("data", "patch_dim") // 2
            patches = np.zeros((15, patch_dim * 2, patch_dim * 2), np.float32)
            img = cv2.imread(self.imgs_path[idx], 2)

            for i, el in enumerate(kpt_in):
                # Slicing for retrieving correct patch
                patch = img[int(el[1]) - patch_dim: int(el[1]) + patch_dim,
                            int(el[0]) - patch_dim: int(el[0]) + patch_dim]
                patches[i][:patch.shape[0], :patch.shape[1]] = patch
                if len(patches[i][patches[i] != 0]) == 0:
                    mean = 0
                    std = 1
                else:
                    mean = np.mean(patches[i][patches[i] != 0])
                    std = np.std(patches[i][patches[i] != 0])
                    patches[i][patches[i] == 0] = mean
                if std == 0:
                    std = 1
                patches[i] -= mean
                patches[i] /= std
            kpts = patches
            kpt_in = kpt_in[:, :-1]

        elif self.type.lower() == "pcloud":
            img = cv2.imread(self.imgs_path[idx], 2)
            data = utils_3d.pointcloud(img, Fx, Fy, Cx, Cy)

            # Retrieving pcloud dimension
            pcloud_dim = self.configer.get("data", "pcloud_dim")
            if self.configer["side"] == "top":
                N = 5000
            else:
                N = 2000
            kpts = np.zeros((15, N + 1, 3), dtype=np.float32)
            for i, el in enumerate(kpt_in):
                if visible[i] == 0 or el[2] == 0:
                    continue
                tmp2 = np.zeros((16, 3))
                pcloud_dim_tmp = pcloud_dim
                while tmp2.shape[0] <= 128 and pcloud_dim_tmp <= 450:
                    tmp2 = data[
                        (data[:, 0] > el[0] - pcloud_dim_tmp / 2) & (data[:, 0] < el[0] + pcloud_dim_tmp / 2) &
                        (data[:, 1] > el[1] - pcloud_dim_tmp / 2) & (data[:, 1] < el[1] + pcloud_dim_tmp / 2) &
                        (data[:, 2] > el[2] - pcloud_dim_tmp / 2) & (data[:, 2] < el[2] + pcloud_dim_tmp / 2)
                        ].copy()
                    pcloud_dim_tmp += 50
                    tmp2 = tmp2[tmp2[:, 2] != 0]
                tmp = tmp2.copy()[:N]
                # max = 4500 points for every body parts, calculated over train + test + val split
                # Need to create cube with 4500 point, correct in the middle, 0 everywhere else
                tmp = pointcloud_normalization(tmp)
                kpts[i][:tmp.shape[0], :] = tmp
                kpts[i][-1] = len(tmp)

        elif self.type.lower() == "base":
            kpts = kpt_in.copy()
            kpts[kpts[:, 2] != 0] -= mean_base
            kpts[kpts[:, 2] != 0] /= std_base
        else:
            raise NotImplementedError('Data type not supported: {}'.format(self.type))

        return kpts, kpt_in, visible, idx