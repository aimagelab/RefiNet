# Import torch Dataset
from torch.utils.data import Dataset

# Import utilities
import h5py
import pickle
import numpy as np
from src.utils.utils_3d import pointcloud_normalization, world_to_depth, zaxis_to_world_np, depth_to_world
from src.utils.normalization import MEAN_itop as mean_base, STD_itop as std_base
from src.utils.normalization import MEAN_patch_depth as mean_depth_patch, STD_patch_depth as std_depth_patch

class ITOP(Dataset):
    """ITOP Dataset class

    Attributes:
        configer (Configer): Configer object, contains procedure configuration.
        mu (int): Mean value for the Gaussian Noise data augmentation
        sigma (int): Sigma value for the Gaussian Noise data augmentation
        ids (list of: str): List of ids value for every frame in the ITOP dataset
        kpts (dict): Input keypoints dictionary
        gt (dict): Ground truth keypoints dictionary
        visible (dict): Visible description for every keypoints
        size (int): Effective size of Dataset for the current procedure phase (train, test, val)

    """
    def __init__(self, configer, base_transform=None,
                 input_transform=None, label_transform=None,
                 split="train"):
        """Constructor method for ITOP Dataset class

        Args:
            configer (Configer): Configer object for current procedure phase (train, test, val)
            base_transform (Object, optional): Data augmentation transformation for every data
            input_transform (Object, optional): Data augmentation transformation only for input
            label_transform (Object, optional): Data augmentation transformation only for labels
            split (str, optional): Current procedure phase (train, test, val)

        """

        print("Loader started.")

        # Setting up useful public variables
        self.configer = configer
        self.base_transform = base_transform            #: Basic data augmentation transform for every data.
        self.input_transform = input_transform          #: Data augmentation transform for input data.
        self.label_transform = label_transform          #: Data augmentation transformation only for labels
        self.split = split                              #: Split indicate Test/Train/Val procedure
        self.side = self.configer["side"]               #: Side for top or side view in ITOP
        self.data_path = self.configer["train_dir"]     #: Path to data directory

        # Setting random state seed for validation and test only
        self.mu = self.configer.get("data_aug", "mu")
        self.sigma = self.configer.get("data_aug", "sigma")
        if self.split == "test":
            self.rand_generator = np.random.RandomState(seed=37)        #: Random Generator for adding Noise during test
        if self.split == "val":
            self.rand_generator = np.random.RandomState(seed=1295185)   #: Random Generator for adding Noise during val

        self.type = self.configer.get("data", "type")   #: str: Data description, can be base, depth, voxel or pcloud.

        # Loading input pcloud/depth ids
        if self.split == 'train' or self.split == 'val':
            with open(self.configer.get("data", "kpts_path"), 'rb') as infile:
                # ITOP data are saved as Meter, we use everything in mm
                data = pickle.load(infile)

            if self.type.lower() == 'voxel' or self.type.lower() == 'pcloud':
                ids = h5py.File(f"{self.data_path}ITOP/ITOP_{self.side}_train_point_cloud.h5", 'r')['id']
            elif self.type.lower() == 'depth' or self.type.lower() == 'base':
                ids = h5py.File(f"{self.data_path}ITOP/ITOP_{self.side}_train_depth_map.h5", 'r')['id']
            else:
                raise NotImplementedError('Data type not supported: {}'.format(self.type))

            if self.split == "train":
                self.ids = [str(el) for el in ids if not (str(el).startswith("b\'04_") or str(el).startswith("b\'05_"))]
            else:
                self.ids = [str(el) for el in ids if (str(el).startswith("b\'04_") or str(el).startswith("b\'05_"))]
            labels = h5py.File(f"{self.data_path}ITOP/ITOP_{self.side}_train_labels.h5", 'r')

        elif self.split == 'test':
            with open(self.configer.get("data", "kpts_path_test"), 'rb') as infile:
                data = pickle.load(infile)

            if self.type.lower() == 'voxel' or self.type.lower() == 'pcloud':
                ids = h5py.File(f"{self.data_path}ITOP/ITOP_{self.side}_test_point_cloud.h5", 'r')['id']
            elif self.type.lower() == 'depth' or self.type.lower() == 'base':
                ids = h5py.File(f"{self.data_path}ITOP/ITOP_{self.side}_test_depth_map.h5", 'r')['id']
            else:
                raise NotImplementedError('Data type not supported: {}'.format(self.type))

            self.ids = [str(el) for el in ids]
            labels = h5py.File(f"{self.data_path}ITOP/ITOP_{self.side}_test_labels.h5", 'r')

        else:
            raise NotImplementedError('Split error: {}'.format(self.split))

        # Loading full annotations for voxel/pcloud
        self.kpts = dict()
        self.gt = dict()
        self.visible = dict()
        tmp = self.ids.copy()
        self.ids_str = list(map(str, labels['id']))     #: list of str:  Ids list for ITOP on test split

        if self.type == "base" or self.type == "pcloud":
            if self.split in ("train", "val"):
                images = h5py.File(f"{self.data_path}ITOP/ITOP_{self.side}_train_depth_map.h5", 'r')['data']
            else:
                images = h5py.File(f"{self.data_path}ITOP/ITOP_{self.side}_test_depth_map.h5", 'r')['data']

        for name in tmp:
            index = self.ids_str.index(name)

            # If is_valid flag is set to 0,
            # you should not use any of the provided human joint locations for the particular frame.
            if labels['is_valid'][index] == 0:
                # Removing the ids from the correct variable
                self.ids.remove(name)
                continue
            else:
                if self.type.lower() in ('voxel', 'pcloud'):
                    # We process data to work with mm value, normally they are expressed in meters
                    self.gt[name] = np.array(labels['real_world_coordinates'][index]) * 1000
                elif self.type.lower() == 'depth' or self.type.lower() == 'base':
                    if self.configer.get('metrics', 'kpts_type').lower() == '2d':
                        self.gt[name] = np.array(labels['image_coordinates'][index])
                    elif self.configer.get('metrics', 'kpts_type').lower() == 'rwc':
                        gt = np.array(labels['real_world_coordinates'][index])
                        gt[:, :2] = world_to_depth(gt)
                        gt[:, 2] = gt[:, 2] * 1000
                        self.gt[name] = gt
                    else:
                        # We process data to work with mm value, normally they are expressed in meters
                        self.gt[name] = np.array(labels['real_world_coordinates'][index]) * 1000
                else:
                    raise NotImplementedError('Type error: {}'.format(self.type))

                if self.configer.get("data", "from_gt") and self.split.lower() in ("val", "test"):
                    # retrieve kpts in mm, input unprocessed is expressed in m
                    if self.type == "base":
                        in_kpt = np.array(labels['image_coordinates'][index])

                        self.kpts[name] = depth_to_world(in_kpt, images[index]) * 1000
                        # self.kpts[name] = np.array(self.gt[name].copy())
                    else:
                        self.kpts[name] = self.gt[name].copy()

                    if self.configer.get('metrics', 'kpts_type').lower() in ('2d', 'rwc'):
                        noise = self.rand_generator.normal(self.mu, self.sigma, (self.kpts[name].shape[0], 2))
                        self.kpts[name][:, :2] = self.kpts[name][:, :2] + noise
                    else:
                        noise = self.rand_generator.normal(self.mu, self.sigma, self.kpts[name].shape)
                        if self.type == "base":
                            if self.sigma < 10:
                                noise = noise[:, :-1]
                                tmp = world_to_depth(self.kpts[name])
                                tmp = tmp + noise
                                tmp = depth_to_world(tmp, images[index]) * 1000
                                self.kpts[name] = tmp
                            else:
                                self.kpts[name][self.kpts[name][:, 2] != 0] = self.kpts[name][self.kpts[name][:, 2] != 0] \
                                                                          + noise[self.kpts[name][:, 2] != 0]
                        else:
                            self.kpts[name] = self.kpts[name] + noise
                elif self.configer.get("data", "from_gt") and self.split.lower() == "train" and self.type == "base":
                    # Adding here gaussian noise on 2D then calculating RWC on the noise input
                    in_kpt = np.array(labels['image_coordinates'][index])
                    noise = np.random.normal(self.mu, self.sigma, (in_kpt.shape[0], 2))
                    in_kpt = in_kpt + noise
                    self.kpts[name] = depth_to_world(in_kpt, images[index]) * 1000
                elif self.configer.get("data", "from_gt") and self.split.lower() == "train" and self.type == "pcloud":
                    self.kpts[name] = self.gt[name]
                else:
                    # retrieve kpts in mm
                    if self.configer["data", "kpts_type"].lower() == "3d" and data[index].shape[-1] != 3 \
                            and self.configer.get("data", "from_gt") is False:
                        self.kpts[name] = depth_to_world(data[index], images[index] * 1000)
                    else:
                        self.kpts[name] = data[index]

            self.visible[name] = labels['visible_joints'][index]
            if self.split in ("test", "val"):
                self.visible[name][self.visible[name] == 0] = 1
            if self.type == 'base':
                for i, el in enumerate(self.visible[name]):
                    if el == 0:
                        self.kpts[name][i] = np.zeros((self.kpts[name][i].shape))
                        self.gt[name][i] = np.zeros((self.gt[name][i].shape))

        # Setting dataset size.
        self.size = len(self.ids)
        print(f"Loaded {self.size} values. Done.")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """getitem method to iterate over ITOP dataset

        Args:
            idx (int): Iteration index

        Returns:
            patches (np.ndarray): Processed patches for patches2D, vitruvian and pcloud of the idx selected value
            gt (np.ndarray): Ground truth value for the idx selected value
            visible (np.ndarray): Visible mask for the idx selected value
            kpts (np.ndarray): Input kpts before extracting the patch for idx selected value
            idx (int): Iteration index for current value

        """
        name = self.ids[idx]
        if self.configer.get("data", "from_gt") and self.split == 'train':
            kpts = self.kpts[name].copy()
            if self.configer.get('metrics', 'kpts_type').lower() in ('2d', 'rwc'):
                noise = np.random.normal(self.mu, self.sigma, (kpts.shape[0], 2))
                kpts[:, :2] = kpts[:, :2] + noise
            else:
                # If base for vitruvian, noise has already been added at 2D level over kpts public variable
                if self.type == 'base':
                    if self.configer["visible_include"] is None:
                        for i, el in enumerate(self.visible[name]):
                            if el == 0:
                                kpts[i] = np.zeros(self.kpts[name][i].shape)
                else:
                    noise = np.random.normal(self.mu, self.sigma, kpts.shape)
                    kpts = kpts + noise
        else:
            kpts = np.array(self.kpts[name])
            if 'zaxis' in self.configer['data','kpts_path_test']:
                kpts = zaxis_to_world_np(kpts)
            if self.configer['data', 'kpts_type'] == "2D" and \
                    ('zaxis' in self.configer['data','kpts_path_test'] or '3d' in self.configer['data','kpts_path_test'].lower()):
                kpts = world_to_depth(kpts)
        gt = self.gt[name].copy()

        split = self.split if self.split != "val" else "train"
        if self.type.lower() == 'voxel' or self.type.lower() == 'pcloud':
            f = h5py.File(f"{self.data_path}ITOP/ITOP_{self.side}_" + split + "_point_cloud.h5", 'r')
        elif self.type.lower() == 'depth' or self.type.lower() == 'base':
            f = h5py.File(f"{self.data_path}ITOP/ITOP_{self.side}_" + split + "_depth_map.h5", 'r')
        else:
            raise NotImplementedError('Data type not supported: {}'.format(self.type))

        index = self.ids_str.index(name)
        data = f['data'][index].copy()

        # Voxel type
        if self.type.lower() == 'voxel':
            """
                Todo:
                    Implement Voxel Extracting procedure
            """

        # Pcloud type
        elif self.type.lower() == 'pcloud':
            data = data * 1000
            # Retrieving pcloud dimension
            pcloud_dim = self.configer.get("data", "pcloud_dim")
            if self.configer["side"] == "top":
                N = 5000
            else:
                N = 2000
            patches = np.zeros((15, N+1, 3))
            for i, el in enumerate(kpts):
                if self.visible[name][i] == 0 or el[2] == 0:
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
                patches[i][:tmp.shape[0], :] = tmp
                patches[i][-1] = len(tmp)
            gt = kpts - gt

        # 2D depth maps type
        elif self.type.lower() == 'depth':
            kpts = kpts[:, :2]
            mean = mean_depth_patch
            std = std_depth_patch
            # Retrieving depth patch dimesnion
            patch_dim = self.configer.get("data", "patch_dim") // 2

            patches = np.zeros((15, patch_dim * 2, patch_dim * 2))
            for i, el in enumerate(kpts):
                # Slicing for retrieving correct patch
                patch = data[int(el[1]) - patch_dim: int(el[1]) + patch_dim,
                             int(el[0]) - patch_dim: int(el[0]) + patch_dim]
                patches[i][:patch.shape[0], :patch.shape[1]] = patch
                patches[i][patches[i] == 0] = np.mean(data)
                if self.configer.get("data", "from_gt") and self.configer.get('metrics', 'kpts_type').lower() == 'rwc':
                    kpts[i, 2] = data[int(el[1]) if el[1] < 240 else 239, int(el[0]) if el[0] < 320 else 319] * 1000
            patches -= mean
            patches /= std
            gt = kpts - gt

        # Option without real patches, only kpts
        # Patches parameters is still used for simplicity
        elif self.type.lower() == 'base':
            patches = kpts.copy()
            if self.configer.get("offset") is True:
                gt = kpts - gt
            patches[patches[:, 2] != 0] -= mean_base
            patches[patches[:, 2] != 0] /= std_base
        else:
            raise NotImplementedError('Data type not supported: {}'.format(self.type))

        # Data augmentation
        if self.base_transform is not None:
            patches = self.base_transform(patches)
        if self.input_transform is not None:
            patches = self.input_transform(patches)
        if self.label_transform is not None:
            gt = self.label_transform(gt)

        return patches.astype(np.float32), gt.astype(np.float32), self.visible[name].astype(np.float32), kpts.astype(
            np.float32), idx
