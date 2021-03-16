import numpy as np
from math import exp
import torch

itop_sigmas = [0.107, 0.107,
               0.079, 0.079, 0.072, 0.072, 0.062, 0.062,
               0.107, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]


class OKS:
    """Object Keypoints Similarity calculator class

    This type of metrics is described in COCO, http://cocodataset.org/#keypoints-eval
    It's used to evaluate 2D precision and accuracy, as mean of AveragePrecision (AP)
    with threshold=np.range(0.5, 0.95, 0.05)
    """
    def __init__(self, n_joints: int = 25, sigmas: float = None):
        """Constructor function for OKS class"""
        # AP, TP and FP from 0.5 to 0.95, 10 thresh
        self.AP = [0] * 10
        self.TP = [0] * 10
        self.FP = [0] * 10

        # Same thing as above but for every single Joint
        self.joints_TP = [[0 for i in range(n_joints)] for j in range(10)]
        self.joints_FP = [[0 for i in range(n_joints)] for j in range(10)]

        # Useful for results table
        self.joints_AP50 = []
        self.joints_AP75 = []

        # Flag to avoid to re-compute Precision
        self._compute = True

        # Name list
        self._names = ['AP_50', 'AP_55', 'AP_60', 'AP_65', 'AP_70', 'AP_75', 'AP_80', 'AP_85', 'AP_90', 'AP_95']

        # Used to compute average
        self.TOT = 0
        self.n_joints = n_joints

        # Setting sigmas for computation
        # Sigmas, different for every joint, if not defined
        if sigmas is not None:
            if isinstance(sigmas, float):
                self.sigmas = [sigmas] * n_joints
            else:
                if len(sigmas) != n_joints:
                    raise ValueError("Sigmas number is not equal to number of joints: {}".format(len(sigmas)))
                else:
                    self.sigmas = sigmas
        else:
            self.sigmas = itop_sigmas

    def __repr__(self):
        """Represent function for OKS"""
        return 'Metrics()'

    def __str__(self):
        """To string function for OKS"""
        if self._compute:       # Compute only if necessary
            self.get_metrics()
        val = "--- OKS COCO METRIC ---\n"
        for i in range(len(self.AP)):
            val += "{} = {:.3f}, ".format(self._names[i], self.AP[i])
        val += "mAP = {:.3f}".format(np.mean(self.AP))
        return val

    # Evaluate one oks and update TP and FP for average metric
    def update(self, oks):
        """Update metric with 1 new sample
        Args:
            oks (float): New sample similarity to update

        """
        self._compute = True
        for i, val in enumerate(np.arange(0.5, 1, 0.05)):
            if oks > val:
                self.TP[i] += 1
            else:
                self.FP[i] += 1

    def update_joints(self, similarity, index):
        """Update metric with 1 new sample for a particular joint
        Args:
            similarity (float): Single Joint similarity to update
            index (int): Joint index to update

        """
        for i, val in enumerate(np.arange(0.5, 1, 0.05)):
            if similarity > val:
                self.joints_TP[i][index] += 1
            else:
                self.joints_FP[i][index] += 1

    def get_metrics(self):
        """Getter for output metrics value
        Note:
            Metrics are calculated only the first time this function is called

        """
        if self._compute:
            for i in range(len(self.TP)):
                self.AP[i] = self.TP[i] / (self.TP[i] + self.FP[i]) if self.TP[i] > 0 else 0
            self._compute = False
        return self.AP, np.mean(self.AP)

    def get_joints_metrics(self):
        """Getter for output metrics value for every single joint
        Note:
            Metrics are calculated only the first time this function is called

        """
        joints_AP = list()
        for i in range(self.n_joints):
            tot = 0
            for j in range(10):
                if self.joints_TP[j][i] == 0:
                    tot += 0
                else:
                    tot += self.joints_TP[j][i] / (self.joints_TP[j][i] + self.joints_FP[j][i])
            joints_AP.append(tot / 10)

            self.joints_AP50.append(
                self.joints_TP[0][i] / (self.joints_TP[0][i] + self.joints_FP[0][i])
                if self.joints_TP[0][i] != 0 else 0)
            self.joints_AP75.append(
                self.joints_TP[5][i] / (self.joints_TP[5][i] + self.joints_FP[5][i])
                if self.joints_TP[5][i] != 0 else 0)
        return joints_AP

    def eval(self, detection, gt):
        """Evaluation function for every new sample
        Args:
            detection (Union(torch.Tensor, np.ndarray)): New Input sample
            gt (Union(torch.Tensor, np.ndarray)): Ground truth value

        """
        # Checking input and gt type
        if isinstance(detection, torch.Tensor):
            detection = detection.data.cpu().squeeze().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.data.cpu().squeeze().numpy()

        # Usually we iterate over full batch
        for x in range(gt.shape[0]):
            num = 0.0
            tot = 0
            # Computing area of joints
            kpts = detection[x]
            area = (np.max(kpts[:, 0]) - np.min(kpts[:, 0])) * (np.max(kpts[:, 1]) - np.min(kpts[:, 1]))
            # Iterate over joints
            for k in range(gt.shape[1]):
                # Working only with n_joints needed
                if k > self.n_joints:
                    continue
                # Only when GT is existent
                elif np.count_nonzero(gt[x, k]):
                    a, b = gt[x, k][0], gt[x, k][1]
                    a2, b2 = detection[x, k][0], detection[x, k][1]
                    if detection[x, k].shape[0] == 2:
                        tmp = ((a - a2) ** 2 + (b - b2) ** 2) / ((2 * self.sigmas[k]) ** 2 * (area + 1e-16) * 2)
                    else:
                        c = gt[x, k][2]
                        c2 = detection[x, k][2]
                        tmp = ((a - a2) ** 2 + (b - b2) ** 2 + (c - c2) ** 2) / (
                                (2 * self.sigmas[k]) ** 2 * (area + 1e-16) * 2)
                    # Update joint after computing distance
                    self.update_joints(exp(-tmp), k)
                    num += exp(-tmp)
                    tot += 1
            # Compute average
            if tot > 0:
                oks = num / tot
                self.update(oks)


class Metric_ITOP:
    """mAP calculator class

    This type of metrics is described in ITOP, arXiv:1603.07076
    It's used to evaluate 3D accuracy with a threshold fixed at 10cm
    """
    def __init__(self, n_joints: int = 15, thresh: int = 10):
        """Constructor function for mAP ITOP class"""
        # Accumulate value over threshold in tot variable
        self.tot = 0.0
        self.tot_up = 0.0
        self.tot_down = 0.0
        # Counting samples
        self.counter = 0
        self.counter_up = 0
        self.counter_down = 0
        # Compute average cm_distance error
        self.dist_cm = 0.0
        self.thresh = thresh
        # Joints you want to consider for the metric
        self.n_joints = n_joints

    def eval(self, kpts, gt, visible=None):
        """Evaluation function for every new sample
        Args:
            kpts (np.ndarray): New Input sample
            gt (np.ndarray): Ground truth value
            visible (np.ndarray, optional): Visible mask for new sample

        """
        kpts = kpts[:, :self.n_joints, :]
        gt = gt[:, :self.n_joints, :]
        # Visible variable could be None, just creating it right processing gt
        if visible is None:
            visible = (gt != [0, 0, 0])[:, 0].astype(np.float32)
        # If taking single input, we un-squeeze to emulate full batch
        if len(kpts.shape) == 2:
            kpts = np.expand_dims(kpts, axis=0)
            gt = np.expand_dims(gt, axis=0)
            visible = np.expand_dims(visible, axis=0)

        for i, (kpt, kpt_gt, v) in enumerate(zip(kpts, gt, visible)):
            if kpt.shape[1] != 3:
                raise ValueError("Error in kpts dimension: {}".format(len(kpt.shape)))

            err_dist = np.sqrt(np.sum((kpt[v == 1] - kpt_gt[v == 1]) ** 2, axis=1))
            self.dist_cm += np.sum(err_dist)
            self.tot += (err_dist < self.thresh).sum()
            err_dist = np.sqrt(np.sum((kpt[:8][v[:8] == 1] - kpt_gt[:8][v[:8] == 1]) ** 2, axis=1))
            self.tot_up += (err_dist < self.thresh).sum()
            err_dist = np.sqrt(np.sum((kpt[8:][v[8:] == 1] - kpt_gt[8:][v[8:] == 1]) ** 2, axis=1))
            self.tot_down += (err_dist < self.thresh).sum()
            self.counter += (v == 1).sum()
            self.counter_up += (v[:8] == 1).sum()
            self.counter_down += (v[8:] == 1).sum()

    def get_values(self):
        """Getter for output metrics value"""
        return self.tot / self.counter if self.counter > 0 else 0, self.dist_cm / self.counter / 10 if self.counter > 0 else 0

    def __str__(self):
        """To str function for output metrics value"""
        return f"--- ITOP METRICS ---\nmAP ITOP -> [{(self.tot / self.counter) if self.counter > 0 else 0}], cm_dist -> " \
            f"[{self.dist_cm / self.counter / 10 if self.counter > 0 else 0}]"
