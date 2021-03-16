import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

JOINTS_COLOR = [(255, 0, 0), (244, 41, 0), (234, 78, 0), (223, 112, 0),
                (213, 142, 0), (202, 168, 0), (192, 191, 0), (151, 181, 0),
                (213, 142, 127), (202, 168, 127), (192, 191, 127), (151, 181, 127),
                (114, 170, 0), (80, 160, 0), (50, 149, 0), (23, 139, 0),
                (114, 170, 127), (80, 160, 127), (50, 149, 127), (23, 139, 127),
                (244, 41, 0)]

LIMBS = [(0, 1), (1, 2), (1, 3), (2, 4), (4, 6), (3, 5), (5, 7),
         (1, 8), (8, 9), (8, 10), (10, 12), (12, 14), (9, 11), (11, 13)]

def point_on_image(kpts, img, visible = None):
    """Plot points on the provided img

    Args:
        kpts (np.ndarray): Array containing keypoints to plot
        img (np.ndarray): img to plot with keypoints
        visible (np.ndarray, optional): Binary visibility mask to plot different color on occluded joints

    Returns:
        np.ndarray: Processed image

    """
    if len(img.shape) == 2:
        tmp = np.zeros((img.shape[0], img.shape[1], 3))
        tmp[:, :, 0] = img.copy()
        tmp[:, :, 1] = img.copy()
        tmp[:, :, 2] = img.copy()
        img_new = tmp
    elif len(img.shape) != 3:
        raise ValueError("Image shape: {} is not correct".format(img.shape[2]))
    else:
        raise ValueError
    img_new = (img_new * 255 / np.amax(img_new)).astype(np.uint8)
    for i, el in enumerate(kpts):
        if visible is not None:
            if visible[i] == 0:
                cv2.circle(img_new, (int(el[0]), int(el[1])), 5, (0, 0, 255), -1)
            else:
                cv2.circle(img_new, (int(el[0]), int(el[1])), 5, JOINTS_COLOR[i], -1)
        else:
            cv2.circle(img_new, (int(el[0]), int(el[1])), 5, JOINTS_COLOR[i], -1)
    return img_new


def plot_2D_3D(coord, image, output_path=None, max_z=4.0, stride=1, limbs=LIMBS):
    """3D plot with image positioned on an axis

    Args:
        coord (np.ndarray): Array containing keypoints to transform
        img (np.ndarray): img to plot on surface with keypoints
        output_path (str, optional): Output path where save the image, if not provided the plot will be showed instead
        max_z (float): Max value to set the range of the z-axis
        stride (int): Stride improve or decrease the surface image quality. 1 equals to better quality.
        limbs (list of tuple of int):  list of joints couples (limbs), used to plot connected line

    """
    assert coord.shape[-1] in (2, 3)
    assert len(image.shape) == 2
    joints = np.zeros((coord.shape[0], 3))
    if coord.shape[-1] == 2:
        # Retrieving z value from the given depth-map
        z = np.array([image[int(el[1]), int(el[0])] for el in coord])
        joints[:, -1] = z
    else:
        z = joints[:, -1]
    joints[:, :coord.shape[-1]] = coord.copy()

    img = np.zeros((image.shape[0], image.shape[1], 3))
    img[..., 0] = image.copy()
    img[..., 1] = image.copy()
    img[..., 2] = image.copy()
    img = (img * 255 / img.max()).astype(np.uint8)
    # Plotting joints over 2D image
    for el in joints:
        cv2.circle(img, (int(el[0]), int(el[1])), 5, (51, 104, 255), -1, lineType=cv2.LINE_AA)

    _ = Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim(max_z, 2.5)
    ax.set_ylim(0, img.shape[1])
    ax.set_zlim(0, img.shape[0])

    X1 = np.arange(0, 320, 1)
    Y1 = np.arange(0, 240, 1)
    X1, Y1 = np.meshgrid(X1, Y1)

    ax.plot_surface(np.atleast_2d(max_z), X1, Y1, rstride=stride, cstride=stride, facecolors=img / 255.)
    c_line = "#9CB7FF"
    for el in joints:
        ax.scatter(el[2], el[0], el[1], c="#3368FF")
    for el in limbs:
        x = [joints[el[j]][0] for j in range(2)]
        y = [joints[el[j]][1] for j in range(2)]
        curr_z = [z[el[j]] for j in range(2)]
        ax.plot(curr_z, x, y, c=c_line)
    ax.view_init(elev=-160, azim=-126)
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.set_zticklabels([])
    if output_path is None:
        plt.show()
    else:
        if ".eps" in output_path:
            plt.savefig(output_path, format="eps")
        else:
            plt.savefig(output_path)
        plt.close()
