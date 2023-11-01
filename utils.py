import cv2
import os
import re
import torch
from functools import reduce
import numpy as np
from scipy import ndimage as nd


def to_torch_var(
        np_array,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        requires_grad=False,
        dtype=torch.float32
):
    """
    Function to convert a numpy array into a torch tensor for a given device
    :param np_array: Original numpy array
    :param device: Device where the tensor will be loaded
    :param requires_grad: Whether it requires autograd or not
    :param dtype: Datatype for the tensor
    :return:
    """
    var = torch.tensor(
        np_array,
        requires_grad=requires_grad,
        device=device,
        dtype=dtype
    )
    return var


def time_to_string(time_val):
    """
    Function to convert from a time number to a printable string that
     represents time in hours minutes and seconds.
    :param time_val: Time value in seconds (using functions from the time
     package)
    :return: String with a human format for time
    """

    if time_val < 60:
        time_s = '%ds' % time_val
    elif time_val < 3600:
        time_s = '%dm %ds' % (time_val // 60, time_val % 60)
    else:
        time_s = '%dh %dm %ds' % (
            time_val // 3600,
            (time_val % 3600) // 60,
            time_val % 60
        )
    return time_s


def color_codes():
    """
    Function that returns a custom dictionary with ASCII codes related to
    colors.
    :return: Custom dictionary with ASCII codes for terminal colors.
    """
    codes = {
        'nc': '\033[0m',
        'b': '\033[1m',
        'k': '\033[0m',
        '0.25': '\033[30m',
        'dgy': '\033[30m',
        'r': '\033[31m',
        'g': '\033[32m',
        'gc': '\033[32m;0m',
        'bg': '\033[32;1m',
        'y': '\033[33m',
        'c': '\033[36m',
        '0.75': '\033[37m',
        'lgy': '\033[37m',
        'clr': '\033[K',
    }
    return codes


def find_file(name, dirname):
    """

    :param name:
    :param dirname:
    :return:
    """
    result = list(filter(
        lambda x: not os.path.isdir(x) and re.search(name, x),
        os.listdir(dirname)
    ))

    return os.path.join(dirname, result[0]) if result else None


def get_dirs(path):
    """
    Function to get the folder name of the patients given a path.
    :param path: Folder where the patients should be located.
    :return: List of patient names.
    """
    # All patients (full path)
    patient_paths = sorted(
        filter(
            lambda d: os.path.isdir(os.path.join(path, d)),
            os.listdir(path)
        )
    )
    # Patients used during training
    return patient_paths


def remove_small_regions(img_vol, min_size=3):
    """
        Function that removes blobs with a size smaller than a minimum from a mask
        volume.
        :param img_vol: Mask volume. It should be a numpy array of type bool.
        :param min_size: Minimum size for the blobs.
        :return: New mask without the small blobs.
    """
    blobs, _ = nd.measurements.label(
        img_vol,
        nd.morphology.generate_binary_structure(3, 3)
    )
    labels = list(filter(bool, np.unique(blobs)))
    areas = [np.count_nonzero(np.equal(blobs, lab)) for lab in labels]
    nu_labels = [lab for lab, a in zip(labels, areas) if a >= min_size]
    nu_mask = reduce(
        lambda x, y: np.logical_or(x, y),
        [np.equal(blobs, lab) for lab in nu_labels]
    ) if nu_labels else np.zeros_like(img_vol)
    return nu_mask


def border_point(image, point, margin=100):
    """
    Function to check if a point (and its area) is in the border of the image.
    :param image: Binary image.
    :param point: Point to check.
    :param margin: Margin of error.
    :return:
    """
    top = np.array(image.shape)
    np.logical_or(point < margin, (top - point) < margin).any()

    return np.logical_or(point < margin, (top - point) < margin).any()


def list_from_binary(filename):
    """
    Function to take a binary image and output the center of masses of its
     connected regions.
    :param filename: Name of the file containing the binary image.
    :return:
    """
    # Open filename
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    point_list = list_from_image(im) if im is not None else []
    return point_list


def list_from_image(im):
    mask = cv2.threshold(255 - im, 40, 255, cv2.THRESH_BINARY)[1]
    return list_from_mask(mask)


def list_from_mask(mask):
    # Compute connected components
    n_labels, _, _, centroids = cv2.connectedComponentsWithStats(mask)
    # print(
    #     "crownSegmenterEvaluator, found {:d} {:d} points"
    #     " for file {:}".format(n_labels, len(centroids), filename)
    # )

    # print(" listFromBinary, found  {:d}".format(len(centroids)))
    # print(centroids)

    new_centroids = [c for c in centroids if not border_point(mask, c)]
    # print(" listFromBinary, refined  {:d}".format(len(new_centroids)))
    # print(new_centroids)

    return new_centroids[1:]