import sys
from scipy.spatial.distance import directed_hausdorff
from sklearn.neighbors import KDTree
import numpy as np
from utils import list_from_binary


def hausdorf_distance(list1, list2):
    """
    Function that computes the Hausdorf distance between two lists of points.

    :param list1: First list of points.
    :param list2: Second list of points.
    :return:
    """
    if len(list1) == 0 or len(list2) == 0:
        distance = -1
    else:
        distances1 = directed_hausdorff(list1, list2)[0]
        distances2 = directed_hausdorff(list2, list1)[0]
        distance = max(distances1, distances2)
    return distance


def euclidean_distances(list1, list2):
    new_list1 = np.asarray([[x, y] for x, y in list1])
    new_list2 = np.asarray([[x, y] for x, y in list2])

    # print("matched perc, first list {:}".format(len(newList1)))
    # print("matched perc, second list {:}".format(len(newList2)))

    kdt = KDTree(new_list2, leaf_size=30, metric='euclidean')
    dist, _ = kdt.query(new_list1, k=1)
    # print(dist)

    return np.squeeze(np.array(dist))


def matched_percentage(list1, list2, epsilon):
    if len(list1) == 0 or len(list2) == 0:
        percentage = -1

    else:
        distances = euclidean_distances(list1, list2)
        count = np.count_nonzero(distances < epsilon)

        percentage = 100 * (count / len(list1))

    return percentage


def avg_euclidean_distance(list1, list2):
    if len(list1) == 0 or len(list2) == 0:
        distance = -1
    else:
        distances = euclidean_distances(list1, list2)
        distance = np.mean(distances)

    return distance


def main(argv):
    # argv[1] contains the distance method
    #  (0 hausdorff, 1, matched point percentage).
    # argv[2], argv[3] contains the names of the files with the first  and
    #  second mask.
    # Further parameters may contain specific information for some methods.

    option = int(argv[1])
    file1 = argv[2]
    file2 = argv[3]

    # First, turn the binary masks of files 1 and 2 into lists of points.
    list1 = list_from_binary(file1)
    list2 = list_from_binary(file2)

    # Now, compute the distance between sets indicated by the option.
    if option == 0:
        # > Hausdorff distance between two masks
        # Second, compute hausdorff distance
        print(format(hausdorf_distance(list1, list2), '.2f'), end=" ")
    elif option == 1:
        # > Number of matched points, (we need one extra parameter epsilon)
        epsilon = float(argv[4])
        # The first file must be the ground truth
        print(
            format(matched_percentage(list1, list2, epsilon), '.2f'), end=" "
        )
    elif option == 2:
        # > Point difference
        # The ground truth file should be the first
        n_real_points = len(list1)
        n_predicted_points = len(list2)
        # print(
        #     "Real {:d} vs predicted {:d}".format(
        #         n_real_points, predictedPointsNumber
        #     )
        # )
        print(
            format(
                100 * (n_real_points - n_predicted_points) / n_real_points,
                '.2f'
            ),
            end=" "
        )
    elif option == 3:
        # > Simple point count
        print("{:d} vs {:d}".format(len(list1), len(list2)))
    elif option == 4:
        # > Average euclidean distance
        print(format(avg_euclidean_distance(list1, list2), '.2f'), end=" ")
    else:
        raise RuntimeError("crownSegmenterEvaluator, Wrong option")


if __name__ == "__main__":
    main(sys.argv)
