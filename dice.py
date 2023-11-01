import sys, getopt
import numpy as np
from PIL import Image
from PIL import ImageOps
import warnings
import cv2

#return the percentage of black pixels in the ground truth covered by the predicted mask
def coveredPerc(gtMask,predictedMask):
    total=np.sum(gtMask == 255)
    auxMask=gtMask.copy()
    #first, compute initial GT pixels
    #print("total gt pixels "+str(total))
    auxMask[predictedMask==255]=0
    covered=np.sum(auxMask == 255)
    #print("total covered gt pixels "+str(covered))

    if total!=0: return 1-(covered/total)
    else: return 0

def FPPerc(gtMask,predictedMask):
    #first, compute initial GT pixels
    total=np.sum(gtMask == 255)
    auxMask=predictedMask.copy()
    auxMask[gtMask==255]=0
    FP=np.sum(auxMask == 255)
    if total!=0: return FP/total
    else: return 0

def decidedPercentage(mask1,mask2, ROI=None):# return the percentage of pixels with the same value, the first mask is the result and the second the GT
    aux=mask1.copy()
    aux[ROI==0]=0
    totalPixels=np.sum(ROI!=0)-np.sum(aux==255)
    if np.sum(ROI!=0)!=0:return 100*totalPixels/np.sum(ROI!=0)
    else: return 0

def equalValue(mask1,mask2, ROI=None):# return the percentage of pixels with the same value, the first mask is the result and the second the GT
    try:
        # the ROI has 1 inside 0 outside
        aux=mask1.copy()
        aux[ROI==0]=0
        #compute equal value only in decided pixels
        totalPixels=np.sum(ROI!=0)-np.sum(aux==255)
        #totalPixels=np.sum(ROI!=0)
        #print("decided Percentage: "+str(100*totalPixels/np.sum(ROI!=0)))
        #print("total Pixels in ROI "+str(totalPixels))
    except Exception as e:
        raise Exception("NO ROI"+str(e))
        #totalPixels=mask1.shape[0]*mask1.shape[1]

    im=mask1-mask2 # subtract masks to find out equal pixels

    #consider only inside the roi
    im[ROI==0]=255
    im[im!=0]=255

    #print(np.sum(im == 0))

    if totalPixels!=0: return np.sum(im == 0)/totalPixels
    else: return 0

#recall, how many positives did we catch
def RecallLabelI(gt,predicted,i):# return the percentage of pixels of class i correctly matched over the total of positives
    sumEquals=np.sum(gt == i)
    if sumEquals!=0:return np.sum(np.logical_and(gt == i, predicted == i)) / sumEquals
    else:return 0

#percentage of incorrectly classified pixels belonging to class i
def incorrectPercLabelI(gt,predicted,i):
    totalIncorrect = np.sum(gt != predicted)
    totalMissedI = np.sum(np.logical_and(gt == i, predicted != i))

    if totalIncorrect !=0 : return totalMissedI/totalIncorrect
    else: return 1

#percentage of pixels belonging to class i
def pixelPercLabelI(gt,i):
    totalI = np.sum(gt == i)
    total = np.sum(gt != 0)

    if total !=0 : return totalI/total
    else: return 1

#percentage of pixels incorrectly attributed to class i
def incorrectPercToI(gt,predicted,i):
    totalIncorrect = np.sum(gt != predicted)
    totalWrongToI = np.sum(np.logical_and(gt != i, predicted == i))

    if totalIncorrect !=0 : return totalWrongToI/totalIncorrect
    else: return 1

def confItoJ(gt,predicted,i,j):
    return np.sum(np.logical_and(gt == i, predicted == j))

#precision, how many of those predicted are correct
def PrecisionLabelI(gt,predicted,i):# return the percentage of pixels of class i incorrectly matched over the total of positives

    TP=np.sum(np.logical_and(gt == i, predicted == i))
    FP=np.sum(np.logical_and(gt != i, predicted == i))
    if (TP+FP)!=0 : return TP/(TP+FP)
    else: return 0

def DiceLabelI(gt,predicted,i):# return the percentage of pixels of class i incorrectly matched over the total of positives

    TP=np.sum(np.logical_and(gt == i, predicted == i))
    FP=np.sum(np.logical_and(gt != i, predicted == i))
    FN=np.sum(np.logical_and(gt == i, predicted != i))
    #2TP/(2TP+FP+FN)
    if ((2*TP)+FP+FN)!=0 : return 2*TP/((2*TP)+FP+FN)
    else: return 0

def CorrectLabelI(gt,predicted,i):# return the percentage of pixels of class i correctly predicted

    TP=np.sum(np.logical_and(gt == i, predicted == i))
    #FP=np.sum(np.logical_and(gt != i, predicted == i))
    #FN=np.sum(np.logical_and(gt == i, predicted != i))
    #2TP/(2TP+FP+FN)
    return TP


def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    #print("im1: " + repr(im1.shape))
    #print("im2: " + repr(im2.shape))

    if im1.shape != im2.shape:
        print("im1: " + repr(im1.shape))
        print("im1: " + repr(im1.shape))
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

#print ("Dice coefficient between two boolean images")
#print ("-------------------------------------------")



def main(argv):
    warnings.simplefilter('ignore', Image.DecompressionBombWarning)

    im1 = Image.open(sys.argv[1])
    im2 = Image.open(sys.argv[2])

    # Error control if only one parameter for inverting image is given.
    if len(sys.argv) == 4:
        print("Invert options must be set for both images")
        sys.exit()

    # Inverting input images if is wanted.
    if len(sys.argv) > 3:
        if int(sys.argv[3]) == 1:
            im1 = ImageOps.invert(im1)

        if int(sys.argv[4]) == 1:
            im2 = ImageOps.invert(im2)

    # Image resize for squared images.
    size = 300, 300
    # im1.thumbnail(size, Image.ANTIALIAS)
    # im1.show()
    # im2.thumbnail(size, Image.ANTIALIAS)
    # im2.show()

    # Converting to b/w.
    gray1 = im1.convert('L')
    im1 = gray1.point(lambda x: 0 if x < 128 else 255, '1')
    gray2 = im2.convert('L')
    im2 = gray2.point(lambda x: 0 if x < 128 else 255, '1')

    # Dice coeficinet computation
    res = dice(im1, im2)

    print(res)

if __name__ == "__main__":

    main(sys.argv[1:])
