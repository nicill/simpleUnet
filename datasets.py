import itertools
from skimage.transform import resize as imresize
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
#from imgaug import augmenters as iaa
import random
import cv2


def augment(imageInit,roiInit,labelsInit,code,verbose=True):
    #outputFile="./fuio.jpg"
    #print("shape!!! "+str(image.shape))
    image=np.moveaxis(imageInit,0,-1)
    roi=np.moveaxis(roiInit,0,-1)
    labels=np.moveaxis(labelsInit,0,-1)

    if code==0:
        if verbose: print("Doing Data augmentation 0 (H fip) to image ")
        image_aug = iaa.Rot90(1)(image=image)
        #roi_aug = roi
        #labels_aug = labels
        roi_aug = iaa.Rot90(1)(image=roi)
        labels_aug = iaa.Rot90(1)(image=labels)
        #cv2.imwrite(outputFile,image_aug)
    elif code==1:
        if verbose: print("Doing Data augmentation 1 (V flip) to image ")
        image_aug = iaa.Flipud(1.0)(image=image)
        roi_aug = iaa.Flipud(1.0)(image=roi)
        labels_aug = iaa.Flipud(1.0)(image=labels)
        #cv2.imwrite(outputFile,image_aug)
    elif code==2:
        if verbose: print("Doing Data augmentation 2 (Gaussian Blur) to image ")
        image_aug = iaa.GaussianBlur(sigma=(0, 0.5))(image=image)
        roi_aug = roi
        labels_aug = labels
        #cv2.imwrite(outputFile,image_aug)
    elif code==3:
        if verbose: print("Doing Data augmentation 3 (rotation) to image ")
        angle=random.randint(0,45)
        rotate = iaa.Affine(rotate=(-angle, angle))
        rotateRoi=iaa.Affine(rotate=(-angle, angle),order=0)
        image_aug = rotate(image=image)
        roi_aug= rotateRoi(image=roi)
        labels_aug=rotateRoi(image=labels)
        #cv2.imwrite(outputFile,image_aug)
    elif code==4:
        if verbose: print("Doing Data augmentation 5 (contrast) to image ")
        image_aug=iaa.LinearContrast((0.75, 1.5))(image=image)
        roi_aug = roi
        labels_aug = labels
        #cv2.imwrite(outputFile,image_aug)
    elif code==5:
        if verbose: print("Doing Data augmentation 4 (elastic) to image ")
        image_aug = iaa.ElasticTransformation(alpha=(0, 1.0), sigma=0.1)(image=image)
        roi_aug = iaa.ElasticTransformation(alpha=(0, 1.0), sigma=0.1,order=0)(image=roi)
        labels_aug = iaa.ElasticTransformation(alpha=(0, 1.0), sigma=0.1,order=0)(image=labels)

        #cv2.imwrite(outputFile,image_aug)
    else:
        raise Exception("datasets.py Data augmentation technique not recognised ")

    return (np.ascontiguousarray(np.moveaxis(image_aug,-1,0)),np.ascontiguousarray(np.moveaxis(roi_aug,-1,0))),np.ascontiguousarray(np.moveaxis(labels_aug,-1,0))



def centers_to_slice(voxels, patch_half):
    slices = [
        tuple(
            [
                slice(idx - p_len, idx + p_len) for idx, p_len in zip(
                    voxel, patch_half
                )
            ]
        ) for voxel in voxels
    ]
    return slices


def get_slices(masks, patch_size, overlap):
    """
    Function to get all the patches with a given patch size and overlap between
    consecutive patches from a given list of masks. We will only take patches
    inside the bounding box of the mask. We could probably just pass the shape
    because the masks should already be the bounding box.
    :param masks: List of masks.
    :param patch_size: Size of the patches.
    :param overlap: Overlap on each dimension between consecutive patches.

    """
    # Init
    # We will compute some intermediate stuff for later.
    patch_half = [p_length // 2 for p_length in patch_size]
    steps = [max(p_length - o, 1) for p_length, o in zip(patch_size, overlap)]

    # We will need to define the min and max pixel indices. We define the
    # centers for each patch, so the min and max should be defined by the
    # patch halves.
    min_bb = [patch_half] * len(masks)
    max_bb = [
        [
            max_i - p_len for max_i, p_len in zip(mask.shape, patch_half)
        ] for mask in masks
    ]

    # This is just a "pythonic" but complex way of defining all possible
    # indices given a min, max and step values for each dimension.
    dim_ranges = [
        [
            np.concatenate([np.arange(*t), [t[1]]])
            for t in zip(min_bb_i, max_bb_i, steps)
        ] for min_bb_i, max_bb_i in zip(min_bb, max_bb)
    ]

    # And this is another "pythonic" but not so intuitive way of computing
    # all possible triplets of center voxel indices given the previous
    # indices. I also added the slice computation (which makes the last step
    # of defining the patches).
    patch_slices = [
        centers_to_slice(
            itertools.product(*dim_range), patch_half
        ) for dim_range in dim_ranges
    ]

    return patch_slices

def classesPresent(th,labIm):
    labelsPresent=[]
    for label in np.unique(labIm.astype(np.uint8)):
        if label!=-1:
            imSize=labIm.shape[0]*labIm.shape[1]
            pixTot=np.sum(labIm.astype(np.uint8)==label)
            pixRatio=(pixTot)/imSize
            #print("label "+str(label)+" had "+str(pixRatio))
            if pixRatio>th:labelsPresent.append(label)
    return labelsPresent

class Cropping2DDataset(Dataset):
    def __init__(
            self,
            data, labels, rois, numLabels,important=[], unimportant=[], ignore=[], augment=0,decrease=0,patch_size=32, overlap=16
    ):
        print("Dataset hello")
        # Init
        self.data = data
        self.labels = labels
        self.rois = rois
        data_shape = self.data[0].shape
        self.presenceThreshold=0.1
        self.numAugmentations=6

        #cv2.imwrite("./mosaic.jpg",np.moveaxis(self.data[0],0,-1))

        # create a list of pixels by label
        self.labelStats=[]
        for i in range(numLabels):self.labelStats.append([i,0,0])
        #print(self.labelStats)

        if type(patch_size) is not tuple:
            patch_size = (patch_size,) * len(data_shape)
        self.patch_size = patch_size
        self.overlap = overlap

        slices = get_slices(
            self.rois, self.patch_size, self.overlap
        )

        # Filter out patches completely outside the ROI
        self.patch_slices = [
            (s, i) for i, (roi, slices_i) in enumerate(zip(self.rois, slices))
            for s in slices_i if np.sum(roi[s]) > 0
        ]
        self.len=len(self.patch_slices)

        # Now traverse all real patches and count how many pixels of each class there are inside the ROI
        for i in range(len(self.patch_slices)):
            input,targ=self.accessRaw(i)
            im=targ
            roi=input[1]
            im[roi>150]=-1
            # now count each pixel of each class inside the patch and the ROI
            for i in range(numLabels):self.labelStats[i][1]+=np.sum(im==i)

        #compute percentages
        totalRelevantPixels=0
        for c,pix,zero in self.labelStats: totalRelevantPixels+=pix
        if totalRelevantPixels!=0:
            for x in self.labelStats: x[2]=100*x[1]/totalRelevantPixels
        print("STATS! "+str(self.labelStats))

        #now we know how many pixels are in the relevant classes
        # now perform data augmentation, go over the whole dataset
        # For every patch, the classes present are those with more than 10% of pixels
        # For patches with important classes present, augment
        # for the other patches
        # If they contain uninteresting classes, decrease
        # If they contain classes to ignore, eliminate patch
        # if not, leave the patch alone
        self.AugmList=[] # make a note of the real patch that corresponds to the index

        for i in range(len(self.patch_slices)):
            input,targ=self.accessRaw(i)
            roi=input[1]
            targ[roi>150]=-1

            classesInPatch=classesPresent(self.presenceThreshold,targ)
            #print(classesInPatch)
            #print(str(important)+" "+str(any(i in classesInPatch for i in important)))
            #print(str(unimportant)+" "+str(any(i in classesInPatch for i in unimportant)))
            if any(i in classesInPatch for i in important):# patch contains some important classes
                #print("important "+str(classesInPatch))
                if augment>0:
                    for j in range(augment):self.AugmList.append((i,j))
                else:self.AugmList.append((i,0))
            elif any(i in classesInPatch for i in unimportant) and random.random()<decrease: # no important classes on patch, but unimportant classes present
                #print("*******************unimportant to be decreased "+str(classesInPatch))
                pass # patch needs to be reduced

            elif all(i in ignore for i in classesInPatch) : # all classes in the patch are ignore classes
                #print("*******************ignoring "+str(classesInPatch)+" classes to Ignore "+str(ignore))
                pass
            else: #no important or unimportant classes, keep patch as is
                #print("meh")
                self.AugmList.append((i,0))

        #print("FINISHED CREATING DATASET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"+str(self.AugmList))
        #print(self.AugmDict)
        self.len=len(self.AugmList)
        print("length was "+str(len(self.patch_slices))+" and will become "+str(self.len))

    def accessRaw(self, index):
        # We select the case
        #print("*******************accessing "+str(index)+" of "+str(self.len))
        slice_i, case_idx = self.patch_slices[index]

        # We get the slice indexes
        none_slice = (slice(None, None),)

        inputs = (
            self.data[case_idx][none_slice + slice_i].astype(np.float32),
            np.expand_dims(
                self.rois[case_idx][slice_i].astype(np.uint8), axis=0
            )
        )

        target = np.expand_dims(
            self.labels[case_idx][slice_i].astype(np.uint8), axis=0
        )

        # target_labs = bwlabeln(target.astype(np.bool))
        # tops = len(np.unique(target_labs[target.astype(np.bool)]))

        return inputs, target

    def __getitem__(self, index):
        #return self.accessRaw(index)
        #print("GETITEM         "+str(index)+" of "+str(self.len))
        if self.AugmList[index][1]==0:
        #    print("raw")
            return self.accessRaw(self.AugmList[index][0])
        else:
        #    print("COOKED")
            input,targ=self.accessRaw(self.AugmList[index][0])
            im=input[0]
            roi=input[1]
            transf=random.randint(0, self.numAugmentations-1)
            return augment(im,roi,targ,transf,False)

    def __len__(self):
        return self.len
        #return len(self.patch_slices)
