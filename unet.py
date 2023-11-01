import argparse
import os
import sys
import re
import cv2
import time
import numpy as np

from torch.utils.data import DataLoader
from utils import color_codes, find_file
from datasets import Cropping2DDataset
from models import Unet2D
#from metrics import hausdorf_distance, avg_euclidean_distance
#from metrics import matched_percentage
from utils import list_from_mask

from postProcessing import readDEM

def checkGT(folder,siteName,sList):
    fileList=os.listdir(folder)
    #print(fileList)

    #go over each folder, if a GT file does not exist, create it
    if not(siteName+"GT.png" in fileList):
        print("no GT! "+siteName+"GT.png")
        #read ROI
        roiFile=folder+"/"+siteName+"ROI.png"
        print(roiFile)
        roi=cv2.imread(roiFile,cv2.IMREAD_GRAYSCALE)
        if roi is None:raise Exception("NO ROI found in site "+roiFile+"")
        mask=roi.copy()
        mask[roi>0]=0 #now the mask is completely black
        # Maybe we could put here something to indicate background and ROI

        #now go over the list of sites and add the code of the classes present
        for i in range(len(sList)):
            print("label "+str(i+1))
            currentMaskFile=folder+"/"+siteName+sList[i]+".png"
            currentMask=cv2.imread(currentMaskFile,cv2.IMREAD_GRAYSCALE)
            if currentMask is None: print("species "+sList[i]+" not present in site "+siteName)
            else:
                # apply the ROI to the mask just in case
                currentMask[roi==0]=0
                mask[currentMask>150] = i+1 #masks are white on black background
                currentMask[roi>0]=0
                #mask[currentMask<150] = (i+1)*50 #masks are black on white background
                # notice that the ith layer is named (and labeled lable i+1)

        #print(np.unique(mask))
        cv2.imwrite(folder+"/"+siteName+"GT.png",mask)

def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')

    # Mode selector
    parser.add_argument(
        '-d', '--mosaics-directory',
        dest='data_dir', # default='/home/mariano/Dropbox/DEM_Annotations',
        default='/home/mariano/Dropbox/280420',
        help='Directory containing the mosaics'
    )
    parser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int,  default=20,
        help='Number of epochs'
    )
    parser.add_argument(
        '-aug', '--augment',
        dest='augment',
        type=int,  default=0,
        help='augmentations per image'
    )
    parser.add_argument(
        '-dec', '--decrease',
        dest='decrease',
        type=int,  default=0,
        help='decreasing percentage per 100 images'
    )
    parser.add_argument(
        '-p', '--patience',
        dest='patience',
        type=int, default=5,
        help='Patience for early stopping'
    )
    parser.add_argument(
        '-B', '--batch-size',
        dest='batch_size',
        type=int, default=32,
        help='Number of samples per batch'
    )
    parser.add_argument(
        '-t', '--patch-size',
        dest='patch_size',
        type=int, default=256,
        help='Patch size'
    )
    parser.add_argument(
        '-l', '--labels-tag',
        dest='lab_tag', default='top',
        help='Tag to be found on all the ground truth filenames'
    )


    options = vars(parser.parse_args())

    return options

"""
Networks
"""
def train(cases, gt_names, roiNames, net_name,  nClasses=4, verbose=1):
    # Init
    print("\n\n\n\n STARTING TRAIN  ")
    options = parse_inputs()

    d_path = options['data_dir']
    c = color_codes()
    n_folds = len(gt_names)
    print("starting cases")
    print(cases)

    print("reading GT ")
    print(gt_names)

    augment=parse_inputs()['augment']
    decreaseRead=parse_inputs()['decrease']
    decrease=decreaseRead/100.

    y=[]
    counter=0
    for im in gt_names:
        image=cv2.imread(im,cv2.IMREAD_GRAYSCALE)
        if image is None: raise Exception("not read "+im)

        counter+=1
        y.append(image.astype(np.uint8))

    mosaics = [cv2.imread(c_i) for c_i in cases]

    print(roiNames)
    rois = [(cv2.imread(c_i,cv2.IMREAD_GRAYSCALE) < 100).astype(np.uint8) for c_i in roiNames]

    #Print Unique values in the labels files
    print("Unique labels in each of the ground truth files")
    for yi in y: print(np.unique(yi))

    originalSizes= []
    for c_i in cases:
        nowIm=cv2.imread(c_i)
        originalSizes.append((nowIm.shape[1],nowIm.shape[0]))

    # Number of Channels
    numChannels=3
    print("Number of channels is "+str(numChannels))
    # technical thing to communicate between opencv and pytorch way of storing things
    x = [np.moveaxis(mosaic,-1, 0).astype(np.float32) for mosaic in mosaics]

    mean_x = [np.mean(xi.reshape((len(xi), -1)), axis=-1) for xi in x]
    std_x = [np.std(xi.reshape((len(xi), -1)), axis=-1) for xi in x]
    norm_x = [
        (xi - meani.reshape((-1, 1, 1))) / stdi.reshape((-1, 1, 1))
        for xi, meani, stdi in zip(x, mean_x, std_x)
    ]

    print(
        '%s[%s] %sStarting cross-validation (leave-one-mosaic-out)'
        ' - %d mosaics%s' % (
            c['c'], time.strftime("%H:%M:%S"), c['g'], n_folds, c['nc']
        )
    )
    training_start = time.time()
    print(cases)
    for i, case in enumerate(cases):

        if verbose > 0:
            print(
                '%s[%s]%s Starting training for mosaic %s %s(%d/%d)%s' %
                (
                    c['c'], time.strftime("%H:%M:%S"),
                    c['g'], case,
                    c['c'], i + 1, len(cases), c['nc']
                )
            )

        test_x = norm_x[i]
        train_y = y[:i] + y[i+1:]
        train_roi = rois[:i] + rois[i+1:]
        train_x = norm_x[:i] + norm_x[i+1:]

        val_split = 0.1
        batch_size = 8
        patch_size = (int(options["patch_size"]), int(options["patch_size"]))

        #patch_size = (64, 64)
        # overlap = (64, 64)
        overlap = (32, 32)
        num_workers = 1

        # We only store one model in case there are multiple flights
        model_name = "LOO"+str(i)+net_name+"augm"+str(augment)+"decrease"+str(decreaseRead)+".mdl"
        print("MODEL NAME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(model_name)

        net = Unet2D(n_inputs=len(norm_x[0]),n_outputs=nClasses)

        epochs = parse_inputs()['epochs']
        patience = parse_inputs()['patience']

        try:
            net.load_model( model_name)
        except IOError:
            # Dataloader creation
            if verbose > 0:
                n_params = sum(
                    p.numel() for p in net.parameters() if p.requires_grad
                )
                print(
                    '%sStarting training with a Unet 2D%s (%d parameters)' %
                    (c['c'], c['nc'], n_params)
                )

            n_samples = len(train_x)

            print("number of sample in train_x")
            print(n_samples)

            n_t_samples = int(n_samples * (1 - val_split))

            d_train = train_x[:n_t_samples]
            d_val = train_x[n_t_samples:]

            l_train = train_y[:n_t_samples]
            l_val = train_y[n_t_samples:]

            r_train = train_roi[:n_t_samples]
            r_val = train_roi[n_t_samples:]

            codedImportant = []
            codedUnImportant = []
            codedIgnore = []

            print('Training datasets (with validation)')
            train_dataset = Cropping2DDataset(
                d_train, l_train, r_train, numLabels=nClasses,important=codedImportant, unimportant=codedUnImportant, ignore=codedIgnore,augment=augment,decrease=decrease, patch_size=patch_size, overlap=overlap
            )

            print('Validation datasets (with validation)')
            val_dataset = Cropping2DDataset(
                d_val, l_val, r_val, numLabels=nClasses,important=codedImportant, unimportant=codedUnImportant, ignore=codedIgnore,augment=augment,decrease=decrease, patch_size=patch_size, overlap=overlap
            )


            train_dataloader = DataLoader(
                train_dataset, batch_size, True, num_workers=num_workers
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size, num_workers=num_workers
            )


            net.fit(
                train_dataloader,
                val_dataloader,
                epochs=epochs,
                patience=patience,
            )

            net.save_model( model_name)

        if verbose > 0:
            print(
                '%s[%s]%s Starting testing with mosaic %s %s(%d/%d)%s' %
                (
                    c['c'], time.strftime("%H:%M:%S"),
                    c['g'], case,
                    c['c'], i + 1, len(cases), c['nc']
                )
            )

        yi = net.test([test_x])
        pred_y = np.argmax(yi[0], axis=0)
        heatMap_y = np.max(yi[0], axis=0)

        # Un-shift class names
        #pred_y+=1

        #write the results
        resultFileName = "test"+str(i)+"augm"+str(augment)+"decrease"+str(decreaseRead)+".png"
        cv2.imwrite(resultFileName,
            (pred_y).astype(np.uint8)
        )
        print("Results FILE!!!!!!!!!!!!!!!!!!!!! "+resultFileName)

    if verbose > 0:
        time_str = time.strftime(
            '%H hours %M minutes %S seconds',
            time.gmtime(time.time() - training_start)
        )
        print(
            '%sTraining finished%s (total time %s)\n' %
            (c['r'], c['nc'], time_str)
        )

def main():

    # Init
    options = parse_inputs()
    c = color_codes()

    #Layer 0 is the background class
    maxLabelCode=3
    labelList=["Layer"+str(i) for i in range(1,maxLabelCode+1)]
    print(labelList)

    numClasses=len(labelList)
    print("Number of different classes "+str(numClasses))

    # Data loading (or preparation)
    d_path = options['data_dir']

    # check how many images (sites we have)
    siteFolders=sorted(os.listdir(d_path))
    print(siteFolders)

    #First, create ground truth files if they do not exist
    for x in siteFolders:checkGT(d_path+x,x,labelList)

    # gather names of ground truth files, RGB images (cases) and rois
    gt_names = [d_path+"/"+x+"/"+x+"GT.png" for x in siteFolders ]
    print(gt_names)

    cases = [ d_path+x+"/"+x+".png" for x in siteFolders ]
    print("\n\n"+str(cases))

    rois = [ d_path+x+"/"+x+"ROI.png" for x in siteFolders ]
    print("\n\n")
    print(rois)

    ''' <Detection task> '''
    net_name = 'semantic-unet'
    train(cases, gt_names, rois, net_name, numClasses,1)

if __name__ == '__main__':
    main()
