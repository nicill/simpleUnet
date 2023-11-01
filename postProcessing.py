# file to Improve the predicted masks from a semantic segmentation algorithm of drone images of a forest
# We will receive the label mask,
import numpy as np
import sys
import cv2
import os
#from osgeo import gdal

#given a window, compute its average
def averageWindow(window): return np.average(window)

def sliding_window(image, stepSize, windowSize, allImage=False):
    if allImage: yield(0,0,image[:,:])
    else:
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def filterByWindow(dem,mask,wSize, th):
    # slide a window
    for (x, y, window) in sliding_window(dem, stepSize=int(wSize/2), windowSize=(wSize, wSize)):
        if window[np.nonzero(window)].any():
             maskWindow=mask[y:y + wSize, x:x + wSize]
             min=np.min(window[np.nonzero(window)])
             max=np.max(window[np.nonzero(window)])
             #print("window "+str(x)+" "+str(y)+" min: "+str(min)+"  max: "+str(max)+" to "+str(min+th*(max-min)))
             maskWindow[window<(min+th*(max-min)) ]+=1

def thresholdLowPixels(dem,th):
    #print("thresholding with "+str(th))
    retVal=dem.copy()
    retVal[dem<th]=0
    retVal[dem>=th]=255
    return retVal

def readDEM(file):

    dem = cv2.imread(file,cv2.IMREAD_UNCHANGED)
    if dem is None:raise Exception("no DEM at "+str(file))
    return dem

"""
ABANDONED DUE TO GDALs inability to install normally (or my inability to understand how to do that)
def readDEM(file):
    dem2 = gdal.Open(file, gdal.GA_ReadOnly)
    #for x in range(1, dem2.RasterCount + 1):
    #    band = dem2.GetRasterBand(x)
        #cv2.imwrite("./band"+str(x)+".png",band.ReadAsArray())
    #    array = band.ReadAsArray().astype(np.float)
        #print(array)

    dem=dem2.GetRasterBand(dem2.RasterCount).ReadAsArray().astype(float)
    dem[dem<0]=0 #eliminate non values
    #minDem=np.min(dem[np.nonzero(dem)])
    #maxDem = np.max(dem[np.nonzero(dem)])
    #print(" shape of the DEM!!!!"+str(dem.shape))
    #print(minDem)
    #print(maxDem)
    if dem is None:raise Exception("no DEM at "+str(file))
    return dem
"""

def demThresholdMask(dem,threshold):
    returnImage=dem.copy()
    minDem=np.min(dem[np.nonzero(dem)])
    returnImage[dem<minDem+threshold]=255
    returnImage[dem>=minDem+threshold]=0
    return returnImage

def expandMaskAroundFloor(mask,dem,wSize,th):
    #move a sliding window
    for (x, y, window) in sliding_window(dem, stepSize=wSize, windowSize=(wSize, wSize)):
        # for nonempty windows
        if window[np.nonzero(window)].any():
             # consider the top "non spurious" floor altitude in the window
             maskWindow=mask[y:y + wSize, x:x + wSize]
             if maskWindow[np.nonzero(maskWindow)].any():
                 floorMarker=np.percentile(window[maskWindow==255],90)
                 print("marker at "+str(floorMarker)+" max at "+str(np.max(dem))+" expanding to "+str((floorMarker+th)))
                 #Now expand the floor
                 maskWindow[window<(floorMarker+th)]=255
                 maskWindow[window==0]=0


def storePretty(mask, name):
    max=np.max(mask)
    mask=mask*(int(255/max))
    cv2.imwrite(name,mask)

def demtoJPG(dem,fileName):
    aux=dem.copy()
    dem=dem-np.min(dem[np.nonzero(dem)])
    dem[aux==0]=0
    cv2.imwrite(fileName,dem.astype("int"))

if __name__ == '__main__':

    # Read a DEM file and a results mask
    demFile=sys.argv[1]
    roiFile=sys.argv[2]

    dem=readDEM(demFile)
    roi=cv2.imread(roiFile,cv2.IMREAD_GRAYSCALE)
    if roi is None:raise Exception("NO ROI found in site "+roiFile+"")

    #ignore anything outside of the ROI
    dem[roi==255]=0
    demtoJPG(dem,"demjpg.jpg")

    sys.exit()

    th=[0.1,0.25,0.2,0.1]
    w=[1,2,5,10]
    wsizes=[250,500,750,1000]
    totalMask=np.zeros((dem.shape[0],dem.shape[1]))


    for i,wsize in enumerate(wsizes):
        mask=np.zeros((dem.shape[0],dem.shape[1]))
        filterByWindow(dem,mask,wsize,th[i])
        mask[dem==0]=0
        totalMask=w[i]*mask+totalMask
    max=np.max(totalMask)
    storePretty(totalMask,"pap.jpg")

    #totalMask=cv2.imread("pap.jpg",0)
    max=np.max(totalMask)
    print(max)
    print(int(0.9*max))
    maskToExpand=totalMask.copy()
    maskToExpand[totalMask<int(0.9*max)]=0
    maskToExpand[maskToExpand!=0]=255

    storePretty(maskToExpand,"pep.jpg")

    expandMaskAroundFloor(maskToExpand,dem,500,10)
    storePretty(maskToExpand,"pop.jpg")
