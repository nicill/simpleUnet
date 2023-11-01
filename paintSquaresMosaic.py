import numpy as np
import sys
import dice
import cv2
import os

# Function to take a binary image and output the center of masses of its connected regions
def listFromBinary(im):

	if im is None: return []
	else:
		mask = cv2.threshold(255-im, 40, 255, cv2.THRESH_BINARY)[1]

		# compute connected components
		numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(mask)

		return centroids[1:]

def paintSquares(centroids,maskList,mosaic):

    # For each mask, find what centroids are black in it
    colorList=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(100,0,0),(0,100,0),(0,0,100),(100,100,0),(0,100,100),(100,0,100)]

    color=0
    rectSize=128
    for m in maskList:
        for i in range(len(centroids)):
            cent=centroids[i]
            if m[int(cent[1]),int(cent[0])]>200:# the centroid belongs to this layer
                #mosaic= cv2.circle(mosaic, (int(cent[0]), int(cent[1])), 10, colorList[color], 5)
                corner1=(int(cent[0]-rectSize/2), int(cent[1]-rectSize/2))
                corner2=(int(cent[0]+rectSize/2), int(cent[1]+rectSize/2))
                mosaic= cv2.rectangle(mosaic, corner1,corner2,colorList[color], 3)



        color+=1
    cv2.imwrite("outputImage.jpg", mosaic)



def main(argv):

    # mosaic in the first parameter
    mosaicFile=argv[1]
    image = cv2.imread(mosaicFile, cv2.IMREAD_COLOR)
    if image is None:raise Exception(" mosaic not read")

    #tree tops mask in the second
    topsFile=argv[2]
    tops = cv2.imread(topsFile, cv2.IMREAD_GRAYSCALE)
    if tops is None:raise Exception(" tops not read")
    centroids=listFromBinary(tops)

    speciesMasks=[]
    for i in range(3,len(argv)):
        currentMask=cv2.imread(argv[i], cv2.IMREAD_GRAYSCALE)
        if currentMask is None:raise Exception("mask "+str(i)+" not read")

        speciesMasks.append(currentMask)

    paintSquares(centroids,speciesMasks,image)

    #print(mosaicFile)
    #print(topsFile)
    #print("Masks")
    #print(speciesMaskFiles)
    #cv2.imwrite(os.path.join(dataPath,outMaskName+".jpg"),outMask)



if __name__ == '__main__':
    main(sys.argv)
