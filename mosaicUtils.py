# file to read a mosaic file and annotations files and divide them along with ROI and treetops into smaller versions
import numpy as np
import sys
import dice
import cv2
import os
from postProcessing import readDEM

def subdivideImage(image,siteName,outputDir,subdiv):
    #print("subdividing ")
    #print(image[0])
    #print(siteName)
    #print(subdiv)

    if image[0]=="ROI": fullImage=cv2.imread(image[1],cv2.IMREAD_GRAYSCALE)
    elif image[0]=="DEM":
        #print("reading "+str(image[1]))
        fullImage=readDEM(image[1])
    else: fullImage=cv2.imread(image[1])

    for i,s in enumerate(subdiv):

        newSiteName=siteName[0:3]+"Div"+str(i)+siteName[3:]
        #print("making "+newSiteName)

        dirName=outputDir+"/"+newSiteName+"/"
        newImage=fullImage[s[0]:s[1],s[2]:s[3]]
        if image[0]=="DEM":newImageName=dirName+newSiteName+image[0]+".tif"
        else: newImageName=dirName+newSiteName+image[0]+".jpg"

        # Add a border to the roi
        #print(newImage.shape)
        if image[0]=="ROI":
            border = 25
            newImage[-border:,:]=255
            newImage[:border,:]=255
            newImage[:,-border:]=255
            newImage[:,:border]=255

        #print(newImageName)
        cv2.imwrite(newImageName,newImage)

def processSite(folder,siteName,spList,nSteps,outFolder):
    fileList=os.listdir(folder)
    #print(fileList)

    roiFile=folder+"/"+siteName+"ROI.jpg"
    roi=cv2.imread(roiFile,cv2.IMREAD_GRAYSCALE)
    if roi is None:raise Exception("NO ROI found in site "+roiFile+"")
    roi[roi>50]=255
    roi[roi<50]=0

    treeTopsFile=folder+"/"+siteName+"Treetops_ROI.jpg"
    mosaicFile=folder+"/"+siteName+".jpg"
    demFile=folder+"/"+siteName+"DEM.tif"

    # Find contours
    cnts = cv2.findContours(255-roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Find extreme coordinates of ROI
    cnt=cnts[0]
    x,y,w,h = cv2.boundingRect(cnt)

    xStep=int(w/nSteps)
    yStep=int(h/nSteps)

    subdiv=[]
    for i in range(nSteps):
        for j in range(nSteps):
            #roiPart=roi[y+j*yStep:y+(j+1)*yStep,x+i*xStep:x+(i+1)*xStep]
            subdiv.append((y+j*yStep,y+(j+1)*yStep,x+i*xStep,x+(i+1)*xStep))

            #print(str(i)+" "+str(j))
            #cv2.imwrite(str(i)+str(j)+"shit.png",roiPart)

    #now that we have the subdivisions that we want to make, add all the images that we want to sudvivide
    imagesToSubdivide=[("ROI",roiFile),("Treetops_ROI",treeTopsFile),("",mosaicFile),("DEM",demFile)]
    for i in range(len(spList)):
        #print("label "+str(i))
        currentMaskFile=folder+"/"+siteName+spList[i]+".jpg"
        currentMask=cv2.imread(currentMaskFile,cv2.IMREAD_GRAYSCALE)
        if currentMask is None:
            #print("species "+spList[i]+" not present in site "+siteName)
            pass
        else: imagesToSubdivide.append((spList[i],currentMaskFile))

    #print(imagesToSubdivide)
    for i,s in enumerate(subdiv):
        newSiteName=siteName[0:3]+"Div"+str(i)+siteName[3:]
        dirName=outFolder+"/"+newSiteName+"/"
        #print("should be making dir "+dirName)
        try:
            # Create target Directory
            os.mkdir(dirName)
            print("Directory " , dirName ,  " Created ")
        except FileExistsError:
            print("Directory " , dirName ,  " already exists")


    for image in imagesToSubdivide:
        #print("shit")
        #print(image)
        subdivideImage(image,siteName,outFolder,subdiv)

if __name__ == '__main__':

    # Read a directory that contains sites and another for the output
    inputDir=sys.argv[1]
    outDir=sys.argv[2]
    subdivisions=int(sys.argv[3]) # must be power of 2

    maxSpeciesCode=46
    speciesDict={}
    speciesList=["S"+'{:02d}'.format(i) for i in range(0,maxSpeciesCode+1)]
    for sp in range(len(speciesList)):speciesDict[speciesList[sp]]=sp

    # for all the folder in the inDIr, consider all annotation files, the ROI file and the treetop files. 1) Compute bounding box of the ROI
    # 2) break everything into "n" parts according to the ROI bounding box and store in the output folder
    siteFolders=sorted(os.listdir(inputDir))
    print(siteFolders)

    for x in siteFolders:
        processSite(inputDir+x,x,speciesList,int(subdivisions/2),outDir)
