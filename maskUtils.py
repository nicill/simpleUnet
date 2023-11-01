import numpy as np
import sys
import dice
import cv2
import os

def mergeMasks(mask1,mask2):
    if mask1 is None:mask1=mask2.copy()
    else:mask1[mask2<10]=0
    return mask1

def main(argv):

    paramsFile=argv[1]
    with open(paramsFile) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            if line[:4]=="FUSE": #ignore comments and anything not startig with FUSE
                #print("Line {}: {}".format(cnt, line.strip()))
                lineSplit=line.split(" ")
                dataPath=lineSplit[1]
                outMaskName=lineSplit[2]
                outMask=None
                for i in range(3,len(lineSplit)):
                    currentMaskName=os.path.join(dataPath,lineSplit[i].strip())
                    #print(currentMaskName)
                    currentMask=cv2.imread(currentMaskName)
                    if currentMask is None: raise Exception("Not found mask "+currentMaskName)
                    outMask=mergeMasks(outMask,currentMask)

            line = fp.readline()
            cnt += 1

    cv2.imwrite(os.path.join(dataPath,outMaskName+".jpg"),outMask)


    sys.exit()




    gtMask=cv2.imread(argv[1],cv2.IMREAD_GRAYSCALE).astype("uint8")
    if gtMask is None: raise Exception("No ground Truth mask read")


    mask=cv2.imread(argv[2],cv2.IMREAD_GRAYSCALE).astype("uint8")
    if mask is None: raise Exception("No coarse mask read")

    #print(np.unique(mask))


    code=int(argv[3])
    numClasses=46

    if code==0:# Evaluate Dice coefficient
        result=dice.dice(mask,gtMask )
        print("*******************************************  dice: "+str(result))
    elif code==1:# Evaluate percentage of matched pixels (includes Background)
        result=dice.equalValue(mask,gtMask,True )
        print("*******************************************  % Pixels with equal value (inlcudes BKG): "+str(result))
    elif code==2:# Evaluate percentage of matched pixels (not including Background)
        result=dice.equalValue(mask,gtMask,False )
        print("*******************************************  % Pixels with equal value (no BKG): "+str(result))
    elif code==3:# Evaluate percentage of TPR for each class
        for i in range(numClasses+1):
            result=dice.RecallLabelI(gtMask.copy(),mask.copy(),i)
            print("******************************************* Class "+str(i)+"  Recall "+str(result))
    elif code==4:# Evaluate percentage of TPR for each class
        for i in range(numClasses+1):
            result=dice.PrecisionLabelI(gtMask.copy(),mask.copy(),i)
            print("******************************************* Class "+str(i)+"  Precision: "+str(result))


    return result


"""

    fineMask=cv2.imread(argv[3],0)
    if fineMask is None: raise Exception("No fine mask read")

    for name,mask in [("coarse",coarseMask),("fine",fineMask)]:

        currentDice=dice.dice(mask,255-gtMask )
        currentCoverPerc=dice.coveredPerc(255-gtMask,mask)
        currentFPPerc=dice.FPPerc(255-gtMask,mask)
        print("******************************************* "+name+" mask,  dice: "+str(currentDice)+" and covered Percentage "+str(currentCoverPerc)+" and FP perc "+str(currentFPPerc))
"""



if __name__ == '__main__':
    main(sys.argv)
