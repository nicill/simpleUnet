import numpy as np
import sys
import dice
import cv2
from statistics import mean

def listToString(l):
    ret=""
    for x in l:ret+=" "+str(x)
    return ret.strip()

def listToFloatList(l):
    ret=[]
    #print("STARTING "+str(l))
    for x in l:ret.append(float(x))
    return ret

def averageRowsToDict(file):
    myDict={}
    file.readline()
    for line in file:
        #print(":::::::::::::::::::::::READ "+str(line))
        k=listToString(line.strip().split(" ")[:3])
        v=mean(listToFloatList(line.strip().split(" ")[3:]))
        myDict[k]=v
    return myDict

def averageClassesInRowsToDict(file,numClasses):
    myDict={}
    file.readline()
    for line in file:
        #print(":::::::::::::::::::::::READ "+str(line))
        k=listToString(line.strip().split(" ")[:3])
        actualData=listToFloatList(line.strip().split(" ")[3:])
        lol=[]
        meanList=[]
        for i in range(numClasses):
            lol.append(actualData[i::numClasses])
            meanList.append(mean(lol[-1]))
        myDict[k]=meanList

        #print("\n lol ")
        #for i in range(len(lol)):
        #    print("Class "+str(i)+" "+str(len(lol[i]))+" "+str(lol[i]))
        #print("\n meanlist "+str(meanList))

    return myDict

def confMatrixToDict(file,numClasses):
    myDict={}
    file.readline()
    for line in file:
        #print(":::::::::::::::::::::::READ "+str(line))
        k=listToString(line.strip().split(" ")[:3])
        actualData=listToFloatList(line.strip().split(" ")[3:])
        lol=[]
        
        for i in range(numClasses*numClasses):
            lol.append(actualData[i::numClasses*numClasses])
            meanList.append(lol[-1])
        myDict[k]=meanList

        print("\n lol ")
        for i in range(len(lol)):
            print("Class "+str(i)+" "+str(len(lol[i]))+" "+str(lol[i]))
        print("\n meanlist "+str(meanList))

    return myDict


def main(argv):

    code=int(argv[1])
    dataFile=argv[2]
    numClasses=8 # this is the name of the highest class considered

    with open(dataFile, encoding = 'utf-8') as f:
        if code==0: # One value per site
            result=averageRowsToDict(f)
        elif code==1: # one value per class per site
            result=averageClassesInRowsToDict(f,numClasses)
        elif code==2: # one list of values per class per site
            result=confMatrixToDict(f,numClasses)

    if isinstance(result,dict):
        for k,v in result.items():
            print("parameters "+str(k)+" : ",end="")
            if isinstance(v,list):
                #for x in v: print("{:.2f} ".format(x),end="")
                #print(sum(v),end="")
                #print("")
                for i in range(len(v)):
                    if i % numClasses==0:print("")
                    print("{:.2f} ".format(v[i]),end="")
            else: print(v)
    else:print(str(result)+" ",end="")

    return result


if __name__ == '__main__':
    main(sys.argv)
