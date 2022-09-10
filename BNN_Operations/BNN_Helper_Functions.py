from glob import glob
from math import floor
from platform import architecture

totalMul = 0
totalSUB = 0
totalAdd = 0
totalComp = 0
total = 0

def cnnBackProp(CNN):
    global totalAdd, totalComp, totalMul, totalSUB, total     
    fcBackProp(CNN[10])
    trueCount = countTrue(CNN)
    count = 0
    Dimensions = CNN[0]
    convFilter = CNN[5]
    poolKernel  = CNN[6]
    for i in range(len(CNN[1])):
        if(CNN[1][i] == True):
            count += 1
            numFiltersUsed = CNN[2][i]
            biasLossOpp(Dimensions , CNN[4], convFilter, numFiltersUsed)
            Dimensions = kernelLossOpp(Dimensions , CNN[4], convFilter, numFiltersUsed)
            updateBiasOpp(Dimensions)
            updateFiltersOpp(convFilter, numFiltersUsed)
            convFilter[2] = Dimensions[2]
            
            #dL/dX is not calculated in the last layer because dL/dZ value is calculated from the back propagation of the 
            #fully connected network layer and that’s why dL/dX should only be calculated in all other convolution layers except last
            if(count < trueCount):
                inputLossOpp(Dimensions, convFilter, convFilter[2])
        else:
            Dimensions = maxPoolBPHelp(Dimensions, CNN[7], CNN[8], poolKernel, CNN[9])
    
    #the next two lines are used for "EvaluatingBNN.py"
    total = totalAdd + totalMul + totalSUB
    return total


# approximates the number of multiplications/additions of dL/dK of one kernel/filter
def kernelLossOpp(inputDimensions, padding, filter, numFiltersUsed):
    Wo = floor((inputDimensions[0] - filter[0] + 2*padding)) + 1
    Ho = floor((inputDimensions[1] - filter[1] + 2*padding)) + 1

    stride = 1
    padding = 0

    #[wo, Ho, 1] was passed as dL/dZ because they will always have the same size.
    bpConvolutionOpp(inputDimensions, stride, padding, [Wo, Ho, 1], numFiltersUsed)
    
    return [Wo, Ho, numFiltersUsed]
# approximates the number of multiplications/additions of dL/dB.
def biasLossOpp(inputDimensions, padding, filter, numFiltersUsed):
    global totalAdd
    Wo = floor((inputDimensions[0] - filter[0] + 2*padding)) + 1
    Ho = floor((inputDimensions[1] - filter[1] + 2*padding)) + 1
    
    ADDop = ((Wo * Ho) - 1) * numFiltersUsed
    totalAdd = totalAdd + ADDop 
    
    
    #Approximates number of Operations for FC Backpropagation of a NN
def fcBackProp(NN): 
    global totalAdd, totalMul, totalSUB
    numWeights = 0
    numBias = 0
    MULop = 0
    ADDop = 0
    SUBop = 0
    for x in range(len(NN) - 1):
        numWeights = numWeights + (NN[x]*NN[x+1])
    for x in range(len(NN)):
        numBias = numBias + (NN[x])

    #for dl/dw 
    MULop = MULop + (numWeights * 2)
    #for dl/db
    MULop = MULop + numBias

    #for calculating the cost / error --> 2*(Ypred-Yactual)
    MULop = MULop + 1
    SUBop = SUBop + 1

    totalMul = totalMul + MULop
    totalAdd = totalAdd + ADDop
    totalSUB = totalSUB + SUBop

    updateWeightBias(NN)

#counter of "True" values in list
def countTrue(CNN):
    count = 0
    for i in CNN[1]:
        if (i == True):
            count += 1
    return count


#approximates how many weight and bias updates take place, w <- w - lambda*(dl/dw)
def updateWeightBias(NN):
    global totalAdd, totalMul, totalSUB
    numWeights = 0
    numBias = 0
    MULop = 0
    SUBop = 0

    for x in range(len(NN) - 1):
        numWeights = numWeights + (NN[x]*NN[x+1])
    for x in range(len(NN)):
        numBias = numBias + (NN[x])

    #for learning rate multiplication 
    MULop = MULop + numWeights
    MULop = MULop + numBias

    #for updating weight and bias
    SUBop = SUBop + numWeights
    SUBop = SUBop + numBias

    totalMul = totalMul + MULop
    totalSUB = totalSUB + SUBop

#approximates # of operations for the update process of the Bias B
def updateBiasOpp(Dimensions):
    global totalMul, totalSUB
    MULop = 0
    SUBop = 0
    #for learning rate multiplication since there is a different Bias for each value in matrix Z(output-matrix)     
    MULop = MULop + Dimensions[0]*Dimensions[1]*Dimensions[2]
    #for updating kernal (subtracting with gradient)
    SUBop = SUBop + Dimensions[0]*Dimensions[1]*Dimensions[2]

    totalMul = totalMul + MULop
    totalSUB = totalSUB + SUBop

  
    
# approximates the number of multiplications/additions of dL/dX.
def inputLossOpp(inputDimensions, filter, depth):    
    stride = 1
    padding = 0
    bpConvolutionOpp([(inputDimensions[0] + 1), (inputDimensions[1] + 1), inputDimensions[2]], stride, padding, filter, depth)
  

#Help funtion to calculate dimensions of output matrix after max pool used for BP operations calculation
def maxPoolBPHelp(inputDimensions, stride, padding, filter, dilation):
    Wo = floor((inputDimensions[0] + 2*padding - dilation*(filter[0] - 1) - 1) / stride) + 1
    Ho = floor((inputDimensions[1] + 2*padding - dilation*(filter[1] - 1) - 1) / stride) + 1

    return [Wo, Ho, inputDimensions[2]]

#Approximates number of Operations for CNN Convolution 
def bpConvolutionOpp(inputDimensions, stride, padding, filter, numFiltersUsed):
    global totalAdd, totalMul
    Wo = floor((inputDimensions[0] - filter[0] + 2*padding) / stride) + 1
    Ho = floor((inputDimensions[1] - filter[1] + 2*padding) / stride) + 1

    MULop = filter[0] * filter [1] * filter[2] * Wo * Ho * numFiltersUsed
    ADDop = ((filter[0] * filter[1] * filter[2]) - 1)  * Wo * Ho * numFiltersUsed
        
    totalMul = totalMul + MULop
    totalAdd = totalAdd + ADDop
    
    return [Wo, Ho, numFiltersUsed]
#approximates # of operations for the update process of the filter
def updateFiltersOpp(convFilter, numFiltersUsed):
    global totalMul, totalSUB
    MULop = 0
    SUBop = 0
    #for learning rate multiplication 
    MULop = MULop + (convFilter[0] * convFilter [1] * convFilter[2] * numFiltersUsed)
    #for updating kernal
    SUBop = SUBop + (convFilter[0] * convFilter [1] * convFilter[2] * numFiltersUsed)

    totalMul = totalMul + MULop
    totalSUB = totalSUB + SUBop


#Parent function which is meant to be called from the user.
def approximateNumberOfOperations(networkArchitecture):
    cnnBackProp(networkArchitecture)

VGG3 = [[28, 28, 1], [True, False, True, False], [64, 0, 64, 0], 1, 1, [3, 3, 1], [2,2], 2, 1, 1, [64, 2048, 10]]
VGG7 = [[32, 32, 3], [True, True, False, True, True, False, True, True, False], [128, 128, 0, 256, 256, 0, 512, 512, 0], 1, 1, [3, 3, 3], [2,2], 2, 1, 1, [512, 1024, 10]]

#please uncomment one at at a time or use your own network architecture
#REALLY IMPORTANT NOTE: COMMENT ALL approximateNumberOfOperations(X) calling functions so that "EvaluatingBNN,py" can work properly
approximateNumberOfOperations(VGG3)