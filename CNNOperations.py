## Python Program to approximate number of operations in a CNN
from glob import glob
from math import floor
from platform import architecture

totalMul = 0
totalSUB = 0
totalAdd = 0
totalComp = 0

###########################################################################################################################################


#Approximates number of Operations for CNN forward Propagation
def cnnForwardProp(CNN):
    Dimensions = CNN[0]
    convFilter = CNN[5]
    poolKernel  = CNN[6]
    for i in range(len(CNN[1])):
        if(CNN[1][i] == True):
            numFiltersUsed = CNN[2][i]
            Dimensions = convolutionOpp(Dimensions , CNN[3], CNN[4], convFilter, numFiltersUsed)
            convFilter[2] = Dimensions[2]
        else:
            Dimensions = maxpoolOp(Dimensions, CNN[7], CNN[8], poolKernel, CNN[9])
    fcForwardProp(CNN[10])

#Approximates number of Operations for CNN Convolution 
def convolutionOpp(inputDimensions, stride, padding, filter, numFiltersUsed):
    global totalAdd, totalMul
    Wo = floor((inputDimensions[0] - filter[0] + 2*padding) / stride) + 1
    Ho = floor((inputDimensions[1] - filter[1] + 2*padding) / stride) + 1

    MULop = filter[0] * filter [1] * filter[2] * Wo * Ho * numFiltersUsed
    ADDop = ((filter[0] * filter[1] * filter[2]) - 1)  * Wo * Ho * numFiltersUsed
    
    #for the bias addition
    ADDop = ADDop + ((Wo * Ho) * numFiltersUsed)
    
    totalMul = totalMul + MULop
    totalAdd = totalAdd + ADDop
    
    return [Wo, Ho, numFiltersUsed]

#Approximates number of Operations for CNN max-pool
def maxpoolOp(inputDimensions, stride, padding, filter, dilation):
    global totalComp
    Wo = floor((inputDimensions[0] + 2*padding - dilation*(filter[0] - 1) - 1) / stride) + 1
    Ho = floor((inputDimensions[1] + 2*padding - dilation*(filter[1] - 1) - 1) / stride) + 1

    compOP = ((filter[0] * filter[1]) - 1) * Wo * Ho * inputDimensions[2]
    
    totalComp = totalComp + compOP    
    return [Wo, Ho, inputDimensions[2]]

###########################################################################################################################################

#approximates the number of Operations for BackProp in CNN's
def cnnBackProp(CNN):     
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
    print(total)
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

# approximates the number of multiplications/additions of dL/dB.
def biasLossOpp(inputDimensions, padding, filter, numFiltersUsed):
    global totalAdd
    Wo = floor((inputDimensions[0] - filter[0] + 2*padding)) + 1
    Ho = floor((inputDimensions[1] - filter[1] + 2*padding)) + 1
    
    ADDop = ((Wo * Ho) - 1) * numFiltersUsed
    totalAdd = totalAdd + ADDop 
    
    
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

#counter of "True" values in list
def countTrue(CNN):
    count = 0
    for i in CNN[1]:
        if (i == True):
            count += 1
    return count

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

###########################################################################################################################################


#Approximates number of Operations for FC ForwardPropagation of a NN
def fcForwardProp(NN):
    global totalAdd, totalMul
    MUL = 0
    ADD = 0
 
    for x in range(len(NN) - 1):
        MUL = MUL + (NN[x]*NN[x+1])
    ADD = MUL    

    totalMul = totalMul + MUL
    totalAdd = totalAdd + ADD


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

    
#Parent function which is meant to be called from the user.
def approximateNumberOfOperations(networkArchitecture):
    #cnnForwardProp(networkArchitecture)
    cnnBackProp(networkArchitecture)
    print("total mul operations := ", totalMul)
    print("total add operations := ", totalAdd)
    print("total sub operations := ", totalSUB)
    print("total comp operations := ", totalComp)
    
###########################################################################################################################################
##############################     input CNN architecture here     ##############################
##   0  element: initial input size                                                            ##
##   1  element: True when layer is convolution false when pooling,                            ##
##   2  element: number of filters used at each corresponding convolution layer                ##
##   3  element: convolution Stride                                                            ##
##   4  element: convolution padding                                                           ##
##   5  element: filter dimensions                                                             ##
##   6  element: Kernel size (maxPool)                                                         ##
##   7  element: max-Pool Stride                                                               ## 
##   8  element: pooling padding                                                               ##
##   9  element: dilation                                                                      ##
##   10 element: the FC Layer architecture within CNN.                                         ##
#################################################################################################

VGG3 = [[28, 28, 1], [True, False, True, False], [64, 0, 64, 0], 1, 1, [3, 3, 1], [2,2], 2, 1, 1, [64, 2048, 10]]
VGG7 = [[32, 32, 3], [True, True, False, True, True, False, True, True, False], [128, 128, 0, 256, 256, 0, 512, 512, 0], 1, 1, [3, 3, 3], [2,2], 2, 1, 1, [512, 1024, 10]]

#please uncomment one at at a time or use your own network architecture
#REALLY IMPORTANT NOTE: COMMENT ALL approximateNumberOfOperations(X) calling functions so that "EvaluatingBNN,py" can work properly
approximateNumberOfOperations(VGG3)
#approximateNumberOfOperations(VGG7)
