
# Python Program to approximate Energy, Power, Latency, and Area of a BNN Architecture 

from glob import glob
from math import ceil
from BNN_Helper_Functions import cnnBackProp


#E1 is the analyzed Energy usage of one 64X64 crossbar multiplication and addition
#CE1 is the energy from the comparator used for the binarized model
E1 = 0.012
#CE1 = 0.001
#E2 is the analyzed Energy usage of one 32-bit FPU multiplication and addition
E2 = 0.013 
#L1 is the analyzed Latency usage of one 64X64 crossbar multiplication and addition
#CL1 is the Latency from the comparator used for the binarized model
L1 = 1.909 * (10 ** -9)
#CL1 = 2.617 * (10** -9)
#L2 is the analyzed Latency usage of one 32-bit FPU multiplication and addition
L2 = 0.013 * (10 ** -9)
#A1 is the analyzed Area usage of one 64X64 crossbar multiplication and addition
#CA1 is the Area from the comparator used for the binarized model
A1 = 193
#CA1 = 2
#A2 is the analyzed Area usage of one 32-bit FPU multiplication and addition
A2 = 714

totalLatency = 0
totalEnergy = 0
totalArea = 0

# BNNForwardProp approximates the total latency, energy, power and area for a given BNN architecture in a forward pass.
# we devide beta by 64 because the crossbar size we used in the hardware design was 64
def BNNForwardProp(bnn):
    global totalArea, totalEnergy, totalLatency, E1, L1, A1
    latency = 0
    energy = 0

    for layer in bnn:
        energy = energy + ((layer[0] * ceil(layer[1]/64) * layer[2]) * E1)
    totalEnergy = energy

    for layer in bnn:
        #we devide by 64 because the number of columns in our crossbar is also 64
        latency = latency + ((ceil(layer[0]/64) * ceil(layer[1]/64) * layer[2]) * L1)
    totalLatency = totalLatency + latency
    totalArea = totalArea + (A1 * 64)
    print(totalLatency)
    print(totalEnergy)
    print(totalArea)


def BNNBackProp(networkArchitecture):
    global total
    global E2, L2, A2, totalEnergy, totalLatency, totalArea
    totalOperations = 1
    totalOperations = cnnBackProp(networkArchitecture)
    latency = 0
    energy = 0
    area = 0
    
    energy = totalOperations * E2
    totalEnergy = totalEnergy + energy
    
    latency = totalOperations * L2
    totalLatency = totalLatency + latency

    area = totalOperations * A2
    totalArea = totalArea + area
    
#Parent function which is meant to be called from the user.
def evaluateBNNModel(alphaBetaDeltaValues, networkArchitecture):
    BNNForwardProp(alphaBetaDeltaValues)
    BNNBackProp(networkArchitecture)

###########################################################################################################################################
##############################     input BNN architecture here     ##############################
##   0 element: Array of Arrays, where each index array contains alpha, beta, delta values     ##
##   1 element: CNN model architecture used                                                    ##
#################################################################################################

VGG3BNNValues = [[64, 576, 196], [2048, 3136, 1]]
VGG7BNNValues = [[128, 1152, 1024], [256, 1152, 256], [256, 2304, 256], [512, 2304, 64], [512, 4608, 64], [1024, 8192, 1]]
#architectures of VGG3 and VGG7 copied from "CNNOperations.Py"
VGG3 = [[28, 28, 1], [True, False, True, False], [64, 0, 64, 0], 1, 1, [3, 3, 1], [2,2], 2, 1, 1, [64, 2048, 10]]
VGG7 = [[32, 32, 3], [True, True, False, True, True, False, True, True, False], [128, 128, 0, 256, 256, 0, 512, 512, 0], 1, 1, [3, 3, 3], [2,2], 2, 1, 1, [512, 1024, 10]]


#evaluateBNNModel(VGG3BNNValues, VGG3)
evaluateBNNModel(VGG7BNNValues, VGG7)

print("total latency of model := ", totalLatency)
print("total Energy of model := ", totalEnergy)
print("total Area of model := ", totalArea, " LUT's")
