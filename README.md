# CNN-number-of-operations-python-tool
we have created a python tool, which calculates how many additions, multiplications, subtractions, and divisions take place within on iteration of a Convolutional neural network (forward and backward pass)

How to use the Tool? 
Simply insert the CNN model architecture found at the end of the python file named "CNNOperations"
we provided two examples of the VGG3 and VGG7 model architectures.

*************************************************************************************************************************
simple explanations for how we developed the tool:

CNN Forward Propagation:
During the Forward Propagation of CNN’s, the weights, biases and filters are randomly initialized. Within the hidden layers in a CNN features are being extracted and this can be broken down into three parts:
1.	The Convolution layers: Extracts features from the input.
2.	The Max Pool Layers: reduces spatial size of matrix.
3.	The Fully connected layers: Uses data from convolution layer to generate output.
We wont be considering batch-Norm in this part nor in the (# operations energy/latency part) because it has been showed that it can be replaced (Link a research paper on how it can be replaced)

1) The Convolution layer: 
suppose we have an input image represented as “X” (H * W * D) and a Kernel/filter represented with “K”, and “Z” as the output, then the convolution Expression would be: Z = X * K. where each filter “convolves” over the input matrix and produces a feature map from the image. Feature maps are then stacked together and used for the next layers. The resulting matrix dimensions can be calculated using the following Formula which can be found in the following link:

https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d 

Hout = ( ⌊ Hin + 2∗padding[0] − dilation[0] * (kernel_size[0] − 1) −1/stride[0] ⌋ ) + 1
Wout = ( ⌊ Win + 2∗padding[1] − dilation[1] * (kernel_size[1] − 1) −1/stride[1] ⌋ ) + 1

Keeping in mind that the depth(channels) of the resulting matrix becomes the number of filters used on the input matrix. The new dimensions are equal to:
Hout * Wout * (number of filters used on input matrix)

2) The Max-Pool Layer
It is common to insert a Pooling layer in-between successive Convolution layers. Its function is to reduce the spatial size of the matrix to reduce the number of computation and parameters within the network. It operates independently on every depth slice of the input matrix and reduces its size spatially using the “Max” Operation. The resulting output matrix after a “Max-Pool Layer” can be calculated using the same Formula above.

3) The Fully Connected Layer:

*************************************************************************************************************************

How we calculated the number of operations for a CNN?

for a Convolution Layer:
Convolution of a CNN is basically the element-wise product and sum between two matrices. And to be able to calculate how many operations it requires, we need the resulting output matrix dimensions explained in the previous section.

For approximating the number of multiplications in a convolutional layer we derived the following formula:
Equation 1:  number of multiplications = filter[0] * filter[1] * filter[2] * Wo * Ho * n
For approximating the number of additions in a convolutional layer we derived the following formula: 
Equation 2:  number of additions = ((filter[0] * filter[1] * filter[2]) - 1) * Wo * Ho * n
Where n is the number of filters used for that layer.
	
  
for the Max-pool Layer:
Max-Pool operates independently on every depth slice of the input matrix and reduces its size spatially using the “Max” Operation. Thus, it only results in comparison operations. First, we calculate the Wo, Ho, D(depth) of the resulting matrix after Max-pool layer and then use those values in the following formula:
Equation 3:  number of comparisons = ((filter[0] * filter[1]) -1) * Wo * Ho * D 

Using the above formulas and the CNN network architecture and initial values, we can derive the number of operations for both the forward and backward propagation of a CNN.

Keeping it simple,
For the Forward-Propagation: for every convolutional layer apply equations 1 and 2 And for every max-pool layer apply equation 3. with correct dimensions and values.
As for the Backward Propagation: we apply convolution equations 1, 2 for calculating ∂L/(∂F_i ),∂L/(∂X_i ) and the sum operation for ∂L/∂B which is the same as the number of biases used within the model.

∂L/(∂F_i) = Convolution(X,Loss gradient ∂L/∂O), ∂L/(∂X_i ) = Convolution(180 degree rotated filter F, padded(∂L/∂O))
∂L/∂B=∑(∂L/∂O)
for a more in-depth explanation of those formulas please check: https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c#:~:text=Chain%20Rule%20in%20a%20Convolutional%20Layer&text=For%20the%20forward%20pass%2C%20we,as%20%E2%88%82L%2F%E2%88%82z%20. 

After backward propagating through all layers we also need to update the values 
F= F - α * ∂L/(∂F_i)  and X = X - α * ∂L/(∂X_i)  and B = B - α * ∂L/(∂B_i)   
Which also requires subtraction operations and multiplication operations (learning rate). Which is equal to the number of filters, input, and bias values for both subtraction and multiplication operations.

as for the fully connected layer at the end of a cnn:
Forward Propagation is basically w * X + b which translates into 
Multiplication operations: #neurons from previous layer * #neurons from current layer. 
(for all layers combined it is simply the number of weights within network)
Addition: #multiplications

As for the backward Propagation:
#multiplications = #weights * 2 for weights
#multiplications = #biases  for bias

What remains is once again the number of subtractions and multiplications resulting from updating the weights and biases:
B = B - α ∂L/(∂B_i)   ,W = W - α ∂L/(∂W_i)   

