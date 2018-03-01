# Center-Loss
This is an implementation of the Center Loss article (2016) trained on MNist dataset.

Paper:
A Discriminative Feature Learning Approach
for Deep Face Recognition

Link to the paper:
https://ydwen.github.io/papers/WenECCV16.pdf

Tensorflow Version: 1.5

Mnist train-dataset: 55000 training examples </br>
Mnist test-dataset: 10000 training examples

Used from tensorflow examples.

Result after 13000 iteration (which is roughly 23 epoch):

![13000.png](https://user-images.githubusercontent.com/13023894/36691720-5b842862-1b36-11e8-9d1e-84e212f8ed9f.png)

The above snapshot was made from a random 1000 training example at step 14200. For more snapshots about the training steps can be seen in training_snapshots folder.

The database was trained with AdamOptimizer.

Crucial steps were for the successful training:
 - remove the bias term from the last layer (before cross entropy)
 - training with 0.0005 learning rate instead of 0.001
 - rescale the training examples from \[0,255\] to \[-1,1\] range


I was running this code on CPU. This is why just 1000 images were tested. I was tried this out with 10000 examples for plotting, but my memory was not enough. The training happened on an old intel-i5 processor single core. Training tooks 1 day with CPU.

The code contains the same nn.py file as in the weight-normalization code. However I am not using any of Weight-Normalization, Batch-normalization initialization for training this code. They are set to false. The only function parameter I use for creating the model template was the use_bias and the use_xavier_initialization parameters. At the last layer use_bias was set to false, and when training the latter parameter was set to true (see model.py).


Implementations and links that I found useful:
  - https://github.com/EncodeTS/TensorFlow_Center_Loss
  - https://github.com/pangyupo/mxnet_center_loss
  - https://github.com/ydwen/caffe-face (Authors implementation in C++) 

Notes: The code is not using weight normalization nor mean only batch normalization nor initialization. I should (have been) try this out, whether it can accelerate the training. I do think initialization here would be very useful (avoiding dead clusters). So every normalization technique could be very helpful to accelerate the training.

About the training loss:

    200:  2.3
    5000: 1.2
    10000: 0.7
    14200: 0.4

I assume at about 5000-6000 step the learning rate should have been divided to half, because it was platoed.





