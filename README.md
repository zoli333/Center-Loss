# Center-Loss
This is an implementation of the Center Loss article (2016) trained on MNist dataset.

Paper:
A Discriminative Feature Learning Approach
for Deep Face Recognition

Link to the paper:
https://ydwen.github.io/papers/WenECCV16.pdf

Tensorflow Version: 1.5

Mnist train-dataset: 55000 training examples
Mnist test-dataset: 10000 training examples
Used from tensorflow examples.

Result after 13000 iteration (which is roughly 23 epoch):

![13000.png](https://user-images.githubusercontent.com/13023894/36691720-5b842862-1b36-11e8-9d1e-84e212f8ed9f.png)

The above snapshot was taken from a random 1000 training examples. The database was trained with AdamOptimizer.

Crucial steps were for the successful training:
 - remove the bias term from the last layer (before cross entropy)
 - training with 0.0005 learning rate instead of 0.001
 - rescale the training examples from \[0,255\] to \[-1,1\] range
 
 
I was running this code on CPU. This is why just 1000 images were tested. I was tried this out with 10000 examples, but my memory was not enough. The training on intel-i5 processor single core was took about 1 day to train.

The code contains the same nn.py file as in the weight-normalization code. However I am not using any of Weight-Normalization, Batch-normalization initialization for training this code. They are set to false. The only function parameter I use for creating the model template was the use_bias and the use_xavier_initialization parameters. At the last layer use_bias was set to false, and when training the latter parameter was set to true (Xavier initialization and Adam optimizer can work together quite well).

More snapshot from about the training steps can be seen in training_snapshots folder. 
