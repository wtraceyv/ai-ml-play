## MNIST Handwriting

Neural networks to classify the handwritten numbers from the MNIST data set are sort of a hello world of machine learning.

`cnn_commented.py` is a convolutional neural network that solves the problem very well, provided as a example by pytorch on github (link in file). I've commented everything not obvious to me to get familiar with either the ML concepts involved or Pytorch-specific conventions. I did strip it of arg parsing so I could just hardcode my info and not fight the cmd just to test.

`trivial_linear.py` is essentially the same setup but the network is a basic feed forward network with one hidden layer, trying to emulate the first network written manually in [http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits](http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits). With the more current loss and optimizing techniques from the CNN, it still performs surprisingly well, maybe even better than the manual model from the book.

Running either model will include automatically downloading the MNIST data if it's not already in the code folder.