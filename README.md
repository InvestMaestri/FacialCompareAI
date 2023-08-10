# FacialCompareAI
A test implementation of a Siamese Network for the comparison of two faces extracted from an image (or live feed/database). The advantage of this implementation being the outputted similarity score which is able to compare never before seen faces without having to re-train the network.

How it works:
The code implements a Siamese neural network for comparing facial images. It prepares image pairs, constructs a Siamese network architecture with shared convolutional layers, defines a custom contrastive loss function to train the network, and optimizes the model's weights using stochastic gradient descent. The network learns to differentiate between similar and dissimilar face pairs, and its training progress is visualized using loss and accuracy plots at the end of the training session.
