# FacialCompareAI
A test implementation of a Siamese Network for the comparison of two faces extracted from an image (or live feed/database). The advantage of this implementation being the outputted similarity score which is able to compare never before seen faces without having to re-train the network.

How it works:

The code implements a Siamese neural network for comparing facial images. It prepares image pairs, constructs a Siamese network architecture with shared convolutional layers, defines a custom contrastive loss function to train the network, and optimizes the model's weights using stochastic gradient descent. The network learns to differentiate between similar and dissimilar face pairs, and its training progress is visualized using loss and accuracy plots at the end of the training session.

The objective of this project was to implement the use of two identical networks that are run side by side, each on an image. The result is a similarity score (contrastive loss).

Overall, the network was trained on a dataset of celebrity photos that were harvested from the internet for specifically for this project, under the following distribution:
- 20% White
- 17% Black
- 35% Asian
- 28% Middle Eastern

Being them: 50% male and 50% female (close to the global average of ~49% male and ~51% female).

Then combinations of all images were created marking those with the same name as being 1 or equal and those having different names as being 0 or different. 80% - 20% of the set was used for training and testing, and then the network was run on a separate database where no statistical difference in the accuracy or loss values were found.

So, one use case example is comparing the input from a security camera feed on an elevator where, once a floor is requested, the AI will then get the live feed from the security camera and run a comparison through the Neural Network between the detected face in the elevator and the database of those allowed on the requested floor. If there is a match the elevator proceeds to the requested floor, if not it will buzz back a negative sound and a red light.

Another example of a use case is searching for a specific individual in a live feed or video.

The idea is that current open-source solutions are trained on recognizing specific people, meaning that if any person added or removed, the entire network needs to be trained again. With this implementation the end user can match any two faces that have never been seen by the Neural Network and still be able to recognize faces and give a positive or negative match result. 
This results in higher up-time and the only update needed would be the implementation of new networks trained on more tailored sets or new topology that is more effective or efficient or trained in larger datasets.
