# one-shot-learning

Training and testing was performed on CelebA Dataset (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)


Three different models were implemented: 

a) k-NN algorithm

b) Siamese Neural Network (Koch et al, 2015, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf). It can be run using different configurations:
	First parameter specifies the mode under which we want to run it: train/test.
	Second parameters refers to the loss function: binary_crossentropy
	Third parameter allows choosing if any pre-trained networks are going to be used for the branches:
		- basic - no pre-trained network
		- resnet
		- facenet
	Fourth parameter specifies if we're working with RGB or black and white images: ""/bw.
	
c) Plastic CNN (Miconi et al, 2018, https://arxiv.org/abs/1804.02464).
