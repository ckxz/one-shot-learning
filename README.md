# one-shot-learning

Training and testing was performed on CelebA Dataset (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)


Three different models were implemented: 

a) k-NN algorithm

b) Siamese Neural Network (G. Koch, R. Zemel and R. Salakhutdinov. “Siamese neural networks for one-shot image recognition”. In: ICML deep learning workshop. Vol. 2. 2015.). It can be run using different configurations:
	First parameter specifies the mode under which we want to run it: train/test.
	Second parameters refers to the loss function: binary_crossentropy
	Third parameter allows choosing if any pre-trained networks are going to be used for the branches:
		- basic - no pre-trained network
		- resnet
		- facenet
	Fourth parameter specifies if we're working with RGB or black and white images: ""/bw.
	
c) Plastic CNN (T. Miconi, J. Clune and K. O. Stanley. “Differentiable plasticity: training plastic neural networks with backpropagation”. In: arXiv preprint arXiv:1804.02464 (2018)).
