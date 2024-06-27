import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F # often for ReLU
import torch.optim as optim
import torchvision # for datasets, transforms

"""
This is an example convolutional network design from the official pytorch examples.
https://github.com/pytorch/examples/blob/main/mnist/main.py

I've commented various parts to explain what I've learned about different 
mechanisms and patterns.

It performs way better than my dumb linear example, unsurprisingly.
"""


# inherit to set up networks (or datasets, dataloaders, layers, anything =~ torch.nn.module)
class Net(nn.Module):
	def __init__(self) -> None:
		super(Net, self).__init__()
		"""
		https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
		https://en.wikipedia.org/wiki/Convolution
		https://www.songho.ca/dsp/convolution/convolution2d_example.html
		Convolutions have to do with comparing input and output together in their computation
		Really good for image recognition due to edge detection and other factors
		"""
		self.conv1 = nn.Conv2d(1, 32, 3, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 1)
		"""
		https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
		Dropout automates random zeroing during training 
		maybe to fight overfitting, and to avoid "co-adaptation" (neurons depend on each other)
		"""
		self.dropout1 = nn.Dropout(0.25)
		self.dropout2 = nn.Dropout(0.5)
		"""
		https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
		Tool for appling linear transformation - nn.Linear(in_features, out_features)
		Essentially equivalent to one basic layer of neurons in a feed forward.
		Param meanings: size of input/output samples
		"""
		self.fc1 = nn.Linear(9216, 128)
		self.fc2 = nn.Linear(128, 10)
	
	def forward(self, x):
		"""
		You are supposed to override forward(self, x) whenever you inherit nn.Module
		It wraps python __call__, which runs when you "call" the class like a func.
		"Calling" the class like this is the standard way to run an inference, e.g. model(data).
		"""

		x = self.conv1(x)
		"""
		relu is the nonlinear activation function
		You need this nonlinearity so that this NN is not just equivalent 
		to one with 0 hidden layers, which has limited use
		https://stackoverflow.com/questions/9782071/why-must-a-nonlinear-activation-function-be-used-in-a-backpropagation-neural-net
		"""
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		"""
		https://deepai.org/machine-learning-glossary-and-terms/max-pooling#:~:text=Advantages%20of%20Max%20Pooling&text=Dimensionality%20Reduction%3A%20By%20downsampling%20the,noise%20in%20the%20input%20data.
		Pooling takes chunk of the tensor, gets a max of that "pool",
		and uses that max to represent the pool.
		Supposed to reduce noise and overfitting, specifically used for CNNs
		"""
		x = F.max_pool2d(x, 2)
		x = self.dropout1(x)
		"""
		flatten turns tensor to 1 dimension (array) like it sounds
		But param 1 means only flatten those with "start dimension" 1 (not the entire tensor at once)
		"""
		x = torch.flatten(x, 1)
		x = self.fc1(x) 
		x = F.relu(x) # After every linear transform, you call a nonlinear activation, unless it's the last layer
		x = self.dropout2(x)
		x = self.fc2(x)
		output = F.log_softmax(x, dim=1) # need log_softmax to use nll loss function later on
		return output

def train(model, device, train_loader, optimizer, epoch):
	"""
	Train a single epoch; this one does mini batches as is standard nowadays
	"""

	"""
	model.eval() is the other option from model.train().
	Each prepares applicable modules for each mode, since some -- like Dropout --
	behave differently in the training vs evaluation usage.
	Some layers are not affected at all.
	"""
	model.train() # "set to train mode" - so inner layers like Dropout know what to do
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device) # loads tensors to "device", gpu in my case
		optimizer.zero_grad() # clear existing derivative info before performing a new backprop later
		output = model(data) # calls "forward" on network
		loss = F.nll_loss(output, target) # negative log likelihood loss -- common loss function I guess
		loss.backward() # execute backprop, creating a graph of gradients to use
		optimizer.step() # use optimizer method to update weights/biases with calculated gradients
		if batch_idx % 10 == 0: # periodic logging to see how it's going
			print('Train Epoch: {} [{}/{} ({:.0f}%)]/tLoss: {:.6f}'.format(
				epoch,
				batch_idx * len(data),
				len(train_loader.dataset),
				100. * batch_idx / len(train_loader),
				loss.item()
			))

def test(model, device, test_loader):
	"""
	Run 10000-test example with no grad info since we're not training.
	Most of the other setup is the same, since you are using the same network and devices.
	"""

	model.eval() # "set to eval mode"
	test_loss = 0
	correct = 0
	with torch.no_grad(): # Save memory and other optimizations by disabling any gradient graph stuff, since we're not training
		for data, target in test_loader: # no enumerate for batch_idx because we don't need to test in batches
			data, target = data.to(device), target.to(device)
			output = model(data) # run forward with existing weights
			test_loss += F.nll_loss(output, target, reduction='sum').item()
			pred = output.argmax(dim=1, keepdim=True) # convert output into the actual guesses the max values represent (tensor of the actual guesses)
			correct += pred.eq(target.view_as(pred)).sum().item() # directly compare individual guesses to real answers for a count of successes
	
	test_loss /= len(test_loader.dataset)

	# Very similar log to train()
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss,
		correct,
		len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)
	))


def main():
	"""
	main() is running all epochs for training and testing.
	Could rewrite this material to provide more convenient hyperparam choices 
	or different optimizer functions, devices, etc. and also inference the model 
	with a different interface.

	In this case I've dumb'd things down to fit my specific case (having pytorch cuda, etc.).
	I could easily implement hyperparameters and interface details any number of ways later on.
	"""

	# Can take this as an arg, but don't care right now
	torch.manual_seed(1)

	# I am using Nvidia, end of story
	device = torch.device("cuda")

	# These are the defaults in the reference, can take as args later
	cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
	train_kwargs = {'batch_size': 64}
	train_kwargs.update(cuda_kwargs)
	test_kwargs = {'batch_size': 1000}
	test_kwargs.update(cuda_kwargs)

	transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.1307,), (0.3081,))
	])

	# Downloading MNIST handwriting set if needed
	data_path = "./"
	train_set = torchvision.datasets.MNIST(data_path, train=True, download=True, transform=transform)
	test_set = torchvision.datasets.MNIST(data_path, train=False, transform=transform)
	# Pack data into data loader for handling batches and things like that
	train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
	test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)

	# The model referenced in train() and test() is this instantiated Net, 
	# since they are not in scope of the class
	model = Net().to(device)

	# lr = learning rate
	optimizer = optim.Adadelta(model.parameters(), lr=1.0) # Adadelta one of many optimizer choices


	"""
	A scheduler was a new concept to me looking at this example network.
	This scheduler automatically adjusts the learning rate hyperparam 
	as the network learns; makes sense that learning rate may have a 
	different optimal value as the network works toward potential 
	overfit and looks for more nuanced pathways.
	"""
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

	epochs = 14
	for epoch in range(1, epochs + 1):
		train(model, device, train_loader, optimizer, epoch)
		test(model, device, test_loader)
		scheduler.step()
	
	# TODO: I have no idea how big this will be
	# torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
	main()