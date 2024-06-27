import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F # often for ReLU
import torch.optim as optim
import torchvision # for datasets, transforms
import matplotlib.pyplot as plt

"""
Essentially the same setup as cnn_commented, but I wanted to implement 
a very basic architecture like the first example on http://neuralnetworksanddeeplearning.com/

It is a feed forward with one hidden layer with however many neurons you 
set hidden_neurons to. It can approach 97% accuracy within a few epochs, 
which seems impressive for a design much dumber than a convolutional net.
"""

DATA_PATH = "./"

class Net(nn.Module):
	def __init__(self) -> None:
		super(Net, self).__init__()
		hidden_neurons = 100
		self.lin1 = nn.Linear(28*28, hidden_neurons)
		self.lin2 = nn.Linear(hidden_neurons, 10)
	
	def forward(self, x):
		x = torch.flatten(x, start_dim=1)
		x = F.relu(self.lin1(x))
		x = self.lin2(x)
		x = F.log_softmax(x, dim=1)
		return x

# https://nextjournal.com/gkoehler/pytorch-mnist
def display_test(train_loader):
	examples = enumerate(train_loader)
	batch_idx, (data, targets) = next(examples)
	fig = plt.figure()
	for i in range(10):
		plt.subplot(5,5,i+1)
		plt.tight_layout()
		plt.imshow(data[i][0], cmap='gray', interpolation='none')
		plt.title("exp: {}".format(targets[i]))
		plt.xticks([])
		plt.yticks([])
	plt.show()

def train(model, device, optimizer, epoch, train_loader):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % 100 == 0: # periodic log
			print(output.shape, target.shape)
			print('Train Epoch: {} [{}/{} ({:.0f}%)]    Loss: {:.6f}'.format(
				epoch,
				batch_idx * len(data),
				len(train_loader.dataset),
				100. * batch_idx / len(train_loader),
				loss.item()
			))

def test(model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad(): # make sure no memory wasted being prepared for grads, since not doing that here
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.nll_loss(output, target, reduction='sum').item()
			pred = output.argmax(dim=1, keepdim=True)
			correct += pred.eq(target.view_as(pred)).sum().item()
	test_loss /= len(test_loader.dataset)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss,
		correct,
		len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)
	))

def main():
	torch.manual_seed(1)
	device = torch.device("cuda")
	cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
	train_kwargs = {'batch_size': 100}
	train_kwargs.update(cuda_kwargs)
	test_kwargs = {'batch_size': 1000}
	test_kwargs.update(cuda_kwargs)

	transform = torchvision.transforms.Compose([ # I think this makes MNIST images usable as tensors
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.1307,), (0.3081,)) # these values specific to data set
	])
	train_set = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
	test_set = torchvision.datasets.MNIST(DATA_PATH, train=False, transform=transform)
	# Pack data into data loader for handling batches and things like that
	train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
	test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)

	model = Net().to(device)

	"""
	Played around to make sure other optimizers work at least,
	but this one is still seemingly much better than the typical SGD example.
	"""
	optimizer = optim.Adadelta(model.parameters(), lr=1.0)
	# optimizer = optim.SGD(model.parameters(), lr=1e-3) # classic backprop alg for test

	epochs = 5
	for epoch in range(1, epochs + 1):
		train(model, device, optimizer, epoch, train_loader)
		test(model, device, test_loader)

	# torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
	main()