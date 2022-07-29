import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision

# sklearn imports
from sklearn.manifold import TSNE
from utils import *

# reparametrization backprop
def reparameterize(mu, logvar, device=torch.device("cpu")):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variance of x
    :param device: device to perform calculations on
    :return z: the sampled latent variable
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std


# encoder - Q(z|X)
class VaeEncoder(torch.nn.Module):
    """
       This class builds the encoder for the VAE
       :param x_dim: input dimensions
       :param hidden_size: hidden layer size
       :param z_dim: latent dimensions
       :param device: cpu or gpu
       """

    def __init__(self, x_dim=28 * 28, hidden_size=256, z_dim=10, device=torch.device("cpu")):
        super(VaeEncoder, self).__init__()
        self.x_dim = x_dim
        self.hidden_size = hidden_size
        self.z_dim = z_dim
        self.device = device

        self.features = nn.Sequential(nn.Linear(x_dim, self.hidden_size),
                                      nn.ReLU())

        self.fc1 = nn.Linear(self.hidden_size, self.z_dim, bias=True)  # fully-connected to output mu
        self.fc2 = nn.Linear(self.hidden_size, self.z_dim, bias=True)  # fully-connected to output logvar

    def bottleneck(self, h):
        """
        This function takes features from the encoder and outputs mu, log-var and a latent space vector z
        :param h: features from the encoder
        :return: z, mu, log-variance
        """
        mu, logvar = self.fc1(h), self.fc2(h)
        # use the reparametrization trick as torch.normal(mu, logvar.exp()) is not differentiable
        z = reparameterize(mu, logvar, device=self.device)
        return z, mu, logvar

    def forward(self, x):
        """
        This is the function called when doing the forward pass:
        z, mu, logvar = VaeEncoder(X)
        """
        h = self.features(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar


# Decoder
class VaeDecoder(torch.nn.Module):
    """
       This class builds the decoder for the VAE
       :param x_dim: input dimensions
       :param hidden_size: hidden layer size
       :param z_dim: latent dimensions
       """

    def __init__(self, x_dim=28 * 28, hidden_size=256, z_dim=10):
        super(VaeDecoder, self).__init__()
        self.x_dim = x_dim
        self.hidden_size = hidden_size
        self.z_dim = z_dim

        self.decoder = nn.Sequential(nn.Linear(self.z_dim, self.hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_size, self.x_dim),
                                     nn.Sigmoid())
        # why we use sigmoid? becaue the pixel values of images are in [0,1] and sigmoid(x) does just that!
        # if you don't work with images, you don't have to use that.

    def forward(self, x):
        """
        This is the function called when doing the forward pass:
        x_reconstruction = VaeDecoder(z)
        """
        x = self.decoder(x)
        return x


# VAE Assemble - end-to-end encoder-decoder model
class Vae(torch.nn.Module):
    def __init__(self, x_dim=28*28, z_dim=10, hidden_size=256, device=torch.device("cpu")):
        super(Vae, self).__init__()
        self.device = device
        self.z_dim = z_dim

        self.encoder = VaeEncoder(x_dim, hidden_size, z_dim=z_dim, device=device)
        self.decoder = VaeDecoder(x_dim, hidden_size, z_dim=z_dim)

    def encode(self, x):
        z, mu, logvar = self.encoder(x)
        return z, mu, logvar

    def decode(self, z):
        x = self.decoder(z)
        return x

    def sample(self, num_samples=1):
        """
        This functions generates new data by sampling random variables and decoding them.
        Vae.sample() actually generates new data!
        Sample z ~ N(0,1)
        """
        z = torch.randn(num_samples, self.z_dim).to(self.device)
        return self.decode(z)

    def forward(self, x):
        """
        This is the function called when doing the forward pass:
        return x_recon, mu, logvar, z = Vae(X)
        """
        z, mu, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z


# Loss function
def loss_function(recon_x, x, mu, logvar, loss_type='bce'):
    """
    This function calculates the loss of the VAE.
    loss = reconstruction_loss - 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param recon_x: the reconstruction from the decoder
    :param x: the original input
    :param mu: the mean given X, from the encoder
    :param logvar: the log-variance given X, from the encoder
    :param loss_type: type of loss function - 'mse', 'l1', 'bce'
    :return: VAE loss
    """
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='sum')
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction='sum')
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction='sum')
    else:
        raise NotImplementedError

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_error + kl) / x.size(0)



# let's load the dataset and see some examples

# in order to create batches of the data, we create a Dataset and a DataLoader, which takes care of:
# 1. pre-processing the images to tensors with values in [0,1]
# 2. shuffling the data, so we add randomness as learned in ML
# 3. if the data size is not divisible by the batch size, we can drop the last batch
# (so the batches are always of the same size)

# define pre-procesing transformation to use
#transform = torchvision.transforms.ToTensor()
#train_data = torchvision.datasets.MNIST('./datasets/', train=True, transform=transform, target_transform=None, download=True)
#test_data = torchvision.datasets.MNIST('./datasets/', train=False, transform=transform, target_transform=None, download=True)


train_path = "/home/maintenance/projects/Unseen Mode 2/Data/Train2"
test_path = "/home/maintenance/projects/Unseen Mode 2/Data/Test2"
x_train, y_train = load_into_x_y(train_path)
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
x_test, y_test = load_into_x_y(test_path)
test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

sample_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

#fig = plt.figure(figsize=(8 ,5))
#samples, labels = next(iter(sample_dataloader))
#for i in range(samples.size(0)):
#    ax = fig.add_subplot(2, 3, i + 1)
#    ax.imshow(samples[i][0, :, :].data.cpu().numpy(), cmap='gray')
#    title = "digit: " + str(labels[i].data.cpu().item())
#    ax.set_title(title)
#    ax.set_axis_off()


# define hyper-parameters
BATCH_SIZE = 32  # usually 32/64/128/256
LEARNING_RATE = 1e-3  # for the gradient optimizer
NUM_EPOCHS = 30  # how many epochs to run?
HIDDEN_SIZE = 256  # size of the hidden layers in the networks
X_DIM = 3  # size of the input dimension
Z_DIM = 5  # size of the latent dimension


# training

# check if there is gpu avilable, if there is, use it
if torch.cuda.is_available():
    torch.cuda.current_device()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("running calculations on: ", device)

# load the data
#dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# create our model and send it to the device (cpu/gpu)

vae = Vae(x_dim=X_DIM, z_dim=Z_DIM, hidden_size=HIDDEN_SIZE, device=device).to(device)

# optimizer
vae_optim = torch.optim.Adam(params=vae.parameters(), lr=LEARNING_RATE)

# save the losses from each epoch, we might want to plot it later
train_losses = []

# here we go
for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    batch_losses = []
    for batch_i, batch in enumerate(x_train):
        # forward pass
        x = torch.from_numpy(batch[0]).to(device).view(-1, X_DIM)
        x = x.type(torch.cuda.FloatTensor)
        x_recon, mu, logvar, z = vae(x)
        # calculate the loss
        loss = loss_function(x_recon, x, mu, logvar, loss_type='mse')
        # optimization (same 3 steps everytime)
        vae_optim.zero_grad()
        loss.backward()
        vae_optim.step()
        # save loss
        batch_losses.append(loss.data.cpu().item())
    train_losses.append(np.mean(batch_losses))
    print("epoch: {} training loss: {:.5f} epoch time: {:.3f} sec".format(epoch, train_losses[-1],
                                                                          time.time() - epoch_start_time))


# saving our model (so we don't have to train it again...)
# this is one of the greatest things in pytorch - saving and loading models

# save
fname = "./vae_imu_" + str(NUM_EPOCHS) + "_epochs.pth"
torch.save(vae.state_dict(), fname)
print("saved checkpoint @", fname)

# load
vae = Vae(x_dim=X_DIM, z_dim=Z_DIM, hidden_size=HIDDEN_SIZE, device=device).to(device)
vae.load_state_dict(torch.load(fname))
print("loaded checkpoint from", fname)


# now let's sample from the vae
n_samples = 6
vae_samples = vae.sample(num_samples=n_samples).view(n_samples, 28, 28).data.cpu().numpy()
fig = plt.figure(figsize=(8 ,5))
for i in range(vae_samples.shape[0]):
    ax = fig.add_subplot(2, 3, i + 1)
    ax.imshow(vae_samples[i], cmap='gray')
    ax.set_axis_off()


# Interpolation in the Latent Space
# let's do something fun - interpolation of the latent space
alphas = np.linspace(0.1, 1, 10)
# take 2 samples
#sample_dataloader = DataLoader(test_data, batch_size=2, shuffle=True, drop_last=True)
#it = iter(sample_dataloader)
#samples, labels = next(it)
#while labels[0] == labels[1]:
    # make sure they are different digits
#    samples, labels = next(it)
#x_1, x_2 = samples

# get their latent representation
_,_, _, z_1 = vae(x_test.view(-1, X_DIM).to(device))
_,_, _, z_2 = vae(x_train.view(-1, X_DIM).to(device))

# let's see the result
fig = plt.figure(figsize=(15 ,8))
for i, alpha in enumerate(alphas):
    z_new = alpha * z_1 + (1 - alpha) * z_2
    x_new = vae.decode(z_new)
    ax = fig.add_subplot(1, 10, i + 1)
    ax.imshow(x_new.view(28, 28).cpu().data.numpy(), cmap='gray')
    ax.set_axis_off()


# Latent Space Representation with t-SNE
# take 2000 samples
num_samples = 2000
sample_dataloader = DataLoader(x_train, batch_size=num_samples, shuffle=True, drop_last=True)
samples, labels = next(iter(sample_dataloader))

labels = labels.data.cpu().numpy()
# decode the samples
_,_, _, z = vae(samples.view(num_samples, X_DIM).to(device))

# t-SNE
perplexity = 15.0
t_sne = TSNE(n_components=2, perplexity=perplexity)
z_embedded = t_sne.fit_transform(z.data.cpu().numpy())

# plot
fig = plt.figure(figsize=(10 ,8))
ax = fig.add_subplot(1, 1, 1)
for i in np.unique(labels):
    ax.scatter(z_embedded[labels==i,0], z_embedded[labels==i, 1], label=str(i))
ax.legend()
ax.grid()
ax.set_title("t-SNE of VAE Latent Space on IMU")