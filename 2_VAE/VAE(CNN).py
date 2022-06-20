previous_model = False

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os.path as path
import os
os.environ['OMP_NUM_THREADS'] = '2'

import lib.dist as dist
import lib.utils as utils


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    'font.family' : 'STIXGeneral',
    'mathtext.fontset' : 'stix',
    'xtick.labelsize' : 13,
    'xtick.top' : True,
    'xtick.direction' : 'in',
    'ytick.direction' : 'in',
    'ytick.labelsize' : 13,
    'ytick.right' : True,
    'axes.labelsize' : 16,
    'legend.frameon': False,
    'legend.fontsize': 13,
    'legend.handlelength' : 1.5,
    'savefig.dpi' : 600,
    'savefig.bbox' : 'tight'
})
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
torch.__version__

b = 60

txt = np.loadtxt(f'./Hirsch-Fye-QMC/Bethe_14_beta{b:d}/field-2.10.dat')
s_num, s_len = txt.shape
print("txt.shape =", s_num, s_len)
s_train = 750
s_test = 1000
if (s_train+s_test > s_num): print("WARNING : not enough datapoints")


U = np.array([0.01*i for i in range(100, 401)])
U_up, U_dn = [], []
for i, u in enumerate(U):
    if path.isfile(f'./Hirsch-Fye-QMC/Bethe_14_beta{b:d}/field-{u:.2f}.dat')==True: U_up.append(u)
    if path.isfile(f'./Hirsch-Fye-QMC/Bethe_41_beta{b:d}/field-{u:.2f}.dat')==True: U_dn.append(u)
U_up = np.array(U_up)
U_dn = np.array(U_dn)
#spin2d_up_train = np.zeros((len(U_up)*s_train, 1, 17, 17))
#spin2d_dn_train = np.zeros((len(U_dn)*s_train, 1, 17, 17))
spin2d_up_test = np.zeros((len(U_up)*s_test, 1, 17, 17))
spin2d_dn_test = np.zeros((len(U_dn)*s_test, 1, 17, 17))
spin2d_up = np.zeros((len(U_up)*s_num, 1, 17, 17))
spin2d_dn = np.zeros((len(U_dn)*s_num, 1, 17, 17))
print("U_up, U_dn len =", len(U_up), len(U_dn))
#print("spin2d_up_train size =", spin2d_up_train.shape)
#print("spin2d_dn_train size =", spin2d_dn_train.shape)

txt = np.zeros((s_num, s_len+1))
for i, u in enumerate(U_up):
    txt[:,:-1] = np.loadtxt(f'./Hirsch-Fye-QMC/Bethe_14_beta{b:d}/field-{u:.2f}.dat')
    txt = 0.5*txt + 0.5
    #spin2d_up_train[i*s_train:(i+1)*s_train] = txt[:s_train,:].reshape(s_train, 1, 17, 17)
    spin2d_up_test[i*s_test:(i+1)*s_test] = txt[-s_test:,:].reshape(s_test, 1, 17, 17)
    spin2d_up[i*s_num:(i+1)*s_num] = txt.reshape(s_num, 1, 17, 17)
for i, u in enumerate(U_dn):
    txt[:,:-1] = np.loadtxt(f'./Hirsch-Fye-QMC/Bethe_41_beta{b:d}/field-{u:.2f}.dat')
    txt = 0.5*txt + 0.5
    #spin2d_dn_train[i*s_train:(i+1)*s_train] = txt[:s_train,:].reshape(s_train, 1, 17, 17)
    spin2d_dn_test[i*s_test:(i+1)*s_test] = txt[-s_test:,:].reshape(s_test, 1, 17, 17)
    spin2d_dn[i*s_num:(i+1)*s_num] = txt.reshape(s_num, 1, 17, 17)

latent_dim = 8
beta = 6
prior_dist = dist.Normal()
q_dist = dist.Normal()

#spin2d = np.concatenate([spin2d_up_train, spin2d_dn_train])

spin2d_up = torch.Tensor(spin2d_up)
spin2d_dn = torch.Tensor(spin2d_dn)
spin2d_up_test = torch.Tensor(spin2d_up_test)
spin2d_dn_test = torch.Tensor(spin2d_dn_test)

spin2d = np.concatenate([spin2d_up, spin2d_dn])
spin2d = torch.Tensor(spin2d)

spin2dtest = np.concatenate([spin2d_up_test, spin2d_dn_test])
spin2dtest = torch.Tensor(spin2dtest)

print("spin2d size =", spin2d.size())


class CEncoder(nn.Module):
    def __init__(self, output_dim):
        super(CEncoder, self).__init__()
        self.output_dim = output_dim

        #self.conv1 = nn.Conv2d(1, 16, 4, 2, 1)  # 16
        #self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(1, 16, 5, 2, 1)  # 8
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 4, 2, 1)  # 4
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 256, 4)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv_z = nn.Conv2d(256, output_dim, 1)

        # setup the non-linearity
        self.act = nn.SELU()

    def forward(self, x):
        h = x.view(-1, 1, 17, 17)
        #h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return z

class CDecoder(nn.Module):
    def __init__(self, input_dim):
        super(CDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 256, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.ConvTranspose2d(256, 32, 4, 1, 0)  # 4
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.ConvTranspose2d(32, 16, 4, 2, 1)  # 8
        self.bn3 = nn.BatchNorm2d(16)
        #self.conv4 = nn.ConvTranspose2d(16, 16, 4, 2, 1)  # 16
        #self.bn4 = nn.BatchNorm2d(16)
        self.conv_final = nn.ConvTranspose2d(16, 1, 5, 2, 1) #17

        # setup the non-linearity
        self.act = nn.SELU()

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        #h = self.act(self.bn4(self.conv4(h)))
        mu_img = self.conv_final(h)
        return mu_img

def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)

def display_samples(model, x):
    fig, ax = plt.subplots(1,2, figsize = (15, 3))
    th = 1
    sample_mu = model.model_sample(batch_size=100).sigmoid()
    xs, x_params, zs, z_params = model.reconstruct_img(x)
    ax[0].imshow(x[th,0], aspect='auto')
    ax[1].imshow(xs[th,0], aspect='auto')
    #ax[2].imshow(x_params[th,0].detach().numpy())
    #ax[3].imshow(z_params[:,:,0].detach().numpy())
    #ax[4].imshow(z_params[:,:,1].detach().numpy())
    plt.savefig('fig.pdf')


### beta-TCVAE
class TCVAE(nn.Module):
    def __init__(self, z_dim, beta, prior_dist=dist.Normal(), q_dist=dist.Normal(),
                 include_mutinfo=True, tcvae=True, mss=True):
        super(TCVAE, self).__init__()

        self.z_dim = z_dim
        self.include_mutinfo = include_mutinfo
        self.tcvae = tcvae
        self.lamb = 0
        self.beta = beta
        self.mss = mss
        self.x_dist = dist.Bernoulli()

        self.prior_dist = prior_dist
        self.q_dist = q_dist
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

        self.encoder = CEncoder(z_dim * self.q_dist.nparams)
        self.decoder = CDecoder(z_dim)

    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    def model_sample(self, batch_size=1):
        prior_params = self._get_prior_params(batch_size)
        zs = self.prior_dist.sample(params=prior_params)
        x_params = self.decoder.forward(zs)
        return x_params

    def encode(self, x):
        x = x.view(x.size(0), 1, 17, 17)
        z_params = self.encoder.forward(x).view(x.size(0), self.z_dim, self.q_dist.nparams)
        zs = self.q_dist.sample(params=z_params)
        return zs, z_params

    def decode(self, z):
        x_params = self.decoder.forward(z).view(z.size(0), 1, 17, 17)
        xs = self.x_dist.sample(params=x_params)
        return xs, x_params

    def reconstruct_img(self, x):
        zs, z_params = self.encode(x)
        xs, x_params = self.decode(zs)
        return xs, x_params, zs, z_params

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()

    def elbo(self, x, dataset_size):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 17, 17)
        prior_params = self._get_prior_params(batch_size)
        x_recon, x_params, zs, z_params = self.reconstruct_img(x)
        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)

        elbo = logpx + logpz - logqz_condx # log p(x|z) + log p(z) - log q(z|x)
        # log q(z) ~ log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = self.q_dist.log_density(zs.view(batch_size, 1, self.z_dim),
                                         z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams))

        if not self.mss: # minibatch weighted sampling
            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
        else: # minibatch stratified sampling
            logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, dataset_size).type_as(_logqz.data))
            logqz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1, keepdim=False).sum(1)

        if self.include_mutinfo:
            elbo2 = logpx - logqz_condx + logqz - self.beta * (logqz - logqz_prodmarginals) - \
                                (1 - self.lamb) * (logqz_prodmarginals - logpz)
        else:
            elbo2 = logpx - self.beta * (logqz - logqz_prodmarginals) - \
                                (1 - self.lamb) * (logqz_prodmarginals - logpz)

        return elbo2, elbo.detach()


if previous_model == True:
    model = TCVAE(z_dim=latent_dim, beta=beta, prior_dist=prior_dist, q_dist=q_dist)
    model.load_state_dict(torch.load('./frozen/VAE_CNN_2.pth'))

model = TCVAE(z_dim = latent_dim, beta = beta, prior_dist = prior_dist, q_dist = q_dist)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train_loader = DataLoader(dataset = spin2d, batch_size = 1000, shuffle = True)

'''
train_elbo = []
dataset_size = len(train_loader.dataset)
num_iter = len(train_loader)*50
elbo_run_mean = utils.RunningAverageMeter()

print("dataset_size, num_iter =", dataset_size, num_iter)

for n in range(num_iter):
    for i,x in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        obj, elbo = model.elbo(x, dataset_size)
        if utils.isnan(obj).any(): break
        obj.mean().mul(-1).backward()
        elbo_run_mean.update(elbo.mean().item())
        optimizer.step()

    if n%20==0:
        train_elbo.append(elbo_run_mean.avg)
        print('[n = %03d] ELBO = %.6f (%.6f)'%(n, elbo_run_mean.val, elbo_run_mean.avg))
        #display_samples(model, x)
        torch.save(model.state_dict(), 'frozen/VAE_CNN_2.pth')
        np.savetxt(f'frozen/elbo_CNN_2', np.array(train_elbo))'''
        

xs, x_params, zs, z_params = model.reconstruct_img(spin2dtest)
xs_up, x_params_up, zs_up, z_params_up = model.reconstruct_img(spin2d_up_test)
xs_dn, x_params_dn, zs_dn, z_params_dn = model.reconstruct_img(spin2d_dn_test)

xs_tot_up, x_params_tot_up, zs_tot_up, z_params_tot_up = model.reconstruct_img(spin2d_up)
xs_tot_dn, x_params_tot_dn, zs_tot_dn, z_params_tot_dn = model.reconstruct_img(spin2d_dn)

U_len = len(U_up)+len(U_dn)
#torch.save(z_params, 'CNN_zp.dat')
#torch.save(z_params_up, 'CNN_zp_up.dat')
#torch.save(z_params_dn, 'CNN_zp_dn.dat')

fig, ax = plt.subplots(1,2, figsize = (12, 5), sharey=True)
im0 = ax[0].imshow(spin2dtest.reshape(U_len*s_test, 289), aspect='auto')
ca0 = inset_axes(ax[0], width="5%", height="100%", loc='center left',\
                bbox_to_anchor=(1.05, 0., 0.8, 1), bbox_transform=ax[0].transAxes, borderpad=0)
cd = plt.colorbar(im0, cax=ca0)
im1 = ax[1].imshow(xs.reshape(U_len*s_test, 289), aspect='auto')
ca1 = inset_axes(ax[1], width="5%", height="100%", loc='center left',\
                bbox_to_anchor=(1.05, 0., 0.8, 1), bbox_transform=ax[1].transAxes, borderpad=0)
cd = plt.colorbar(im1, cax=ca1)
plt.savefig('CNN_result.png')

'''
test_mu_up = z_params_up[:, :, 0].detach().numpy()
test_logsig_up = z_params_up[:, :, 1].detach().numpy()
test_var_up = np.exp(test_logsig_up * 2)
test_mu_dn = z_params_dn[:, :, 0].detach().numpy()
test_logsig_dn = z_params_dn[:, :, 1].detach().numpy()
test_var_dn = np.exp(test_logsig_dn * 2)

test_ens_mu_up = np.zeros((len(U_up), latent_dim))
test_ens_logsig_up = np.zeros((len(U_up), latent_dim))
test_ens_mu_dn = np.zeros((len(U_dn), latent_dim))
test_ens_logsig_dn = np.zeros((len(U_dn), latent_dim))
for i in range(len(U_up)):
    test_ens_mu_up[i] = np.average(z_params_up[i*s_test:(i+1)*s_test, :, 0].detach().numpy(), axis=0)
    test_ens_logsig_up[i] = np.average(z_params_up[i*s_test:(i+1)*s_test, :, 1].detach().numpy(), axis=0)
for i in range(len(U_dn)):
    test_ens_mu_dn[i] = np.average(z_params_dn[i*s_test:(i+1)*s_test, :, 0].detach().numpy(), axis=0)
    test_ens_logsig_dn[i] = np.average(z_params_dn[i*s_test:(i+1)*s_test, :, 1].detach().numpy(), axis=0)
test_ens_var_up = np.exp(test_ens_logsig_up * 2)
test_ens_var_dn = np.exp(test_ens_logsig_dn * 2)

test_ens_mu = np.concatenate([test_ens_mu_up, test_ens_mu_dn])
test_ens_var = np.concatenate([test_ens_var_up, test_ens_var_dn])


fig, ax = plt.subplots(2,4, figsize = (12, 12))
ax[0,0].imshow(test_mu_up, aspect = 'auto')
ax[0,1].imshow(test_var_up, aspect = 'auto')
ax[0,2].imshow(test_mu_dn, aspect = 'auto')
ax[0,3].imshow(test_var_dn, aspect = 'auto')

ax[1,0].imshow(test_ens_mu_up, aspect = 'auto')
ax[1,1].imshow(test_ens_var_up, aspect = 'auto')
ax[1,2].imshow(test_ens_mu_dn, aspect = 'auto')
ax[1,3].imshow(test_ens_var_dn, aspect = 'auto')
plt.savefig('CNN_zparam.png')

n = 3
#if model == model_LR2: n = 2

U_c1, U_c2 = 2.44, 2.59

pca1 = PCA(n_components=n)
pca1.fit(test_ens_mu)
mu_pca_up = pca1.transform(test_ens_mu_up)
mu_pca_dn = pca1.transform(test_ens_mu_dn)

pca2 = PCA(n_components=n)
pca2.fit(test_ens_var)
var_pca_up = pca2.transform(test_ens_var_up)
var_pca_dn = pca2.transform(test_ens_var_dn)

UU = np.concatenate([U_up, U_dn])

fig, ax = plt.subplots(2,2, figsize = (9, 9), sharex=True)

ax[0,0].axvline(U_c1, c='#99cc99', lw='6')
ax[0,0].axvline(U_c2, c='#99cc99', lw='6')
ax[0,0].plot(U_up, mu_pca_up[:,0], 'b.-')
ax[0,0].plot(U_dn, mu_pca_dn[:,0], 'k.-')

ax[0,1].axvline(U_c1, c='#99cc99', lw='6')
ax[0,1].axvline(U_c2, c='#99cc99', lw='6')
ax[0,1].plot(U_up, var_pca_up[:,0], 'b.-')
ax[0,1].plot(U_dn, var_pca_dn[:,0], 'k.-')

ax[1,0].axvline(U_c1, c='#99cc99', lw='6')
ax[1,0].axvline(U_c2, c='#99cc99', lw='6')
ax[1,0].plot(U_up, mu_pca_up[:,1], 'b.-')
ax[1,0].plot(U_dn, mu_pca_dn[:,1], 'k.-')

ax[1,1].axvline(U_c1, c='#99cc99', lw='6')
ax[1,1].axvline(U_c2, c='#99cc99', lw='6')
ax[1,1].plot(U_up, var_pca_up[:,1], 'b.-')
ax[1,1].plot(U_dn, var_pca_dn[:,1], 'k.-')

ax[0,0].set_xlim(2.2, 2.8)

plt.savefig(f'CNN_pca.png')'''
