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

#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
torch.__version__


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-hnode", help = "number of hidden node", type = int, default = None)
parser.add_argument("-model", help = "model name", type = str, default = None)
parser.add_argument("-prefix", help = "prefix", type = str, default = './Hirsch-Fye-QMC')
args = parser.parse_args()




latent_dim = 8
beta = 4
modelname = args.model
h_node = args.hnode
prefix = args.prefix
prefix2 = prefix


b = 60
#prefix = '../1_QMC/train' #'./Hirsch-Fye-QMC'

txt = np.loadtxt(prefix + f'/Bethe_14_beta{b:d}/field-2.10.dat')
s_tot, s_len = txt.shape
print("txt.shape =", s_tot, s_len)
s_train = 15
s_test = 15
s_num = 30
t_len = 100
if (s_num*t_len > s_tot): print("WARNING : not enough datapoints")

U = np.array([0.01*i for i in range(100, 401)])
U_up, U_dn = [], []
for i, u in enumerate(U):
    if path.isfile(prefix + f'/Bethe_14_beta{b:d}/field-{u:.2f}.dat')==True: U_up.append(u)
    if path.isfile(prefix + f'/Bethe_41_beta{b:d}/field-{u:.2f}.dat')==True: U_dn.append(u)
U_up = np.array(U_up)
U_dn = np.array(U_dn)
spin2d_up_train = np.zeros((len(U_up)*s_train, t_len, s_len))
spin2d_dn_train = np.zeros((len(U_dn)*s_train, t_len, s_len))
spin2d_up_test = np.zeros((len(U_up)*s_test, t_len, s_len))
spin2d_dn_test = np.zeros((len(U_dn)*s_test, t_len, s_len))
spin2d_up = np.zeros((len(U_up)*s_num, t_len, s_len))
spin2d_dn = np.zeros((len(U_dn)*s_num, t_len, s_len))
print("U_up, U_dn len =", len(U_up), len(U_dn))

for i, u in enumerate(U_up):
    txt = np.loadtxt(prefix + f'/Bethe_14_beta{b:d}/field-{u:.2f}.dat')
    txt = 0.5*(txt+1)
    spin2d_up_train[i*s_train:(i+1)*s_train] = txt[:s_train*t_len,:].reshape(s_train, t_len, s_len)
    spin2d_up_test[i*s_test:(i+1)*s_test] = txt[-s_test*t_len:,:].reshape(s_test, t_len, s_len)
    spin2d_up[i*s_num:(i+1)*s_num] = txt.reshape(s_num, t_len, s_len)
for i, u in enumerate(U_dn):
    txt = np.loadtxt(prefix + f'/Bethe_41_beta{b:d}/field-{u:.2f}.dat')
    txt = 0.5*(txt+1)
    spin2d_dn_train[i*s_train:(i+1)*s_train] = txt[:s_train*t_len,:].reshape(s_train, t_len, s_len)
    spin2d_dn_test[i*s_test:(i+1)*s_test] = txt[-s_test*t_len:,:].reshape(s_test, t_len, s_len)
    spin2d_dn[i*s_num:(i+1)*s_num] = txt.reshape(s_num, t_len, s_len)

prior_dist = dist.Normal()
q_dist = dist.Normal()

spin2d = np.concatenate([spin2d_up_train, spin2d_dn_train])
spin2dtest = np.concatenate([spin2d_up_test, spin2d_dn_test])
spin2d = torch.Tensor(spin2d)
spin2dtest = torch.Tensor(spin2dtest)
spin2d_up_test = torch.Tensor(spin2d_up_test)
spin2d_dn_test = torch.Tensor(spin2d_dn_test)
spin2d_up = torch.Tensor(spin2d_up)
spin2d_dn = torch.Tensor(spin2d_dn)
print("spin2d size =", spin2d.size())

class NEncoder(nn.Module):
    def __init__(self, output_dim):
        super(NEncoder, self).__init__()
        self.output_dim = output_dim

        self.L1 = nn.Linear(s_len, h_node)
        self.b1 = nn.Linear(h_node, 1, bias=False)
        self.L2 = nn.Linear(h_node, output_dim)
        self.b2 = nn.Linear(output_dim, 1, bias=False)

        # setup the non-linearity
        self.act = nn.Tanh() #nn.Tanh()
        self.sig = nn.Sigmoid()

        nn.init.xavier_normal_(self.L1.weight)
        nn.init.xavier_normal_(self.L2.weight)
        self.b1.weight.data.fill_(0)
        self.b2.weight.data.fill_(0)

    def forward(self, x):
        h = x.view(-1, t_len, s_len)
        h = self.act(self.L1(h)+self.b1.weight)
        h = self.act(self.L2(h)+self.b2.weight)
        z = h.view(x.size(0), t_len, self.output_dim)
        return z

class NDecoder(nn.Module):
    def __init__(self, input_dim):
        super(NDecoder, self).__init__()
        self.L1 = nn.Linear(h_node, s_len)
        self.b1 = nn.Linear(s_len, 1, bias=False)
        self.L2 = nn.Linear(input_dim, h_node)
        self.b2 = nn.Linear(h_node, 1, bias=False)

        # setup the non-linearity
        self.act = nn.Tanh() #nn.Tanh()
        self.sig = nn.Sigmoid()

        nn.init.xavier_normal_(self.L1.weight)
        nn.init.xavier_normal_(self.L2.weight)
        self.b1.weight.data.fill_(0)
        self.b2.weight.data.fill_(0)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), z.size(2))
        h = self.act(self.L2(h)+self.b2.weight)
        mu_img = self.act(self.L1(h)+self.b1.weight)
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
        self.register_buffer('prior_params', torch.zeros(t_len, self.z_dim, 2))

        self.encoder = NEncoder(z_dim * self.q_dist.nparams)
        self.decoder = NDecoder(z_dim)

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
        x = x.view(x.size(0), 100, s_len)
        z_params = self.encoder.forward(x).view(x.size(0), t_len, self.z_dim, self.q_dist.nparams)
        zs = self.q_dist.sample(params=z_params)
        return zs, z_params

    def decode(self, z):
        x_params = self.decoder.forward(z).view(z.size(0), t_len, s_len)
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
        x = x.view(batch_size, 100, s_len)
        prior_params = self._get_prior_params(batch_size)
        x_recon, x_params, zs, z_params = self.reconstruct_img(x)
        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)

        elbo = logpx + logpz - logqz_condx # log p(x|z) + log p(z) - log q(z|x)
        # log q(z) ~ log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = self.q_dist.log_density(zs.view(batch_size, 1, t_len, self.z_dim),
                                         z_params.view(1, batch_size, t_len, self.z_dim, self.q_dist.nparams))

        if not self.mss: # minibatch weighted sampling
            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
        else: # minibatch stratified sampling
            logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, dataset_size).type_as(_logqz.data))
            logqz = logsumexp(logiw_matrix + _logqz.sum(3).sum(2), dim = 1, keepdim=False)
            logqz_prodmarginals = logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz.sum(2), dim = 1, keepdim = False).sum(1)

        if self.include_mutinfo:
            elbo2 = logpx - logqz_condx + logqz - self.beta * (logqz - logqz_prodmarginals) - \
                                (1 - self.lamb) * (logqz_prodmarginals - logpz)
        else:
            elbo2 = logpx - self.beta * (logqz - logqz_prodmarginals) - \
                                (1 - self.lamb) * (logqz_prodmarginals - logpz)

        return elbo2, elbo.detach()

if previous_model == True:
    model = TCVAE(z_dim=latent_dim, beta=beta, prior_dist=prior_dist, q_dist=q_dist)
    model.load_state_dict(torch.load('./frozen/VAE_NN10.pth'))

model = TCVAE(z_dim = latent_dim, beta = beta, prior_dist = prior_dist, q_dist = q_dist)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train_loader = DataLoader(dataset = spin2d, batch_size = 10, shuffle = True)

train_elbo = []
dataset_size = len(train_loader.dataset)
num_iter = 4001#len(train_loader)*20
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

    if n%10==0:
        train_elbo.append(-elbo_run_mean.avg)
        print('[n = %03d] ELBO = %.6f (%.6f)'%(n, elbo_run_mean.val, elbo_run_mean.avg))
        torch.save(model.state_dict(), f'frozen/VAE_{modelname}_5.pth')
        np.savetxt(f'frozen/elbo_{modelname}_5', np.array(train_elbo))

    if n>=200 and n%200==0: #testdata
        print('~~start recording')
        xs, x_params, zs, z_params = model.reconstruct_img(spin2dtest)
        xs_up, x_params_up, zs_up, z_params_up = model.reconstruct_img(spin2d_up_test)
        xs_dn, x_params_dn, zs_dn, z_params_dn = model.reconstruct_img(spin2d_dn_test)
        torch.save(z_params, f'frozen/{modelname}_zp_5.dat')
        torch.save(z_params_up, f'frozen/{modelname}_zp_up_5.dat')
        torch.save(z_params_dn, f'frozen/{modelname}_zp_dn_5.dat')
        xs_tot_up, x_params_tot_up, zs_tot_up, z_params_tot_up = model.reconstruct_img(spin2d_up)
        xs_tot_dn, x_params_tot_dn, zs_tot_dn, z_params_tot_dn = model.reconstruct_img(spin2d_dn)
        torch.save(xs_tot_up, f'frozen/{modelname}_field_up_5.dat')
        torch.save(xs_tot_dn, f'frozen/{modelname}_field_dn_5.dat')
        print('~~end recording')
        
        del xs, x_params, zs, z_params
        del xs_up, x_params_up, zs_up, z_params_up
        del xs_dn, x_params_dn, zs_dn, z_params_dn
        del xs_tot_up, x_params_tot_up, zs_tot_up, z_params_tot_up
        del xs_tot_dn, x_params_tot_dn, zs_tot_dn, z_params_tot_dn