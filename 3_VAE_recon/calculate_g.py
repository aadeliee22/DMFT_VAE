#!/usr/bin/env python3
import os
os.environ['OMP_NUM_THREADS'] = '4'
from mpi4py import MPI
import numpy as np
from gfmod import *

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader


import lib.dist as dist
import lib.utils as utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-Uini", help = "on-site interaction", type = float, default = None)
parser.add_argument("-Ufin", help = "on-site interaction", type = float)
parser.add_argument("-beta", help = "1/temperature", type = float, default = 60)
parser.add_argument("-way", help = "direction", type = int)
parser.add_argument("-nwarm", help = "mc warmup step", type = int, default = 300)
parser.add_argument("-nmeas", help = "mc measure step", type = int, default = 3000)
parser.add_argument("-dg", help = "delta G", type = float, default = 1e-3)
parser.add_argument("-znode", help = "number of z node", type = int, default = None)
parser.add_argument("-model", help = "model name", type = str, default = None)
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

beta = args.beta
if beta == 40:
    niwn = 120
    ntau = 196
if beta == 60:
    niwn = 180
    ntau = 289

# DMFT-HFQMC parameter
Ufin = args.Ufin
mu = Ufin / 2.0
D = 1
nwarm = args.nwarm
nmc_per_meas = 10
nmeas = args.nmeas
nmeas_per_cpu = nmeas//comm.Get_size()
seed = 0
seedDist = 2*rank*ntau*(nmeas*nmc_per_meas+nwarm)
nloop = 30
dG = args.dg
way = args.way

# Beta-TCVAE parameter
latent_dim = args.znode
b = 4
modelname = args.model
prefix = './Hirsch-Fye-QMC'


def print_log_msg(logtype, msg, comm):
    if comm.Get_rank() == 0:
        print(f'[{logtype}] : {msg}')
def allreduce(target, comm):
    dtype = target.dtype
    if dtype == 'float64':
        mpi_dtype = MPI.DOUBLE
    else:
        raise Exception("Check target.dtype : ", dtype)
    target_all = np.zeros_like(target)
    comm.Reduce([target, mpi_dtype], [target_all, mpi_dtype], op=MPI.SUM, root=0)
    comm.Bcast(target_all, root=0)
    return np.array(target_all/comm.Get_size(), dtype = dtype)

iwn = np.array([1j*(2*n+1)*np.pi/beta for n in range(niwn)])
tau = np.linspace(0, beta, ntau)

# initial guess: semicircular or load
Giwn = gftools.SemiCircularGreen(niwn, beta, 0, 1, D)
if (args.Uini!=None):
    Uini = args.Uini
    print_log_msg("INFO", f"Load previous giwn from {Uini}", comm)
    w, Gr, Gi = np.loadtxt(f'./Bethe_{way:.0f}_beta{beta:.0f}/Giwn-{Uini:.2f}.dat', unpack = True, dtype = 'float64')
    Giwn = Gr + 1j*Gi
Gtau = gftools.GreenInverseFourier(Giwn, ntau, beta)

# spin generation from VAE
print_log_msg("INFO", f"Generate U = {Ufin} spins from {modelname}", comm)
txt = np.loadtxt(f'./Bethe_{way:.0f}_beta{beta:d}/field-{Ufin:.2f}.dat')
s_tot, s_len = txt.shape
s_num = 30
t_len = 100
txt = 0.5*(txt+1)
spin2d = txt.reshape(s_num, t_len, s_len)
spin2d = torch.Tensor(spin2d)
prior_dist = dist.Normal()
q_dist = dist.Normal()


class NEncoder(nn.Module):
    def __init__(self, output_dim):
        super(NEncoder, self).__init__()
        self.output_dim = output_dim
        self.L1 = nn.Linear(s_len, output_dim)
        self.b1 = nn.Linear(output_dim, 1, bias=False)
        self.act = nn.Tanh() #nn.Tanh()
        self.sig = nn.Sigmoid()
        nn.init.xavier_normal_(self.L1.weight)
        self.b1.weight.data.fill_(0)
    def forward(self, x):
        h = x.view(-1, t_len, s_len)
        z = self.act(self.L1(h)+self.b1.weight).view(x.size(0), t_len, self.output_dim)
        return z

class NDecoder(nn.Module):
    def __init__(self, input_dim):
        super(NDecoder, self).__init__()
        self.L1 = nn.Linear(input_dim, s_len)
        self.b1 = nn.Linear(s_len, 1, bias=False)
        self.act = nn.Tanh() #nn.Tanh()
        self.sig = nn.Sigmoid()
        nn.init.xavier_normal_(self.L1.weight)
        self.b1.weight.data.fill_(0)
    def forward(self, z):
        h = z.view(z.size(0), z.size(1), z.size(2))
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

### Beta-TCVAE
class TCVAE(nn.Module):
    def __init__(self, z_dim, b, prior_dist=dist.Normal(), q_dist=dist.Normal(),
                 include_mutinfo=True, tcvae=True, mss=True):
        super(TCVAE, self).__init__()

        self.z_dim = z_dim
        self.include_mutinfo = include_mutinfo
        self.tcvae = tcvae
        self.lamb = 0
        self.b = b
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

    
model = TCVAE(z_dim=latent_dim, b=b, prior_dist=prior_dist, q_dist=q_dist)
model.load_state_dict(torch.load(f'../frozen/VAE_{modelname}_5.pth'))
xs, x_params, zs, z_params = model.reconstruct_img(spin2d)

spins = np.array(xs.reshape(s_tot, s_len))
spins = 2*spins-1
print_log_msg("INFO", f"spin2d.shape = {spins.shape}", comm)


for niter in range(nloop):
    giwn0 = 1/(iwn + mu - (D/2)**2 * Giwn - Ufin/2)
    gtau0 = gftools.GreenInverseFourier(giwn0, ntau, beta)
    solver = qmc.hfqmc(gtau0, beta, Ufin, seed, seedDist)
    
    print_log_msg("INFO", \
        f"measuring full Green's function ({nmeas_per_cpu} measurements per cpu)", comm)
    Gtau_up, err_up, Gtau_dw, err_dw = solver.meas(spins, nmeas_per_cpu)
    comm.Barrier()

    # A system is in the paramagnetic phase.
    Gtau_new = allreduce((Gtau_up + Gtau_dw) / 2, comm)
    Gtau_err = np.sqrt(allreduce(((err_up + err_dw) / 2)**2, comm)/comm.Get_size()**3)
    dist = np.mean(np.abs(Gtau_new - Gtau))
    if np.mean(Gtau_err) > dG:
        print_log_msg("WARNING", \
            f"Measurement error is larger than 'dG({dG})'. \
            One should increase the # of measurements.", comm)
    
    Gtau = np.array(Gtau_new)
    Giwn = gftools.GreenFourier(Gtau, niwn, beta)
    print_log_msg("INFO", f"MEAN(|G_new - G_old|) : {dist:.3E},  \
                    ERR(G_new) : {np.mean(Gtau_err):.3E}", comm)

    zs = model.q_dist.sample(params=z_params)
    xs, x_params = model.decode(zs)
    spins = np.array(xs.reshape(s_tot, s_len))
    spins = 2*spins-1

    print_log_msg("INFO", \
        f"Saving to test_{args.model}\n", comm)

    if rank == 0:
        np.savetxt(f"./test_{args.model}/Bethe_{way:.0f}_beta{beta:.0f}/Gtau_recon-{Ufin:.2f}.dat", \
            np.array([tau, Gtau, Gtau_err]).T)
        np.savetxt(f'./test_{args.model}/Bethe_{way:.0f}_beta{beta:.0f}/Giwn_recon-{Ufin:.2f}.dat', \
            np.array([iwn.imag, Giwn.real, Giwn.imag]).T)
        # Dyson equation : \Sigma(iwn) = G0^{-1}(iwn) - G^{-1}(iwn)
        self_energy = (iwn + mu - (D/2)**2 * Giwn) - 1/Giwn
        np.savetxt(f"./test_{args.model}/Bethe_{way:.0f}_beta{beta:.0f}/self_energy_recon-{Ufin:.2f}.dat", \
            np.array([iwn.imag, self_energy.real, self_energy.imag]).T)
    if dist < dG:
        print_log_msg("INFO", "CONVERGE!", comm)
        break