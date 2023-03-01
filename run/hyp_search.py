#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division

import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.linalg as sl
from enterprise.signals.parameter import Uniform
import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals.deterministic_signals import Deterministic
import corner
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc


# In[2]:


def hms_to_rad(hh, mm, ss):
    sgn = np.sign(hh)
    return sgn * (sgn * hh + mm / 60 + ss / 3600) * np.pi / 12


def dms_to_rad(dd, mm, ss):
    sgn = np.sign(dd)
    return sgn * (sgn * dd + mm / 60 + ss / 3600) * np.pi / 180


# In[3]:


datadir = f"{enterprise.__path__[0]}/datafiles/mdc_open1"
parfile = f"{datadir}/J0030+0451.par"
timfile = f"{datadir}/J0030+0451.tim"

psr = Pulsar(parfile, timfile)


# In[4]:


from gw_waveform_res import hyp_pta_res


# In[5]:


RA_GW = hms_to_rad(4, 0, 0)
DEC_GW = dms_to_rad(-45, 0, 0)


# In[6]:


tref1 = (max(psr.toas)+min(psr.toas))/2


# In[7]:


def memory_block_hyp(
    cos_gwtheta=Uniform(-1,1)("hyp_cgwt"),
    gwphi=Uniform(0,2*np.pi)("hyp_cgwp"),
    psi=Uniform(0,np.pi)("hyp_psi"),
    cos_inc=Uniform(-1,1)("hyp_ci"),
    log10_M=Uniform(8,10)("hyp_m"),
    q=1,
    b=Uniform(60,100)("hyp_b"),
    e0=Uniform(1.1,1.4)("hyp_e"),
    log10_z=Uniform(-1,0.3)("hyp_z"),
    tref=tref1):
    return Deterministic(hyp_pta_res(cos_gwtheta=cos_gwtheta,gwphi=gwphi,psi=psi,cos_inc=cos_inc
                                     ,log10_M=log10_M,q=q,b=b,e0=e0,log10_z=log10_z,tref=tref),name="hyp")


# In[8]:


hyp = memory_block_hyp()


# In[9]:


##### parameters and priors #####

# Uniform prior on EFAC
efac = parameter.Uniform(0.1, 5.0)

# red noise parameters
# Uniform in log10 Amplitude and in spectral index
log10_A = parameter.Uniform(-18,-12)
gamma = parameter.Uniform(0,7)

##### Set up signals #####

# white noise
ef = white_signals.MeasurementNoise(efac=efac)

# red noise (powerlaw with 30 frequencies)
pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
rn = gp_signals.FourierBasisGP(spectrum=pl, components=30)

# timing model
tm = gp_signals.TimingModel()

# full model is sum of components
model = ef + rn + tm+hyp

# initialize PTA
pta = signal_base.PTA([model(psr)])


# In[10]:


print(pta.params)


# In[11]:


xs = {par.name: par.sample() for par in pta.params}
print(xs)


# In[12]:




# In[13]:


x0 = np.hstack(p.sample() for p in pta.params)


# In[14]:


# dimension of parameter space
ndim = len(xs)

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2)

# set up jump groups by red noise groups 
ndim = len(xs)
groups  = [range(0, ndim)]
groups.extend([[1,2]])

# intialize sampler
sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups, 
                 outDir='chains/mdc/open1/')


# In[15]:


# sampler for N steps
N = int(1e6)
x0 = np.hstack(p.sample() for p in pta.params)
sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)


# In[ ]:


# chain = np.loadtxt('chains/mdc/open1/chain_1.txt')
# pars = sorted(xs.keys())
# burn = int(0.25 * chain.shape[0])


# # In[ ]:


# truths = [1.0, 4.33, np.log10(5e-14)]
# #corner.corner(chain[burn:,:-4], 30, truths=truths, labels=pars);
# corner.corner(chain[burn:,:-4], 30, labels=pars)
# plt.savefig('plot.pdf',dpi=300)


# In[6]:




# In[ ]:




