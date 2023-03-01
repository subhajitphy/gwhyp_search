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


def hms_to_rad(hh, mm, ss):
    sgn = np.sign(hh)
    return sgn * (sgn * hh + mm / 60 + ss / 3600) * np.pi / 12


def dms_to_rad(dd, mm, ss):
    sgn = np.sign(dd)
    return sgn * (sgn * dd + mm / 60 + ss / 3600) * np.pi / 180


datadir = enterprise.__path__[0] + '/datafiles/mdc_open1/'

parfiles = sorted(glob.glob(datadir + '/*.par'))
timfiles = sorted(glob.glob(datadir + '/*.tim'))

psrs = []
for p, t in zip(parfiles, timfiles):
    psr = Pulsar(p, t)
    psrs.append(psr)
    

# find the maximum time span to set GW frequency sampling
tmin = [p.toas.min() for p in psrs]
tmax = [p.toas.max() for p in psrs]
Tspan = np.max(tmax) - np.min(tmin)

##### parameters and priors #####

# white noise parameters
# in this case we just set the value here since all efacs = 1 
# for the MDC data
#efac = parameter.Constant(1.0)

efac = parameter.Uniform(0.1, 5.0)
# red noise parameters 
log10_A = parameter.Uniform(-18,-12)
gamma = parameter.Uniform(0,7)

##### Set up signals #####

# white noise
ef = white_signals.MeasurementNoise(efac=efac)

# red noise (powerlaw with 15 frequencies)
pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
rn = gp_signals.FourierBasisGP(spectrum=pl, components=15, Tspan=Tspan)

# gwb
# We pass this signal the power-law spectrum as well as the standard 
# Hellings and Downs ORF
orf = utils.hd_orf()
gwb = gp_signals.FourierBasisCommonGP(pl, orf, components=15, name='gw', Tspan=Tspan)

# timing model
tm = gp_signals.TimingModel()

##########################################################################

from gw_waveform_res import hyp_pta_res





RA_GW = hms_to_rad(4, 0, 0)
DEC_GW = dms_to_rad(-45, 0, 0)



tref1 = (max(psr.toas)+min(psr.toas))/2





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


hyp = memory_block_hyp()

#############################
# full model is sum of components
model = ef + tm +hyp


# initialize PTA
pta = signal_base.PTA([model(psr) for psr in psrs])


# initial parameters
xs = {par.name: par.sample() for par in pta.params}

# dimension of parameter space
ndim = len(xs)

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2)

# set up jump groups by red noise groups 
ndim = len(xs)
groups  = [range(0, ndim)]
groups.extend(map(list, zip(range(0,ndim,2), range(1,ndim,2))))

sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups, 
                 outDir='chains/mdc/open1_hyp/')

# sampler for N steps
N = 10000
x0 = np.hstack(p.sample() for p in pta.params)
sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)