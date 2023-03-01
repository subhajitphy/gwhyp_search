import numpy as np
import matplotlib.pyplot as plt

from enterprise.signals.signal_base import function as enterprise_function, PTA
from enterprise.signals.deterministic_signals import Deterministic
from enterprise.signals.parameter import Uniform
from enterprise.signals.gp_signals import MarginalizingTimingModel
from enterprise.signals.white_signals import MeasurementNoise
from enterprise.pulsar import Pulsar
import enterprise
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
import nestle
from gw_waveform_res import hyp_pta_res

import json
import os, sys

import enterprise
import libstempo as lst
import libstempo.plot as lstplot
import libstempo.toasim as toasim
import matplotlib.pyplot as plt

# code = sys.argv[0]
# index = int(sys.argv[1])

def hms_to_rad(hh, mm, ss):
    sgn = np.sign(hh)
    return sgn * (sgn * hh + mm / 60 + ss / 3600) * np.pi / 12


def dms_to_rad(dd, mm, ss):
    sgn = np.sign(dd)
    return sgn * (sgn * dd + mm / 60 + ss / 3600) * np.pi / 180

RA_GW = hms_to_rad(4, 0, 0)
DEC_GW = dms_to_rad(-45, 0, 0)


inc0=0;M0=2e10;q0=1;b0=60;e0=1.15;z0=0.3;

output_dir = "gwhyp_sims"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)



def save_psr_sim(psr1, savedir):
    print("Writing simulated data for", psr1.name)
    psr1.savepar(f"{savedir}/{psr1.name}_simulate.par")
    psr1.savetim(f"{savedir}/{psr1.name}_simulate.tim")
    lst.purgetim(f"{savedir}/{psr1.name}_simulate.tim")
    
day_to_s = 24 * 3600


#datadir = f"{enterprise.__path__[0]}/datafiles/mdc_open1"
psrname='J2124-3358'
datadir=f"/home/subhajit/Desktop/hypsearch/run2/NanoGrav_open_mdc"
parfile = f"{datadir}/"+psrname+".par"
timfile = f"{datadir}/"+psrname+".tim"

psr = Pulsar(parfile, timfile)

tref1 = (max(psr.toas)+min(psr.toas))/2

hyp_gw =hyp_pta_res(
cos_gwtheta=np.sin(DEC_GW),
gwphi=RA_GW,
psi=0,
cos_inc=np.cos(inc0),
log10_M=np.log10(M0),
q=q0,
b=b0,
e0=e0,
tref=tref1,
log10_z=np.log10(z0))


hyp_gw_fn = hyp_gw(name="hyp_gw", psr=psr)

res = hyp_gw_fn()

psrl = lst.tempopulsar(parfile=parfile, timfile=timfile)

def add_gwecc_1psr(psr):
    toas = (psr.toas() * day_to_s).astype(float)

    signal = (
        np.array(res

        )/day_to_s

    )

    #psr.stoas[:] += signal

    return signal

toasim.make_ideal(psrl)
toasim.add_efac(psrl, 1)
signal = add_gwecc_1psr(psrl)

lstplot.plotres(psrl, label="Residuals")
plt.plot(psrl.toas(), signal * day_to_s * 1e6, c="k", label="Injected signal")
plt.title(psrl.name)
plt.legend()
#plt.savefig(arr[index][:-4]+'.png',dpi=200)
#plt.show()
plt.savefig('sampleplot.png',dpi=200)
psrl.fit()

save_psr_sim(psrl, output_dir)



