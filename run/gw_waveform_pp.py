from constants import *
import numpy as np
from numpy import sin, cos, cosh, sqrt, pi
from scipy.integrate import odeint
from getx import get_x
from gw_functions import rx, phitx, phiv, rtx
import matplotlib.pyplot as plt
from eval_max import get_max, Fomg
import antenna_pattern as ap
from rr_method1 import solve_rr
from rr_method2 import solve_rr2
from hypmik3pn import get_u, get_u_v2
from scipy.integrate import cumtrapz
from scipy.optimize import curve_fit
from astropy.cosmology import Planck18
from scipy.interpolate import CubicSpline
from enterprise.signals import signal_base

def get_hyp_waveform(M,q,et0,b,ti,tf,t_step,inc,distance,order
                     ,estimatepeak='None',rr='None',rrmethod='None'):
   
    
        eta=q/(1+q)**2
        Time=M*tsun
        dis=M*dsun
        scale=distance/dis
        x0=get_x(et0,eta,b,3)[0]
        n0=x0**(3/2)
        tarr=np.linspace(ti,tf,t_step)
        t_arr=tarr/Time
        t_i=t_arr[0]
        t_f=t_arr[len(t_arr)-1]
        l_i=n0*t_i
        
        if rr=='False':
            larr=n0*t_arr
            u_method1=get_u(larr,et0,eta,b,3)
            earr=et0*np.ones(len(larr))
            narr=n0*np.ones(len(larr))

        #solve using (u,e_t,n) method: LAL way
            
        elif rrmethod=='dudt':
            u_i=get_u(l_i,et0,eta,b,3)
            y0=[et0,n0,u_i]
            sol=solve_rr(eta,b,y0,t_i,t_f,t_arr)
            uarr=sol[2]
            earr=sol[0]
            narr=sol[1]

        #solve using (u,e_t,n) method: PTA way
        else:
            y0=[et0,n0,l_i]
            sol2=solve_rr2(eta,b,y0,t_i,t_f,t_arr)
            larr=sol2[2]
            narr=sol2[1]
            earr=sol2[0]
            xarr=narr**(2/3) 
            uarr=get_u_v2(larr,earr,eta,xarr,3)
            


        step=len(tarr)
        hp_arr=np.zeros(step)
        hx_arr=np.zeros(step)
        X=np.zeros(step)
        Y=np.zeros(step)
        for i in range(step):
            et=earr[i]
            u=uarr[i]
            x=narr[i]**(2/3) 
            phi=phiv(eta,et,u,x,order)
            r1=rx(eta,et,u,x,order)
            z=1/r1
            phit=phitx(eta,et,u,x,order)
            rt=rtx(eta,et,u,x,order)
            phi=phiv(eta,et,u,x,order)
            phi=phiv(eta,et,u,x,order)
            r1=rx(eta,et,u,x,order)
            X[i]=r1*cos(phi)
            Y[i]=r1*sin(phi)
            hp_arr[i]=(-eta*(sin(inc)**2*(z-r1**2*phit**2-rt**2)+(1+cos(inc)**2)*((z
            +r1**2*phit**2-rt**2)*cos(2*phi)+2*r1*rt*phit*sin(2*phi))))
            hx_arr[i]=(-2*eta*cos(inc)*((z+r1**2*phit**2-rt**2)*sin(2*phi)-2*r1*rt*phit*cos(2*phi)))
        Hp=hp_arr/scale
        Hx=hx_arr/scale

        #Eliminate DC offset term at -infinity
        if estimatepeak=='True':
            dimless_peak=get_max(eta,b,et0)
            peak=dimless_peak/(2*np.pi*Time)
            return Hp-Hp[0],Hx-Hx[0], peak, X, Y
        else:
            return Hp-Hp[0],Hx-Hx[0]


def func(x, a0, a1, a2):
    return (a0+a1*x+a2*x**2)


@signal_base.function
def hyp_pta_res(toas,
    theta,
    phi,
    cos_gwtheta,
    gwphi,
    psi,
    cos_inc,
    log10_M,
    q,
    b,
    e0,
    log10_z,
    tref,
    interp_steps=1000
):
    """
    Compute the PTA signal due to a hyperbolic encounter.

    toas        are pulsar toas in s in SSB frame
    theta       is pulsar zenith angle in rad
    phi         is pulsar RA in rad
    cos_gwtheta is cos zenith angle of the GW source
    gwphi       is the RA of the GW source in rad
    psi         is the GW polarization angle in rad
    cos_inc     is the cos inclination of the GW source
    log10_M     is the log10 total mass of the GW source in solar mass
    q           is the mass ratio of the GW source
    b           is the impact parameter of the GW source in solar mass
    e0          is the eccentricity of the GW source
    log10_z     is the log10 cosmological redshift of the GW source
    tref        is the fiducial time in s in SSB frame
    interp_steps is the number of samples used for interpolating the PTA signal
    """
    order = 3

    z = 10**log10_z
    D_GW = 1e6 * Planck18.luminosity_distance(z).value * pc # meter

    ts = toas - tref

    # ti, tf, tzs in seconds, in source frame
    ti = min(ts)/(1+z)
    tf = max(ts)/(1+z)
    tz_arr = np.linspace(ti, tf, interp_steps)
    delta_t_arr = (tz_arr[1]-tz_arr[0]) * (1+z) # second, in SSB frame

    tzs = ts/(1+z)
    
    M = 10**log10_M # Solar mass

    inc = np.arccos(cos_inc)

    gwra = gwphi
    gwdec = np.arcsin(cos_gwtheta)

    psrra = phi
    psrdec = np.pi/2 - theta

    
    hp_arr, hx_arr = get_hyp_waveform(M, q, e0, b, ti, tf, interp_steps, inc, D_GW, order)

    cosmu, Fp, Fx = ap.antenna_pattern(gwra, gwdec, psrra, psrdec)

    c2psi = np.cos(2*psi)
    s2psi = np.sin(2*psi)
    Rpsi = np.array([[c2psi, -s2psi],
                     [s2psi, c2psi]])
    h_arr = np.dot([Fp,Fx], np.dot(Rpsi, [hp_arr,hx_arr]))

    # Integrate over time in SSB frame
    s_arr = cumtrapz(h_arr, initial=0)*delta_t_arr

    s_spline = CubicSpline(tz_arr, s_arr)
    s_pre = s_spline(ts)

    #return  s_pre, tzs/yr

    return s_pre


class waveform:
   
    def __init__(self,M,q,et0,b,toas,tref,inc,z,psi,psrra='None'
                 ,psrdec='None',gwra='None',gwdec='None',order='None',
                 estimatepeak='None',rr='None',rrmethod='None'):
        D_GW = 1e6 * Planck18.luminosity_distance(z).value * pc # meter
        ts = toas - tref
        interp_steps=1000
        # ti, tf, tzs in seconds, in source frame
        ti = min(ts)/(1+z)
        tf = max(ts)/(1+z)
        tz_arr = np.linspace(ti, tf, interp_steps)
        delta_t_arr = (tz_arr[1]-tz_arr[0])*(1+z) #in source frame

        if order=='None':
            order=3

        
        
        if psrra=='None' and psrdec=='None' and gwra=='None' and gwdec=='None'and estimatepeak=='None':
            wv=get_hyp_waveform(M,q,et0,b,ti,tf,interp_steps,inc,D_GW,order)
            hp,hx=wv
            self.hp = hp
            self.hx = hx
            self.get_sampletimes=tz_arr/yr
        
        elif psrra=='None' and psrdec=='None' and gwra=='None' and gwdec=='None' and estimatepeak=='True':
            hp1,hx1,peak1,x1,y1=get_hyp_waveform(M,q,et0,b,ti,tf,interp_steps,inc,D_GW,order,estimatepeak='True')
            
            self.hp = hp1
            self.hx = hx1
            self.x=x1
            self.y=y1
            self.peak=peak1
            self.get_sampletimes=tz_arr/yr
            
        else:
            if rrmethod=='dudt':
                hp,hx=get_hyp_waveform(M,q,et0,b,ti,tf,interp_steps,inc,D_GW,order,rrmethod='dudt')
            else:
                hp,hx=get_hyp_waveform(M,q,et0,b,ti,tf,interp_steps,inc,D_GW,order)
    
            
            cosmu, Fp, Fx = ap.antenna_pattern(gwra, gwdec, psrra, psrdec)

            sp=cumtrapz(hp,initial=0)*delta_t_arr
            sx=cumtrapz(hx,initial=0)*delta_t_arr

            c2psi = np.cos(2*psi)
            s2psi = np.sin(2*psi)
            Rpsi = np.array([[c2psi, -s2psi],
                             [s2psi, c2psi]])
            h_arr = np.dot([Fp,Fx], np.dot(Rpsi, [hp,hx]))

            # Integrate over time in SSB frame
            res_pre = cumtrapz(h_arr, initial=0)*delta_t_arr

            s_spline = CubicSpline(tz_arr, res_pre )
            s_pre = s_spline(ts)
            

            popt, pcov = curve_fit(func, ts, s_pre)
            s_post=s_pre-func(ts, *popt)

            hp_1 = CubicSpline(tz_arr, hp )
            hx_1= CubicSpline(tz_arr, hx )

            sp_1 = CubicSpline(tz_arr, sp )
            sx_1 = CubicSpline(tz_arr, sx )

            sp=sp_1(ts)
            sx=sx_1(ts)

            hp=hp_1(ts)
            hx=hx_1(ts)

            self.hp = hp
            self.hx = hx
            
            self.sp=sp/1e-9
            self.sx=sx/1e-9

            self.get_sampletimes=ts/yr

            
            self.prefitres=s_pre/1e-9
            self.postfitres=s_post/1e-9

