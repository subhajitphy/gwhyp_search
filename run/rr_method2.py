import numpy as np
from numpy import cosh, sqrt, pi
from hypmik3pn import get_u, get_u_hat, get_u_v2
from getx import get_x
from scipy.integrate import solve_ivp
from scipy.integrate import odeint

def odes(eta,b1,y,t):
    et=y[0]
    n=y[1]
    l=y[2]
    #u=get_u_hat(l,et)
    #u=get_u(l,et,eta,b1,3)
    x=n**(2/3)
    u=get_u_v2(l,et,eta,x,3)
    w=et*cosh(u)-1
    
    
    dedt_Q=(8/15*(et**2-1)*x**4*(35*(1-et**2)+(49-9*et**2)*w+17*w**2+3*w**3)*eta/et/w**6)
    dndt_Q=(8/5*x**(11/2)*(35*(1-et**2)+49*w+32*w**2+6*w**3-9*w*et**2)*eta/w**6)
    
    dedt_1PN=(-2/315/et/w**8*x**5*(-17640*(et**2-1)**4+63*w*(et**2-1)**3*(140*eta+657)-105*w**2*
    (et**2-1)**2*(13+454*eta+9*et**2*(3+2*eta))-w**4*(et**2-1)*(36825-53060*eta+9*et**2*(
    -2169+560*eta))+6*w**6*(360-553*eta+et**2*(-444+637*eta))-28*w**3*(et**2-1)*(1827-
    2755*eta+et**2*(-1767+1105*eta)+w**5*(10215-18088*eta+et**2*(-12735+20608*eta)))))
    
    dndt_1PN=(2/35/w**8*x**(13/2)*(w**6*(180-588*eta)+w**5*(1340-5852*eta)+2*w**4*(9*et**2*
    (21*eta-1)-8589*eta+1003)+35*w**3*(et**2*(244*eta-5)-684*eta+21)+35*w**2*(et**2-1)*
    (9*et**2*(2*eta-17)+454*eta+193)-21*w*(et**2-1)**2*(140*eta+657)+5880*(et**2-1)**3))
    dldt=n
    return [(dedt_Q),(dndt_Q),dldt]
    #return [(dedt_Q+dedt_1PN),(dndt_Q+dndt_1PN),dldt]


def solve_rr(eta,b1,y0,Ti,Tf,Tarr):
    sol = solve_ivp(lambda t,y:odes(eta,b1,y,t),[Ti,Tf],y0,t_eval=Tarr,rtol=1e-10, atol=1e-10)
    Earr=sol.y[0]
    Narr=sol.y[1]
    Larr=sol.y[2]
    return Earr, Narr, Larr


def solve_rr2(eta,b1,y0,Ti,Tf,Tarr):
    sol =odeint(lambda t,y:odes(eta,b1,y,t),y0,Tarr,tfirst=True,rtol=1e-10, atol=1e-10)
    earr=sol[:, 0]
    narr=sol[:,1]
    larr=sol[:,2]
    return earr, narr, larr

def solve_rr3(eta,b1,y0,Ti,Tf):
    sol = solve_ivp(lambda t,y:odes(eta,b1,y,t),[Ti,Tf],y0,rtol=1e-10, atol=1e-10)
    Earr=sol.y[0]
    Narr=sol.y[1]
    Larr=sol.y[2]
    Tarr=sol.t
    return Earr, Narr, Larr, Tarr


