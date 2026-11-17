#@title --- imports and definitions ---

# !apt install libgraphviz-dev
# !pip install pygraphviz

import pandas as pd
from pyfish import process_data, setup_figure, fish_plot
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.cm as cm
import matplotlib as mpl

import networkx as nx
from matplotlib.colors import to_rgba
import colorsys
from copy import deepcopy

import pickle
import bz2

try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

if IN_COLAB:
  from google.colab import files
  from tqdm.notebook import tqdm
else:
  tqdm = tqdm = lambda x:x

I=np.newaxis

def monod(Z, F):
    """Elementwise Monod: Z/(Z+F). Shapes must broadcast."""
    return Z / (Z + F)

# pack/unpack -----------------------------------------------------------
def pack_state(X, Y, m, P, R):
    # X:(M,1), Y:(S,1), m:(S,1), P:(M,S), R:(S,1)
    return np.concatenate([
        X.ravel(),            # M
        Y.ravel(),            # S
        m.ravel(),            # S
        P.ravel(),            # M*S
        R.ravel()             # S
    ])

def unpack_state(z, M, S):
    NX, NY, NM, NP = M, S, S, M*S
    off = 0
    X = z[off:off+NX].reshape(M, 1); off += NX
    Y = z[off:off+NY].reshape(S, 1); off += NY
    m = z[off:off+NM].reshape(S, 1); off += NM
    P = z[off:off+NP].reshape(M, S); off += NP
    R = z[off:off+NY].reshape(S, 1)
    return X, Y, m, P, R

# RHS -------------------------------------------------------------------
def rhs(t, z, params, optAlloc=False):
    """
    params:
      M,S (ints)
      D (scalar)
      U:(M,S), T:(S,1), L=Lambda:(M,1)
      Fn=(F^n):(M,S), Ft=(F^t):(S,1)
      Phi:(M,S) target uptake proteome fractions (per species)
      psi:(S,1)  target ribosomal fraction
      f_n,f_t: Monod-like callables (default: monod)
      enforce_m_cap: bool -> project growth if sum m > 1 (optional)
    """

    M, S = params["M"], params["S"]
    D  = params["D"]
    U  = params["U"]
    T  = params["T"]
    L  = params["L"]
    Fn = params["Fn"]
    Ft = params["Ft"]
    Phi = params["Phi"]
    psi = params["psi"]
    f_n = params.get("f_n", monod)
    f_t = params.get("f_t", monod)

    X, Y, m, P, R = unpack_state(z, M, S)

    eps_A = 112
    eps_P = 33600
    eps_R = 835408
    eps_C = 180.16
    cP    = 3e6

    if optAlloc:
        critC= U * monod(X,Fn)
        global smoothChoice

        if not smoothChoice:
            ## A. Argmax-choice
            winner = np.argmax(critC,axis=0)
            wi = winner,np.arange(U.shape[1])
            d=U[wi] * monod(X, Fn)[wi] /T.T*eps_R/eps_P*eps_C/eps_A
            # d=critC[wi] /T.T*eps_R/eps_P*eps_C/eps_A# does it work?
            psi = d/(d+1)
            Phi=np.zeros((M,S));Phi[wi]=1-psi
            psi = psi.reshape(S, 1)

        else:
            ## B. Smooth choice
            softWin = smoothArgMax(critC,.1)
            d=(critC*softWin).sum(0) /T.T*eps_R/eps_P*eps_C/eps_A
            psi = d/(d+1)
            Phi= (1-psi)*softWin
            psi = psi.reshape(S, 1)

        ## C. Silence any optimization
        # psi=params["psi"]
        # Phi=params["Phi"]


    xi_AR = psi*eps_A/eps_R
    xi_AP = Phi*eps_A/eps_P
    xi_CA =     eps_C/eps_A
    sms = xi_AP.sum(0,keepdims=True).T+xi_AR

    # Monod terms
    fX = f_n(X, Fn)          # (M,S) via X:(M,1) broadcasting across species
    fY = f_t(Y, Ft)          # (S,1)

    TRfY = T * R * fY
    # Growth factor (non-dimensional)
    g = (T * R * fY * sms - 1)*D

    # Fluxes
    supply_Y = (U * P * fX).sum(axis=0, keepdims=True).T      # (S,1)
    uptake_X = (U * P * fX * m.T).sum(axis=1, keepdims=True)  # (M,1)

    # ODEs (dimensionless)
    dm = m * g                                          # (S,1)
    dY = D * (supply_Y * xi_CA - TRfY * (Y * sms + 1.0))                               # (S,1)
    dX = D * (L - X - uptake_X)                                      # (M,1)
    dP = D * (TRfY.T * xi_AP - P*(1 + g/D).T)                                        # (M,S)
    dR = D * (TRfY   * xi_AR - R*(1 + g/D))                                 # (S,1)

    return pack_state(dX, dY, dm, dP, dR)

# Integrator ------------------------------------------------------------
def integrate_system(X0, Y0, m0, P0, R0,
                     D, U, T, L, Fn, Ft, Phi, psi,
                     t_span, t_eval=None, method="BDF", optAlloc=False):
    """
    Shapes:
      X0:(M,1),  Y0:(S,1),  m0:(S,1),
      P0:(M,S),  R0:(S,1)
      U:(M,S),   L:(M,1),   T:(S,1), Fn:(M,S), Ft:(S,1),
      Phi:(M,S), psi:(S,1)
    """
    M, _ = X0.shape
    S, _ = Y0.shape
    z0 = pack_state(X0, Y0, m0, P0, R0)

    params = dict(M=M, S=S, D=D, U=U, T=T, L=L, Fn=Fn, Ft=Ft,
                  Phi=Phi, psi=psi, f_n=monod, f_t=monod,
                  enforce_m_cap=False)

    sol = solve_ivp(lambda t, z: rhs(t, z, params, optAlloc=optAlloc),
                    t_span, z0, t_eval=t_eval,
                    method=method, atol=1e-8, rtol=1e-6)
    if not sol.success:
        raise RuntimeError(sol.message)

    nT = sol.t.size
    X, Y, m, P, R = unpack_time_series(sol.y.T, M, S, nT)
    return dict(t=sol.t, X=X, Y=Y, m=m, P=P, R=R)

def unpack_time_series(YT, M, S, nT):
    # inverse of pack_state over time
    NX, NY, NM, NP = M, S, S, M*S
    X = np.empty((nT, M, 1));  Y = np.empty((nT, S, 1))
    m = np.empty((nT, S, 1));  P = np.empty((nT, M, S))
    R = np.empty((nT, S, 1))
    for k in range(nT):
        X[k], Y[k], m[k], P[k], R[k] = unpack_state(YT[k], M, S)
    return X, Y, m, P, R

def gen_rng(seed=None, verbose=False):
  srng = np.random.default_rng()
  seed = srng.integers(1000) if seed is None else seed
  if verbose: print ("seed:", seed)
  return np.random.default_rng(seed)

def lambda_from_alloc(c, params, *, atol=1e-12):
    """
    Compute the growth rate, given a particular environment and conditions.
    It does it assuming fast internal metabolism.

    Returns:
      lam : (S,1) growth rate for each species
    """

    M, S = c.shape
    D  = params["D"]
    U  = params["U"]
    T  = params["T"]
    Fn = params["Fn"]
    Ft = params["Ft"]
    Phi = params["Phi"]
    psi = params["psi"]
    f_n = params.get("f_n", monod)
    f_t = params.get("f_t", monod)
    global eps_A, eps_P, eps_R, eps_C # molar masses don't change

    xi_AP = Phi * (eps_A / eps_P)      # (M,S)
    xi_AR = psi * (eps_A / eps_R)      # (S,1)
    xi_CA =  eps_C / eps_A             # scalar

    # sum of AA-budget shares
    sms = xi_AP.sum(axis=0, keepdims=True).T + xi_AR   # (S,1)

    # b collects terms in dot(ã)=0 that don't depend on a
    # b_s = [sum_i U_is * xi_AP_is * f_n(c_i; Fn_is)] / [T_s * xi_AR_s] * xi_CA
    b_num = (U * xi_AP * f_n(c, Fn)).sum(axis=0, keepdims=True).T   # (S,1)
    b_den = T * xi_AR                                               # (S,1)
    # Avoid divide by zero; if b_den=0, b→∞ is impossible physically (psi=0 implies no translation)
    b = (b_num * xi_CA) / np.maximum(b_den, atol)

    # Solve for x = f_t(a*) analytically without recovering a:
    # (Ft*sms - 1) x^2 + (b+1) x - b = 0  (choose physical root in [0,1))
    Ft_sms = Ft * sms                                 # (S,1)
    A = Ft_sms - 1.0
    B = b + 1.0
    C = -b

    # Handle the knife-edge A≈0 separately: then x = b/(b+1)
    near_edge = np.isclose(A, 0.0, atol=1e-12)
    x_star = np.empty_like(b)

    # General case (A != 0): take the root with '+' in the numerator
    disc = np.maximum(B*B - 4*A*C, 0.0)
    x_general = (np.sqrt(disc) - B) / (2*A)  # algebraically equal to previous form
    # Clip to [0, 1 - tiny] for numerical safety
    x_general = np.clip(x_general, 0.0, 1.0 - 1e-12)

    # Edge case A≈0 → linear (b+1)x - b = 0
    x_edge = b / np.maximum(b + 1.0, atol)
    x_edge = np.clip(x_edge, 0.0, 1.0 - 1e-12)

    x_star[~near_edge] = x_general[~near_edge]
    x_star[ near_edge] = x_edge[ near_edge]

    # Growth: λ = D * (T * xi_AR * x* - 1)
    lam = D * (T * xi_AR * x_star - 1.0)
    return lam

def optimize_allocation(c, params):
    """
    Given an instantaneous condition of external concentrations of nutrients and
    the parameters defining the features of the chemostat and cell strains, find
    the optimal allocation of proteome fractions that maximizes growth rates. In
    other words, maximizes lambda_from_alloc() through Phi, psi.
    """
    def nls(Phiflat, c, params): # negative sum of lambdas, function to minimize
        ps = params.copy()
        Phi = Phiflat.reshape(params['M'], params['S'])
        psi = (1-Phi.sum(0,keepdims=True)).T
        ps['Phi'] = Phi
        ps['psi'] = psi
        return -lambda_from_alloc(c, ps).sum()
    # constraints (sum of phis should be smaller than 1)
    cons = [
        {'type': 'ineq',
        'fun': lambda v, s=s: 1.0 - np.sum(v.reshape(params['M'], params['S'])[:, s])}
        for s in range(params['S'])
    ]

    bnds = [(0, 1)] * (params['M'] * params['S'])
    res = minimize(nls, Phi.ravel(), args=(c, params),
                  bounds=bnds, constraints=cons, method="SLSQP")

    ps=params.copy()
    Phi_opt = res.x.reshape(params['M'], params['S'])
    psi_opt = (1-Phi_opt.sum(0,keepdims=True)).T

    if res.success and res.status==0:
      return Phi_opt, psi_opt
    else:
      return None, None

# For the dynamical optimized allocation case, integrator goes crazy due to the
# lack of a smooth gradient (choice of preferred nutrient is boolean).
def rk4_step(t, z, dt, params, optAlloc=False):
    k1 = rhs(t,         z,              params, optAlloc=optAlloc)
    k2 = rhs(t + dt/2,  z + dt*k1/2,    params, optAlloc=optAlloc)
    k3 = rhs(t + dt/2,  z + dt*k2/2,    params, optAlloc=optAlloc)
    k4 = rhs(t + dt,    z + dt*k3,      params, optAlloc=optAlloc)

    return z + dt*(k1 + 2*k2 + 2*k3 + k4)/6


def integrate_system_rk4(X0, Y0, m0, P0, R0,
                         D, U, T, L, Fn, Ft, Phi, psi,
                         t_span, dt, optAlloc=False):
    M, _ = X0.shape
    S, _ = Y0.shape

    z0 = pack_state(X0, Y0, m0, P0, R0)

    params = dict(M=M, S=S, D=D, U=U, T=T, L=L,
                  Fn=Fn, Ft=Ft, Phi=Phi, psi=psi,
                  f_n=monod, f_t=monod, enforce_m_cap=False)

    t0, tf = t_span
    n = int(np.ceil((tf - t0)/dt))

    t = np.empty(n+1)
    Z = np.empty((n+1, z0.size))

    t[0] = t0
    Z[0] = z0

    for k in range(n):
        Z[k+1] = rk4_step(t[k], Z[k], dt, params, optAlloc=optAlloc)
        t[k+1] = t[k] + dt

    X, Y, m, P, R = unpack_time_series(Z, M, S, n+1)
    return dict(t=t, X=X, Y=Y, m=m, P=P, R=R)

# Euler
def euler_step(t, z, dt, params, optAlloc=False):
    return z + dt * rhs(t, z, params, optAlloc=optAlloc)

def integrate_system_euler(X0, Y0, m0, P0, R0,
                           D, U, T, L, Fn, Ft, Phi, psi,
                           t_span, dt, optAlloc=False):
    M, _ = X0.shape
    S, _ = Y0.shape

    z0 = pack_state(X0, Y0, m0, P0, R0)

    params = dict(M=M, S=S, D=D, U=U, T=T, L=L,
                  Fn=Fn, Ft=Ft, Phi=Phi, psi=psi,
                  f_n=monod, f_t=monod, enforce_m_cap=False)

    t0, tf = t_span
    n = int(np.ceil((tf - t0) / dt))

    t = np.empty(n + 1)
    Z = np.empty((n + 1, z0.size))

    t[0] = t0
    Z[0] = z0

    for k in range(n):
        Z[k+1] = euler_step(t[k], Z[k], dt, params, optAlloc=optAlloc)
        t[k+1] = t[k] + dt

    X, Y, m, P, R = unpack_time_series(Z, M, S, n+1)
    return dict(t=t, X=X, Y=Y, m=m, P=P, R=R)


# Development of smooth argmax (Bradley–Terry smooth argmax)
def smoothArgMax(M,a):
  M = (M[:, None, :] - M[None, :, :])
  def sigmoid (d, a):
    return 1/(1+np.exp(-a*d))
  choice = sigmoid(M, a)
  softWin = choice.prod(axis=1)*2
  softWin/=softWin.sum(0)
  return softWin


# transt_mutEffs[...,0].shape
def multivariateLogNorm(var, cfactor, shape,rng=None):
  # actually just bivariate
  if rng is None:
    rng = gen_rng()
  # mean forced to 1
  nVar = np.log(1+var) # multivariate normal variance
  c = cfactor * np.sqrt(var[0]*var[1]) # covariance in the log-normal dist
  nC = np.log(1 + c) # covariance in the normal dist
  U_cov=np.array([[nVar[0], nC], [nC, nVar[1]]]); trport_mutEffs = rng.multivariate_normal(-nVar/2, U_cov, shape)
  return np.exp(trport_mutEffs)

def generate_mutation_effects(M, U_corr, T_corr, U_vars, T_vars, nsamples=1, single_molec=True, single_sector=True, single_param=False, rng=None):
	# M : number of uptake transporters
	# U_corr : covariance between mutation effects on Kcat and halfsat in uptake transporters
	# T_corr : covariance between mutation effects on Kcat and halfsat in ribosomes
	# U_vars : mutation effect variances in Kcat, F for uptakers
	# T_vars : mutation effect variances in Kcat, F for ribosomes
	# nsamples : number of mutant strains
	# single_molec : when mutating uptake, mutate just one transporter at a time
	# single_sector : mutate just one sector at a time (uptake or translation)
	# single_param : when mutating uptake, affect only one parameter of the transporter (either Kcat or michaelis const.)

	if rng is None:
		rng = gen_rng()
	trport_mutEffs = multivariateLogNorm(U_vars, U_corr, (M,nsamples), rng)
	ribosm_mutEffs = multivariateLogNorm(T_vars, T_corr,    nsamples , rng)

	if single_molec: # affects only the way mutations occur on transporters
		meffTEST = np.ones((M,nsamples,2))
		if single_param:
			mIX = rng.choice(M,nsamples),np.arange(nsamples),rng.choice(2,nsamples) # index of transporter to mutate
		else:
			mIX = rng.choice(M,nsamples),np.arange(nsamples) # index of transporter to mutate
		meffTEST[*mIX]*=trport_mutEffs[*mIX]
		trport_mutEffs = meffTEST

	if single_sector:
		mutateSector = rng.choice(2, nsamples) # 1/0 whether to mutate uptake sector (otherwise, mutate translation sector)

		pu = mutateSector[I,:,I]
		pt = mutateSector[:,I]

		trport_mutEffs = trport_mutEffs * pu + 1 - pu
		ribosm_mutEffs = ribosm_mutEffs * (1 - pt) + pt
	return(trport_mutEffs, ribosm_mutEffs)

def get_c_p_m_star(params):
  U=params['U']
  Fn=params['Fn']
  T=params['T']
  L=params['L']
  D=params['D']
  S=params['S']
  M=params['M']
  global d_AR, d_dd, eps_P, eps_R
  #d_dd = d_AR/(d_AP*d_CA) #gamma
  #----------------------------------------- compute solution for equilibrium nutrient concentrations
  nuMiki = d_dd*D/(d_AR-1/T)
  cMiki = Fn/(U*D/nuMiki.T-1)
  cstar = np.min(np.concatenate((cMiki,L),axis=1),axis=1)
  #----------------------------------------- compute solution for equilibrium population sizes
  critC= U * monod(cstar[:,I],Fn)
  softWin = smoothArgMax(critC,.1)
  d=(critC*softWin).sum(0)/T.T/d_dd # d=(critC*softWin).sum(0) /T.T*eps_R/eps_P*eps_C/eps_A
  psi = d/(d+1)
  Phi= (1-psi)*softWin
  psi = psi.reshape(S, 1) # Phi.sum(0)+psi.T

  p_star = Phi/((1-psi)+psi*eps_P/eps_R).T
  mstar = np.linalg.solve(U * p_star * monod(cstar[:,I],Fn), L - cstar[:,I])
  #-----------------------------------------
  return cstar, p_star, mstar

from scipy.optimize import lsq_linear

def get_c_p_m_star2(params, get_cMiki=False):
  # less fragile way to obtain steady state solution
  U=params['U']
  Fn=params['Fn']
  Ft=params['Ft']
  T=params['T']
  L=params['L']
  D=params['D']
  S=params['S']
  M=params['M']
  global d_AR, d_dd, eps_P, eps_R
  #d_dd = d_AR/(d_AP*d_CA) #gamma
  #----------------------------------------- compute solutions for equilibrium
  # nutrient concentrations
  nuMiki = d_dd*D/(d_AR-1/T)  # same as (Phi*U*monod(cstar[:,I],Fn)*D).sum(0)/(1-psi.T)
  cMiki = Fn/(U*D/nuMiki.T-1)
  cstar = np.min(np.concatenate((cMiki,L),axis=1),axis=1)

  critC= U * monod(cstar[:,I],Fn)
  softWin = smoothArgMax(critC,.1)
  d=(critC*softWin).sum(0)/T.T/d_dd # d=(critC*softWin).sum(0) /T.T*eps_R/eps_P*eps_C/eps_A
  psi = d/(d+1) # 1/(T*d_AR)
  Phi= (1-psi)*softWin
  psi = psi.reshape(S, 1) # Phi.sum(0)+psi.T

  # compute p star
  p_star = Phi/((1-psi)+psi*eps_P/eps_R).T

  # compute a star
  xi_AP = Phi*d_AP
  xi_AR = psi*d_AR
  b  = (U * xi_AP * monod(cstar[:,I], Fn)).sum(axis=0, keepdims=True).T / (T * xi_AR ) * d_CA
  sms = xi_AR+(1-psi)*d_AP
  a_star = (b -1 + np.sqrt(b **2	+ b  * (4*Ft*sms-2) + 1 )) / (2*sms)

  # compute m star with mild Tikhonov regularization
  A = U * p_star * monod(cstar[:, I], Fn)     # shape (M,S)
  b = (L - cstar[:, I]).ravel()               # shape (M,)

  lam = 1e-10 # nonnegative
  A_aug = np.vstack([A, np.sqrt(lam)*np.eye(A.shape[1])])
  b_aug = np.concatenate([b, np.zeros(A.shape[1])])

  res = lsq_linear(A_aug, b_aug, bounds=(0, np.inf), lsmr_tol='auto')
  mstar = res.x.reshape(-1, 1)

  # mstar = np.linalg.solve(U * p_star * monod(cstar[:,I],Fn), L - cstar[:,I])
  # mstar = np.linalg.solve(A, b)
  #-----------------------------------------
  # !!!! r_star is 1-p_star.sum(0)
  if get_cMiki:
    return cstar, p_star, mstar, a_star, cMiki
  else:
    return cstar, p_star, mstar, a_star

# @markdown new funcs

def stabilize(X0,Y0,m0,P0,R0,params,final_time=2000, resolution= 401, v_thres=1e-20, m_thres=1e-8):
    raise Exception('stabilize() is deprecated. Code still must be migrated.')
    return
    t_eval = np.linspace(0, final_time, resolution)
    X0_ = X0
    Y0_ = Y0
    m0_ = m0
    P0_ = P0
    R0_ = R0
    D = params['D']
    U = params['U']
    T = params['T']
    L = params['L']
    Fn = params['Fn']
    Ft = params['Ft']
    Phi = params['Phi']
    psi = params['psi']
    hasStabilized=False
    attempt=0
    window=int(resolution/2)
    while not hasStabilized:
        attempt+=1
        # print(f'attempt {attempt}')
        # print()
        # print(f'stabilize : {psi.shape}, {Y0_.shape}')
        traj = integrate_system(X0_, Y0_, m0_, P0_, R0_, D, U, T, L, Fn, Ft, Phi, psi,
                             (0.0, t_eval[-1]), t_eval=t_eval, optAlloc=True)
        X0_ = traj['X'][-1]
        Y0_ = traj['Y'][-1]
        m0_ = traj['m'][-1]
        P0_ = traj['P'][-1]
        R0_ = traj['R'][-1]

        # hasStabilized = np.all([np.all(traj[k][-int(resolution/2):].var(0)< 1e-20) for k in list(traj.keys())[1:]])
        diffList = [np.diff(traj[k][-window:], axis=0) for k in list(traj.keys())[1:]]
        hasStabilized = np.all([np.all((d.var(axis=0) < v_thres) & (np.abs(d.mean(axis=0)) < m_thres)) for d in diffList])
        # print (hasStabilized)

    # {k: np.concatenate((traj[k],traj[k]), axis=0) for k in list(traj)[1:]}['P'].shape
    return traj


def introduce_mutant(ps, parent, mEffs):
    U_mutEffs, Fn_mutEffs, T_mutEffs, Ft_mutEffs = mEffs

    #          introduce mutation effects
    mutant_U   = np.c_[ps['U'  ][:,parent]] * U_mutEffs # * 1 + U_mutEffs
    mutant_T   =       ps['T'  ][  parent ] * T_mutEffs # * 1 + T_mutEffs
    mutant_Fn  = np.c_[ps['Fn' ][:,parent]] * Fn_mutEffs    # * 1 + Fn_mutEffs
    mutant_Ft  =       ps['Ft' ][  parent ] * Ft_mutEffs    # * 1 + Ft_mutEffs
    mutant_Phi = np.c_[ps['Phi'][:,parent]]
    mutant_psi =       ps['psi'][  parent ]

    mutantCheck = np.all([np.all(p>0) for p in [mutant_U,mutant_T,mutant_Fn,mutant_Ft,mutant_Phi,mutant_psi]])

    ps['U'  ] = np.hstack([np.c_[ps['U'  ]], mutant_U   ])
    ps['T'  ] = np.vstack([      ps['T'  ],  mutant_T   ])
    ps['Fn' ] = np.hstack([np.c_[ps['Fn' ]], mutant_Fn  ])
    ps['Ft' ] = np.vstack([      ps['Ft' ],  mutant_Ft  ])
    ps['Phi'] = np.hstack([np.c_[ps['Phi']], mutant_Phi ])
    ps['psi'] = np.vstack([      ps['psi'],  mutant_psi ])

    ps['S'] += 1

def introduce_several_mutants(ps, parent, mEffs):
    U_mutEffs, Fn_mutEffs, T_mutEffs, Ft_mutEffs = mEffs
    n_mutants = U_mutEffs.shape[1]

    ps['U']   = np.hstack([ps['U'],  ps['U'] [:, [parent]] * U_mutEffs])
    ps['Fn']  = np.hstack([ps['Fn'], ps['Fn'][:, [parent]] * Fn_mutEffs])
    ps['Phi'] = np.hstack([ps['Phi'], np.repeat(ps['Phi'][:, [parent]], n_mutants, axis=1)])

    for key, effects in [
        ('T',   T_mutEffs),
        ('Ft',  Ft_mutEffs),
        ('psi', np.ones(n_mutants)),
    ]:
        effects = np.asarray(effects).reshape(
            (n_mutants,) + (1,) * (ps[key].ndim - 1)
        )
        mutants = ps[key][parent] * effects
        ps[key] = np.concatenate([ps[key], mutants], axis=0)

    ps['S'] += n_mutants

def formatted_mutEffs(M, U_corr, T_corr, U_vars, T_vars, single_molec, single_sector, single_param, rng, nsamples=1):
    trport_mutEffs, ribosm_mutEffs = generate_mutation_effects(M, U_corr, T_corr, U_vars, T_vars, single_molec=single_molec,single_sector=single_sector, single_param=single_param, rng=rng,nsamples=nsamples)

    U_mutEffs  = trport_mutEffs[...,0]
    Fn_mutEffs = trport_mutEffs[...,1]
    T_mutEffs  = ribosm_mutEffs[...,0]
    Ft_mutEffs = ribosm_mutEffs[...,1]

    return [U_mutEffs ,
            Fn_mutEffs,
            T_mutEffs ,
            Ft_mutEffs]

def extinct_by_ID(ps, exIX):
    #exIX: indices that should go extinct
    ps['U'  ] = np.delete(ps['U'  ], exIX, axis=1)
    ps['T'  ] = np.delete(ps['T'  ], exIX, axis=0)
    ps['Fn' ] = np.delete(ps['Fn' ], exIX, axis=1)
    ps['Ft' ] = np.delete(ps['Ft' ], exIX, axis=0)
    ps['Phi'] = np.delete(ps['Phi'], exIX, axis=1)
    ps['psi'] = np.delete(ps['psi'], exIX, axis=0)
    # ps['S'] -= sum(exIX) # deprecated, when exIX was a mask
    ps['S'] -= len(exIX)

def create_residents(seed, M, S, V=1e15, cP=3e6):
    global eps_A, eps_R, eps_P, eps_C

    rng = gen_rng(seed, True)

    psi  = rng.random((S,1))
    Phi  = rng.random((M,S))
    Phi *= (1-psi).T / Phi.sum(0,keepdims=True)
    v_cell = np.clip(rng.exponential(30,  (S,1)), 1.3, 50) # exponential dist cropped between E. coli and yeast sizes
    v_flow  = rng.random()*V/5 # Fastest turnover: 5 min.

    xi_AR = psi*eps_A/eps_R
    xi_AP = Phi*eps_A/eps_P
    xi_CA = eps_C/eps_A
    sms = xi_AP.sum(0,keepdims=True).T+xi_AR

    T = (rng.random((S,1))+1)/xi_AR # condition : T*xi_AR>1
    Fn = rng.random((M, S)) *4
    Ft = rng.random((S, 1)) *4
    Ft=Ft*0+7/cP # parameterization from lit.

    rlstar = xi_AR/sms
    plstar = xi_AP/sms.T
    Ystar = Ft/(T*xi_AR-1)

    U = (rng.random((M,S))+1) * ((Ystar+1)/(xi_CA*sms)).T # condition : U.sum(0) > ((Ystar+1)/(xi_CA*sms)).T

    L = rng.random((M, 1)); L/L.sum()/rng.random()*(T*(Ystar+1)).sum()/xi_CA # condition : L.sum()<(T*(Ystar+1)).sum()/xi_CA #!!!!!!!!!!!!!!!!!!!!!
    D = v_flow/V

    X0 = rng.random((M, 1)) * L # X < L
    Y0 = rng.random((S, 1)) * (U.sum(0)*sms.T*xi_CA-1).T # Y < (U.sum(0)*sms.T*xi_CA-1).T
    m0 = rng.random((S, 1)); m0/=m0.sum(0)/rng.random() # m.sum() < 1

    R0=psi
    P0=Phi

    params = dict(M=M, S=S, D=D, U=U, T=T, L=L, Fn=Fn, Ft=Ft,
                  Phi=Phi, psi=psi, f_n=monod, f_t=monod,
                  enforce_m_cap=False)

    init_state = (X0,Y0,m0,P0,R0)

    return params, init_state


# def copy_parent_initial_conditions(parent,Y0,m0,P0,R0,initPop=1e-11):
#     # initPop: float or zero/False for copying parent popsize
#     Y0_ = Y0
#     m0_ = m0
#     P0_ = P0
#     R0_ = R0
#     if not initPop:
#         initPop = m0_[parent  ] # in biologically plausible circumstances, this would be a very small number (mutant pop)
#     Y0_ = np.vstack([Y0_,      [Y0_[parent  ]]])
#     m0_ = np.vstack([m0_,      [initPop      ]]) # add mutant population size
#     P0_ = np.hstack([P0_, np.c_[P0_[:,parent]]])
#     R0_ = np.vstack([R0_,      [R0_[  parent]]])

#     return Y0_,m0_,P0_,R0_

def copy_parent_initial_conditions(parent, Y0, m0, P0, R0, initPop=1e-11, n=1):
    # initPop: float or zero/False for copying parent popsize
    mutant_m0 = (
        np.full((n,) + m0.shape[1:], initPop)
        if initPop
        else np.repeat(m0[[parent]], n, axis=0)# in biologically plausible circumstances, this would be a very small number (mutant pop)
    )

    Y0_ = np.concatenate([Y0, np.repeat(Y0[  [parent]], n, axis=0)], axis=0)
    m0_ = np.concatenate([m0,              mutant_m0],               axis=0)
    P0_ = np.concatenate([P0, np.repeat(P0[:,[parent]], n, axis=1)], axis=1)
    R0_ = np.concatenate([R0, np.repeat(R0[  [parent]], n, axis=0)], axis=0)

    return Y0_, m0_, P0_, R0_

def plot_muller(frequencies, colors=None, labels=None, time=None, ax=None):
    """
    Plot a Muller diagram from population data (raw counts or frequencies).
    The y-axis shows total population size (sum of all lineages at each time).
    """
    m, n = frequencies.shape
    if time is None:
        time = np.arange(n)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    if colors is None:
        colors = plt.cm.tab20(np.linspace(0, 1, m))
    if labels is None:
        labels = [f"Lineage {i}" for i in range(m)]

    # Compute cumulative sum for stacking (no normalization)
    stacked = np.cumsum(frequencies, axis=0)
    bottom = np.vstack((np.zeros((1, n)), stacked[:-1, :]))

    # Plot each lineage as a stacked area
    for i in range(m):
        ax.fill_between(time, bottom[i, :], stacked[i, :],
                        color=colors[i], label=labels[i], alpha=0.8,linewidth=0,)

    ax.set_xlabel("Time")
    ax.set_ylabel("Population Size")
    ax.set_xlim(time.min(), time.max())
    ax.set_title("Muller Plot (Variable Total Population)")
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    return ax


def hierarchical_node_order(G, root):
    """
    Reorder nodes in a genealogy tree using hierarchical traversal.

    Parameters:
    -----------
    G : nx.DiGraph
        Directed graph representing genealogy (edges from parent to child)
    root : node
        The root node (e.g., earliest ancestor)

    Returns:
    --------
    ordered_nodes : list
        List of nodes ordered by hierarchical clustering (subtrees grouped together)
    """
    ordered_nodes = []

    def dfs_subtree(node):
        # Add node first (pre-order)
        ordered_nodes.append(node)
        # Recursively add all descendants
        for child in sorted(G.successors(node)):  # Sort children for deterministic order
            dfs_subtree(child)

    dfs_subtree(root)
    return ordered_nodes


def assign_hierarchical_colors(
    G,
    root=None,
    hue_start=30,
    hue_end=390,
    hue_fraction=0.75,  # Use more of the hue circle
    chroma=60,
    luminance=70,
    chroma_slope=5,
    luminance_slope=-10,
    depth=0,
    max_depth=None
):
    """
    Assign colors using HCL space with better branch distinction.
    """
    if root is None:
        root = next(iter(G.nodes))
    if max_depth is None:
        max_depth = max(nx.shortest_path_length(G, source=root).values())

    children = list(G.successors(root)) if G.is_directed() else list(G.neighbors(root))

    # Spread hues across available range
    n_children = len(children)
    if n_children == 0:
        h = hue_start
    else:
        step = (hue_end - hue_start) / n_children
        h = hue_start + step / 2  # Center hue

    # Vary HCL values by depth
    c = chroma + chroma_slope * depth
    l = luminance + luminance_slope * depth

    # Convert HCL to RGB (approximate)
    from scipy.spatial.transform import Rotation as R
    import numpy as np

    # Simplified HCL to RGB approximation
    try:
        import colorspace
        color = colorspace.HCL(h, c, l).colors()[0]
    except ImportError:
        # Fallback: use HSV with controlled saturation/value
        h_norm = h % 360 / 360
        s = max(min(c / 100, 1), 0)
        v = max(min(l / 100, 1), 0)
        r, g, b = colorsys.hsv_to_rgb(h_norm, s, v)
        color = (r, g, b)

    color_map = {root: color}

    # Recursively assign to children with subdivided hue ranges
    for i, child in enumerate(children):
        child_start = hue_start + i * step
        child_end = child_start + step * hue_fraction  # Leave gaps for distinction
        child_colors = assign_hierarchical_colors(
            G, child, child_start, child_end, hue_fraction,
            chroma, luminance, chroma_slope, luminance_slope,
            depth + 1, max_depth
        )
        color_map.update(child_colors)

    return color_map


def merge_trajectories(tl, exIX_list, fill_value=0.0):
    nt = tl[0].shape[0]

    # global species IDs alive after each realization
    alive = []
    next_species_id = 0

    traj_blocks = []

    for i, arr in enumerate(tl):
        ns = arr.shape[1]

        # initialize first realization
        if i == 0:
            alive = list(range(ns))
            next_species_id = ns

        # global matrix for this realization
        block = np.full((next_species_id, nt), fill_value)

        # insert trajectories
        vals = arr[..., 0].T  # (nspecies, nt)
        for local_ix, global_ix in enumerate(alive):
            block[global_ix] = vals[local_ix]

        traj_blocks.append(block)

        # update alive species for next realization
        if i < len(exIX_list):
            extinct = set(exIX_list[i])

            # remove extinct local indices
            alive = [
                gid for j, gid in enumerate(alive)
                if j not in extinct
            ]

            # detect newly added species
            n_new = tl[i + 1].shape[1] - len(alive) if i + 1 < len(tl) else 0

            for _ in range(n_new):
                alive.append(next_species_id)
                next_species_id += 1

    # total species ever existed
    N = next_species_id

    # pad old blocks to final N
    padded = []
    for block in traj_blocks:
        pad = np.full((N, nt), fill_value)
        pad[:block.shape[0]] = block
        padded.append(pad)

    return np.hstack(padded)

def reconstruct_history(parents, deaths):
    # Start with a single founder (id 0)
    current_ids = [0]  # label -> birth id

    # genealogy: child birth id -> parent birth id
    parent_of = {0: None}

    history = []

    for t, (p, d) in enumerate(zip(parents, deaths), start=1):

        # ---- deaths before reproduction ----
        if len(d) > 0:
            alive_mask = np.ones(len(current_ids), dtype=bool)
            alive_mask[d] = False
            current_ids = list(np.array(current_ids)[alive_mask])

        # ---- parent reproduces ----
        parent_identity = int(current_ids[p])

        # child gets birth id = timestep
        child_id = t
        parent_of[child_id] = parent_identity

        # append child at end
        current_ids.append(child_id)

        history.append({
            "step": t,
            "deaths": d.tolist(),
            "chosen_label": p,
            "actual_parent": parent_identity,
            "population": np.array(current_ids.copy())
        })

    return history, parent_of, current_ids


from IPython.display import Audio,display

sample_rate = 22050  # Hz
frequency1 = 512  # Hz
frequency2 = 1024  # Hz
play_time = 0.25  # seconds

# Generate time array
t = np.arange(0, play_time, 1/sample_rate)

# Generate sound waves
wave1 = np.sin(2 * np.pi * frequency1 * t) * np.exp(-t/0.7)
wave2 = np.sin(2 * np.pi * frequency2 * t) * np.exp(-t/0.7)

# Create silence array
silence = np.zeros(int(sample_rate * play_time))

# Concatenate sound waves and silence
sound = np.concatenate((silence, wave1, wave2))
def notify():
    display(Audio(sound, rate=sample_rate,autoplay=True))

notify()

def get_growth_rates(params, c):
    # IMPORTANT: gives growth rates under external concentrations but does not account for ecology (i.e., resource competition)
    # Each value is calculated as if the strain was isolated.
    global d_dd,d_AR
    _,_,_,U,T,_,Fn,_,_,_,_,_,_ = params.values()
    return T.T*d_AR / (1+d_dd*T.T/U*(1+Fn/c))-1

def merge_pops(ps1, ps2):
    merged = ps1.copy()

    for key in ('U', 'Fn', 'Phi'):
        merged[key] = np.concatenate([ps1[key], ps2[key]], axis=1)

    for key in ('T', 'Ft', 'psi'):
        merged[key] = np.concatenate([ps1[key], ps2[key]], axis=0)

    merged['S'] = ps1['S'] + ps2['S']
    return merged

def integrate_system_euler(X0, Y0, m0, P0, R0,
                           D, U, T, L, Fn, Ft, Phi, psi,
                           t_span, t_eval=None, optAlloc=False, *, dt):
    t0, tf = map(float, t_span)
    if not np.isfinite(dt) or dt <= 0 or tf <= t0:
        raise ValueError("Require dt > 0 and a forward t_span.")

    times = np.asarray([t0, tf] if t_eval is None else t_eval, dtype=float)
    if (times.ndim != 1 or times.size == 0 or
        not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0) or
        times[0] < t0 or times[-1] > tf):
        raise ValueError("t_eval must be increasing and within t_span.")

    M, S = X0.shape[0], Y0.shape[0]
    z = np.array(pack_state(X0, Y0, m0, P0, R0), dtype=float, copy=True)

    params = dict(M=M, S=S, D=D, U=U, T=T, L=L, Fn=Fn, Ft=Ft,
                  Phi=Phi, psi=psi, f_n=monod, f_t=monod,
                  enforce_m_cap=False)

    history = np.empty((times.size, z.size))
    delta = np.empty_like(z)
    t = t0

    for i, target in enumerate(times):
        # Subdivide each output interval into steps no larger than dt.
        steps = int(np.ceil((target - t) / dt))
        if steps:
            h = (target - t) / steps
            for j in range(steps):
                np.multiply(rhs(t + j*h, z, params, optAlloc=optAlloc),
                            h, out=delta)
                z += delta

        history[i] = z
        t = target

    X, Y, m, P, R = unpack_time_series(history, M, S, times.size)
    return dict(t=times, X=X, Y=Y, m=m, P=P, R=R)

def reconstruct_history_bulk(parent_list_bulk, exIX_list):
    current_ids = [0]
    parent_of = {0: None}
    history = []

    next_birth_id = 1

    for t, (parents, deaths) in enumerate(zip(parent_list_bulk, exIX_list), start=1):

        # ---- deaths before reproduction ----
        if len(deaths) > 0:
            alive_mask = np.ones(len(current_ids), dtype=bool)
            alive_mask[deaths] = False
            current_ids = list(np.array(current_ids)[alive_mask])

        # ---- resolve all parents ----
        actual_parents = [int(current_ids[p]) for p in parents]

        # ---- create children with unique birth IDs ----
        child_ids = list(range(next_birth_id, next_birth_id + len(parents)))

        for child_id, parent_id in zip(child_ids, actual_parents):
            parent_of[child_id] = parent_id

        current_ids.extend(child_ids)
        next_birth_id += len(parents)

        history.append({
            "step": t,
            "deaths": deaths.tolist(),
            "chosen_labels": list(parents),
            "actual_parents": actual_parents,
            "child_ids": child_ids,
            "population": np.array(current_ids.copy())
        })

    return history, parent_of, current_ids


def prune_length1_branches(G,return_remove=False):
    H = G.to_undirected()
    remove = [
        n for n in H.nodes
        if H.degree(n) == 1
        and H.degree(next(iter(H.neighbors(n)))) > 2
    ]
    G.remove_nodes_from(remove)
    if return_remove:
        return G, remove
    else:
        return G


#molar masses
eps_A = 112
eps_P = 33600
eps_R = 835408
eps_C = 180.16
#proteome size
cP    = 3e6
popthres = 1e-12 # minimum population size stochasticity threshold

d_AR = eps_A/eps_R
d_AP = eps_A/eps_P
d_CA = eps_C/eps_A
d_dd = d_AR/(d_AP*d_CA) #gamma


M, S = 2,2
final_time=30000.0
# seed=np.random.randint(1000)
seed = 607
# 2,2 : 577
# 1,1 : 48
# 3,2 : 31
# 2,1 : 919
optAlloc=False
smoothChoice=True
resolution=501 # 401 is good if opt_alloc=False
