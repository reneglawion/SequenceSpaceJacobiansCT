"""
1. Normalize Y = 1 , calibrate r and B; G. Set T = G + rB.
2. Use Code from Hugget model, where now e_{it}*(Y - T)
3. Choose discount rate to match A=B
4. Market clearing G + C = Y holds by Walras law 

Based on the Matlab codes of Achdou et al. (2022) 
and the Python codes of Auclert et al. (2021)
"""

# %% Import packages
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy import linalg
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib import cm

# Settings
maxIterations = 100
Delta=1000
dt=1/100

G = .2                                  # Government spending
B = .8                                  # Government debt
Y = 1.                                  # output calibrated to 1

T = 200                                 # length of the IRF
rho_G = 0.8                             # persistence of the shock

#--------------------------------------------------
#PARAMETERS
ga = 2                                  # CRRA utility with parameter gamma
w = 1.                                  # mean O-U process (in levels). This parameter has to be adjusted to ensure that the mean of z (truncated gaussian) is 1.
r = 0.03                                # interest rate
Corr = .9                       
the = -np.log(Corr)                     # mean reversion parameter in O-U process
sig2 = 0.6                              # sigma^2 O-U process
rho = 0.05                              # discount rate beta in discrete time

relax = 0.999                           # relaxation parameter (see Achdou et al. 2022)
zmin = .5                               # range z
zmax = 1.5
amin = 0.                               # borrowing constraint     
amax = 40                               # range a
I = 200  
J = 10

# simulation parameters
maxit  = 100                            # maximum number of iterations in the HJB loop
maxitK = 100                            # maximum number of iterations in the K loop
crit = 10**(-6)                         # criterion HJB loop
critK = 1e-5                            # criterion K loop

# Utility function
def U(cc):
  return cc**(1-ga)/(1-ga)

def dUinv(vv):
  return vv**(-1/ga)

# Can be used to compute other mean reversion processes (e.g. CIR)
def beta(rr):
    return 1.

#--------------------------------------------------
# Grid 
a = np.linspace(amin,amax,num=I, endpoint=True)     # wealth vector
da = (amax-amin)/(I-1)  
da2 = da**2    
z = np.linspace(zmin,zmax,num=J, endpoint=True)     # productivity vector
dz = (zmax-zmin)/(J-1)
dz2 = dz**2
aa = np.reshape(a,(I,1))*np.ones((1,J)) 
zz = np.ones((I,1))*z

mu = the*(w - z)                                    # Drift from Itô's Lemma (see Matlab codes of Achdou et al. 2022)
s2 = np.squeeze(sig2*np.ones((1,J)))

Id = sparse.eye(I*J)

def fiscal(B, r, G, Y):
    T = r * B + G                                   # total tax burden in CT
    Z = Y - T                                       # after tax income
    deficit = G - T
    return T, Z, deficit

def getAswitch():
    """ get the matrix Aswitch which summarizes the transition of z

    Parameters
    ----------
        nothing, but uses global variables z, zz, dz, dz2, mu, s2

    Returns
    -------
    Aswitch : sparse matrix 
        matrix summarizing evolution of z
    """
    Aswitch = sparse.lil_matrix((I*J, I*J))
    chi = -np.where(mu<0,mu,0)/dz + s2/(2*dz2)
    zeta = np.where(mu>0,mu,0)/dz + s2/(2*dz2)
    yy = -zeta-chi 

    # centdiag  
    B_diag_0=np.tile(chi[0]+yy[0],(I,1))
    for j in range(1, J-1):
        B_diag_0 = np.append(B_diag_0, np.tile(yy[j], (I,1)))
    B_diag_0 = np.append(B_diag_0, np.tile(yy[-1] + zeta[-1], (I,1)))

    # updiag 
    B_diag_pM=[] #np.zeros((I,1))
    for j in range(J):
        B_diag_pM = np.append(B_diag_pM, np.tile(zeta[j], (I,1)))

    # lowdiag 
    B_diag_mM=np.tile(chi[1],(I,1))
    for j in range(2,J):
        B_diag_mM = np.append(B_diag_mM, np.tile(chi[j], (I,1)))

    # Add up the upper, center, and lower diagonal into a sparse matrix
    Aswitch.setdiag(B_diag_0) 
    Aswitch.setdiag(B_diag_pM,k=I) 
    Aswitch.setdiag(B_diag_mM,k=-I) 

    return Aswitch

def backwardIteration(V, Aswitch, r, rho, Z):
    """ Performs a backward iteration of the HJB equation

    Parameters
    ----------
    V : dense matrix
        value function in time t+1 (or current guess for SS iteration)
    Aswitch: sparse matrix
        matrix summarizing evolution of z
    r : float
        interest rate
    rho : float
        discount rate
    Z : float
        total after tax income

    Returns
    -------
    V : dense matrix
        value function at time t
    Va_Upwind : dense matrix
        costate at time t
    c : dense matrix
        consumption policy at time t
    s : dense matrix
        savings policy at time t 
    A : sparse matrix
        transition matrix at time t       
    """

    sA = sparse.lil_matrix((I*J, I*J))

    # Finite difference approximation of the partial derivatives
    Vaf = np.zeros((I,J))
    Vab = np.zeros((I,J))

    # forward difference 
    Vaf[0:-1,:] = (V[1:,:] - V[0:-1,:])/da 
    Vaf[-1,:] = (Z*z+ r*amax)**(-ga) 
    # backward difference
    Vab[1:, :] = (V[1:,:] - V[0:-1,:])/da  
    Vab[0, :] = (Z*z+ r*amin)**(-ga) #  state constraint boundary condition

    # indicator whether value function is concave (problems arise if this is not the case)
    I_concave = Vab > Vaf        
    # consumption and savings with forward difference
    cf = Vaf**(-1/ga)
    sf = Z*zz + aa*r - cf
    # consumption and savings with backward difference
    cb = Vab**(-1/ga)
    sb = Z*zz + aa*r - cb
    # consumption and derivative of value function at steady state
    c0 = Z*zz + aa*r
    Va0 = c0**(-ga)

    # dV_upwind makes a choice of forward or backward differences based on
    # the sign of the drift
    If = sf > 0 #positive drift --> forward difference
    Ib = sb < 0#negative drift --> backward difference
    I0 = (1-If-Ib) #at steady state

    Va_Upwind = Vaf*If + Vab*Ib + Va0*I0 #important to include third term # check‚‚

    c = Va_Upwind**(-1/ga)
    s = Z*zz + aa*r - c 
    u = c**(1-ga)/(1-ga) 

    # construct the matrix A
    XX = - np.where(sb<0,sb,0)/da
    ZZ = np.where(sf>0,sf,0)/da  
    YY = -XX-ZZ    

    A_diag_0=np.reshape(np.transpose(YY),(I*J)) # the transpose is really important here 
                                                # (in contrast to the Matlab implentation
                                                # by Achdou et al (2022))

    # updiag 
    A_diag_p1 = []  # Matlab needs a zero here - python does not 
    for j in range(J):
        A_diag_p1 = np.append(A_diag_p1, ZZ[0:-1,j]) 
        A_diag_p1 = np.append(A_diag_p1, 0)

    # lowdiag 
    A_diag_m1 =  XX[1:, 0]
    for j in range(1, J):
        A_diag_m1 = np.append(A_diag_m1, 0)
        A_diag_m1 = np.append(A_diag_m1, XX[1:,j])

    sA.setdiag(A_diag_0)         
    sA.setdiag(A_diag_p1,k=1)   
    sA.setdiag(A_diag_m1,k=-1)  

    A = Aswitch + sA

    u = U(np.reshape(np.transpose(c),I*J))
    v_stacked = np.reshape(np.transpose(V),I*J)
        
    # V = (sparse.linalg.inv((1/Delta+rho)*Id-A)).dot(u+v_stacked/Delta) # to check, but slow
    BB = (1/Delta+rho)*Id-A
    b = u+v_stacked/Delta
    V = spsolve(BB, b)

    V=np.reshape(V,(J,I)).T

    return V, Va_Upwind, c, s, A

def policy_ss(r, Z, rho):
    """
    Computes the steady state policy functions

    Parameters
    ----------
    r : float
        interest rate
    Z : float
        total after tax income
    rho : float
        discount rate

    Returns
    -------
    v_ss : dense matrix
        value function in steady state
    Va_ss : dense matrix 
        derivative of value function in steady state (costate)
    c_ss : dense matrix
        consumption in steady state 
    s_ss : dense matrix
        savings in steady state
    A_ss : sparse matrix
        matrix summarizing evolution of z
    """

    Aswitch = getAswitch()

    # initial guess for value function
    v = np.zeros((I,J)) 
    for i in range(I):
        for j in range(J):
            v[i, j] = U(Z*z[j]+r*a[i])/rho

    # Steady state iteration until convergence
    for i in range(maxit):

        vold = v.copy()

        v, Va, c, s, A = backwardIteration(vold, Aswitch, r, rho, Z)

        if np.max(np.max(np.abs(v-vold))) < crit:
            break

    return v, Va, c, s, A, Aswitch

def distribution_ss(A):
    """
    Computes the steady state distribution of agents

    input:  A = infenitesimal operator of the HJB equation (operator curlyA in the paper)

    output: pdf = steady state distribution of agents (g_ss in the paper)
    """

    AT = A.T   

    pdf = np.zeros((I,J))
    [eigenvalue, eigenvec] = sparse.linalg.eigs(
            AT, k=1, sigma=0., return_eigenvectors=True)
        
    #print("Eigenvalue = {}".format(eigenvalue[0]))
    for i in range(I):
        for j in range(J):
            pdf[i, j] = abs(eigenvec[j*I+i][0])
    
    # normalize pdf to sum to 1
    M=np.zeros(J)
    for j in range(J):
        M[j]=da*np.sum(pdf[:,j])

    Mass=dz*np.sum(M)
    pdf=pdf/Mass

    return pdf

def getAggregateConsumptionAndSavings(g, c, s):
    """
    Computes aggregate consumption and savings

    input:  g =  distribution of agents
            c = consumption of agents on the grid
            s = savings of agents on the grid

    output: C = aggregate consumption
            S = aggregate savings
            Assets = aggregate assets
    """
    
    S0 = np.zeros(J)
    C0 = np.zeros(J)
    Assets0 = np.zeros(J)

    for j in range(J):

        S0[j] = da*np.sum(g[:,j]*s[:,j]) 
        C0[j] = da*np.sum(g[:,j]*c[:,j])
        Assets0[j] = da*np.sum(g[:,j]*a[:])
        
    S=dz*np.sum(S0)
    C=dz*np.sum(C0)
    Assets=dz*np.sum(Assets0)
    
    return C, S, Assets

def forwardIteration(g_t, A, dt):
    """
    Computes the forward iteration for the FPE 

    Parameters
    ----------
    g_t : dense matrix
        distribution at time t
    A : sparse matrix
        Generator of the HJB equation (operator curlyA in the paper)
    dt : float
        time step size

    Returns
    -------
    g_tplus1 : dense matrix
        distribution at time t+dt
    """
    I, J = g_t.shape
    g_t = np.reshape(g_t.T, (I*J, 1))
    g_tplusdt = spsolve(Id - A.T*dt, g_t) # solve linear system with A^T
    g_tplusdt = np.reshape(g_tplusdt,(J,I)).T
    return g_tplusdt

def expectation_iteration(X_t, A, dt):
    """
    Computes the expectation iteration 

    Parameters
    ----------
    X_t : dense matrix
        function/policy at time t
    A : sparse matrix
        Generator of the HJB equation (operator curlyA in the paper)
    dt : float
        time step size
   
    Returns
    -------
    X_tplusdt : dense matrix
        function/policy at time t+dt
    """
    # This is also expectation policy
    X_t = np.reshape(X_t.T, (I*J, 1))
    X_tplusdt = X_t + A*X_t*dt
    X_tplusdt = np.reshape(X_tplusdt, (J, I)).T
    return X_tplusdt

def expectationVectors(X, A, dt, T):
    """
    Computes the expectation vectors for a given function/policy

    Parameters
    ----------
    X : dense matrix
        function/policy at time t
    A : sparse matrix
        Generator of the HJB equation (operator curlyA in the paper)
    dt : float
        time step size
    T : int
        number of time steps
    
    Returns
    -------
    curlyE : sequence of dense matrices
        Expectation vectors for a given function/policy
    """

    curlyE = np.empty((T, ) + X.shape)
    curlyE[0] = X
    
    # recursively apply law of iterated expectations
    for j in range(1, T):
        curlyE[j] = expectation_iteration(curlyE[j-1], A, dt)
        
    return curlyE


def steady_state(r,rho, B=0, G=0, Y=1.):
    """
    Computes the steady state distribution of agents

    Parameters
    ----------
    r : float
        interest rate
    rho : float
        discount rate
    B : float
        government debt, default 0
    G : float
        government expenditures, default 0
    Y : float
        Output, default 1
    
    Returns
    -------
    d : dictionary
        steady state values of all variables with keys:
        'g' : steady state distribution of agents
        'V' : steady state value function
        'Va' : steady state derivative of value function (costate)
        'A' : steady state generator of the HJB equation
        'Aswitch' : Matrix summarizing the evolution of z
        'a' : grid for a
        'c' : steady state consumption
        's' : steady state savings
        'C' : steady state aggregate consumption
        'S' : steady state aggregate savings
        'Assets' : steady state aggregate assets
        'r' : steady state interest rate
        'w' : steady state mean wage
        'rho' : steady state discount rate
        'B' : steady state amount of government debt
        'G' : steady state government expenditures
        'asset_mkt_error' : steady state asset market clearing error
        'goods_mkt_error' : steady state goods market clearing error
        'deficit' : steady state government deficit
        'T' : steady state government taxes
        'Z' : steady state level of Z
        'Y' : steady state level of Output
        'Zs' : possible path of shocks
    """

    T, Z, deficit = fiscal(B, r, G, Y)
    V, Va, c, s, A, Aswitch = policy_ss(r, Z, rho)
    g = distribution_ss(A)

    C, S, Assets = getAggregateConsumptionAndSavings(g, c, s)

    
    return dict(g=g, Va=Va, V=V, A=A, Aswitch=Aswitch,
                a=a, c=c, s=s,
                S=S, C=C, Assets = Assets,
                r=r, w=w, rho=rho, B=B, G=G,
                asset_mkt_error = np.abs(Assets-B),
                goods_mkt_error = np.abs(Y-C-G),
                deficit = deficit, T=T, Z=Z,
                Y = 1.,
                Zs=1.)

SS = steady_state(r=r, rho=rho, B=B, G=G)

# %% Calibration GE
def SS_rho(rho_guess):
    SS = steady_state(r=r, rho=np.squeeze(rho_guess), B=B, G=G)
    return SS["asset_mkt_error"]

rho_calib = optimize.fsolve(
                SS_rho,
                0.05)
rho_calib = np.squeeze(rho_calib)
print("rho_calib = {}".format(rho_calib))
SS = steady_state(r=r, rho=rho_calib, B=B, G=G)
print("Asset Market error = {}".format(SS["asset_mkt_error"]))
print("Goods Market error = {}".format(SS["goods_mkt_error"]))
print("Total savings = {}".format(SS["S"]))
print("Consumption = {}".format(SS["C"]))

# Plot the value function v and the stationary distribution pdf in one plot beside each other as surface plots
# below that plot the consumption policy function c and besides that the savings policy function s 
fig = plt.figure(figsize=(12, 12))
fig.suptitle("General Equilibrium Steady State", fontsize=16)
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax4 = fig.add_subplot(2, 2, 4, projection='3d')

# plot value function
X, Y = np.meshgrid(a, z)
ax1.plot_surface(X, Y, SS['V'].T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax1.set_xlabel('a')
ax1.set_ylabel('z')
ax1.set_zlabel('v')
ax1.set_title('Value function')

# plot stationary distribution
ax2.plot_surface(X, Y, SS['g'].T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax2.set_xlabel('a')
ax2.set_ylabel('z')
ax2.set_zlabel('pdf')
ax2.set_title('Stationary distribution')

# plot consumption function
ax3.plot_surface(X, Y, SS['c'].T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax3.set_xlabel('a')
ax3.set_ylabel('z')
ax3.set_zlabel('c')
ax3.set_title('Consumption function')

# plot savings function
ax4.plot_surface(X, Y, SS['s'].T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax4.set_xlabel('a')
ax4.set_ylabel('z')
ax4.set_zlabel('s')
ax4.set_title('Savings function')

# safve the figure with the title as name as a pdf file
fig.savefig('GeneralEquilibriumSteadyState.pdf')
plt.show()

# %% MPCs 
def compute_weighted_mpc(c):
    """Approximate mpc out of wealth, with symmetric differences where possible, exactly setting mpc=1 for constrained agents."""
    mpc = np.empty_like(c)
    post_return = (1 + r) * aa
    mpc[1:-1, :] = (c[2:, :] - c[0:-2, :]) / (post_return[2:] - post_return[:-2])
    mpc[0, :] = (c[1, :] - c[0, :]) / (post_return[1] - post_return[0])
    mpc[-1, :] = (c[-1, :] - c[-2, :]) / (post_return[-1] - post_return[-2])
    mpc[a == a[0]] = 1
    #mpc = mpc * z[np.newaxis, :]

    MPC0 = np.zeros(J)

    for j in range(J):

        MPC0[j] = da*np.sum(SS['g'][:,j]*mpc[:,j]) 
        
    MPC=dz*np.sum(MPC0)

    return mpc, MPC

c = SS['c']
mpcs, MPC = compute_weighted_mpc(SS['c'])
print("MPCs = {}".format(MPC))

# plot the mpc out of wealth as a surface plot 
fig = plt.figure(figsize=(12, 12))
fig.suptitle("MPC out of wealth", fontsize=16)
ax1 = fig.add_subplot(1, 1, 1, projection='3d')

# plot mpc out of wealth
AA, ZZ = np.meshgrid(a, z)
ax1.plot_surface(AA, ZZ, mpcs.T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax1.set_xlabel('a')
ax1.set_ylabel('z')
ax1.set_zlabel('mpc')

# safve the figure with the title as name as a pdf file
fig.savefig('MPC.pdf')
plt.show()

# %% match market clearing and MPC=0.25

target_MPC = .25

def SS_helper(current_sol):
    rho_guess = current_sol[0]
    B_guess = current_sol[1]

    SS = steady_state(r=r, rho=rho_guess, B=B_guess, G=G)

    _, MPC = compute_weighted_mpc(SS['c'])

    error = np.empty(2)
    error[0] = SS['asset_mkt_error']
    error[1] = (MPC - target_MPC)**2
    return error

initial_guess = np.array([0.08, 0.5])
params_calib = optimize.fsolve(
                SS_helper,
                initial_guess)
rho_calib = np.squeeze(params_calib[0])
B_calib = np.squeeze(params_calib[1])
print("rho_calib = {}".format(rho_calib))
print("B_calib = {}".format(B_calib))
SS_calib= steady_state(r=r, rho=rho_calib, B=B_calib, G=G)
print("Asset Market error = {}".format(SS_calib["asset_mkt_error"]))
print("Goods Market error = {}".format(SS_calib["goods_mkt_error"]))
print("Total savings = {}".format(SS_calib["S"]))
print("Consumption = {}".format(SS_calib["C"]))
mpcs, MPC = compute_weighted_mpc(SS_calib['c'])
print("MPCs = {}".format(MPC))

SS = SS_calib

# %% Fake News Algorithm to get Jacobian J

def J_from_F(F):
    """
    Computes the Jacobian J from  curlyF_{t,s} dx = curlyE_{t-1} dg_1^s

    Parameters
    ----------
    F : sequence of dense matrices
        curlyF_{t,s} dx = curlyE_{t-1} dg_1^s
    
    Returns
    -------
    J : dense matrix
        Jacobian J
    """
    J = F.copy()
    for t in range(1, F.shape[0]):
        J[1:, t] += J[:-1, t-1]
    return J

def step1_backward(ss, shock, T, h):
    """
    Computes the first step of the backward algorithm

    Parameters
    ----------
    ss : dictionary
        steady state values of all variables
    shock : array
        shock to the steady state
    T : int
        number of time steps
    h : float
        step size for the derivative

    Returns
    -------
    curlyY : sequence of dense matrices
        Outputs/Outcomes in each time step t after shock
    curlyg : sequence of dense matrices
        Disrtributions in each time step t after shock
    """

    # preliminaries: g_1 with no shock, ss inputs to backward_iteration
    g1_noshock = forwardIteration(ss["g"], ss["A"], dt)
    ss_inputs = {k: ss[k] for k in ('Aswitch', 'r', 'rho', 'Z')} 
    
    # allocate space for results
    curlyY = {'C': np.empty(T), 'S': np.empty(T)}
    curlyg = np.empty((T,) + ss['g'].shape)
    
    V = ss['V'] # initialize with steady-state value function
    # backward iterate
    for s in range(T):
        if s == 0:
            # at horizon of s=0, 'shock' actually hits, override ss_inputs with shock
            shocked_inputs = {k: ss[k] + h*shock[k][s] for k in shock}
            V, _, c, savings, Atmp = backwardIteration(**{'V' : V, **ss_inputs, **shocked_inputs})
        else:
            # now the only effect is anticipation, so it's just Va being different
            V, _, c, savings, Atmp = backwardIteration(**{**ss_inputs, 'V': V})

        # aggregate effects on A and C
        S0, C0 = np.zeros(J), np.zeros(J)
        
        for j in range(J):
            S0[j] = da*np.sum(SS["g"][:,j]*(savings[:,j]-SS["s"][:,j]))
            C0[j] = da*np.sum(SS["g"][:,j]*(c[:,j]-SS["c"][:,j]))
        
        S=dz*np.sum(S0)
        C=dz*np.sum(C0)

        curlyY['S'][s] = S/h
        curlyY['C'][s] = C/h
        
        # what is effect on one-period-ahead distribution
        curlyg[s] = (forwardIteration(SS["g"], SS['A'], dt) - g1_noshock) / h
        
    return curlyY, curlyg


def jacobian(ss, shocks, T):
    """
    Computes the Jacobian J in response to a shock to the steady state

    Parameters
    ----------
    ss : dictionary
        steady state values of all variables
    shock : array
        shock to the steady state
    T : int
        number of time steps
    
    Returns
    -------
    J : dense matrix
        Jacobian J
    """
    # step 1 for all shocks i, allocate to curlyY[o][i] and curlyg[i]
    curlyY = {'S': {}, 'C': {}}
    curlyg = {}

    for i, shock in shocks.items():
        curlyYi, curlyg[i] = step1_backward(ss, shock, T, 1E-4)
        curlyY['S'][i], curlyY['C'][i] = curlyYi['S'], curlyYi['C']
    
    # step 2 for all outputs o of interest (here A and C)
    curlyE = {}
    for o in ('S', 'C'):
        curlyE[o] = expectationVectors(ss[o.lower()], SS["A"], dt, T-1)                          
    
    # steps 3 and 4: build fake news matrices, convert to Jacobians
    Js = {'S': {}, 'C': {}}
    for o in Js:
        for i in shocks:
            F = np.empty((T, T))
            F[0, :] = curlyY[o][i]
            # Is curlyE'_{t-1}*curlyg_s in the paper
            F[1:, :] = curlyE[o].reshape(T-1, -1) @ curlyg[i].reshape(T, -1).T
            Js[o][i] = J_from_F(F)
    
    return Js

# %% From Canonical HANK code of Auclert et al. (2021)

dG = 0.01 * rho_G ** np.arange(T)
dZ = dG*0 
# goods market clearing H := C + G - Y 

Js = jacobian(SS, {
                    'Z': {'Z': dZ}, # since we want to compute H^{C,Z}
                    }, 
            T)

# %% Get general equilibirum Jacobian
G = -np.linalg.inv(-np.identity(T) + Js['C']['Z']) # don't need [:-1, 1:] in continuous time (in contrast to the discrete time code)
# %%
plt.plot(G[:50, [0, 10, 20]])
plt.legend(['s=0', 's=10', 's=20'])
plt.title('First-order response of $Y_t$ to $G_s$ shocks')
plt.savefig('first_order_response.pdf')
plt.show()

# %% Again just perforn the example of Auclert et al (2021)
rhos = np.array([0.5, 0.8, 0.9, 0.95, 0.975])
dGs = rhos**np.arange(T)[:, np.newaxis] # each column is a dG impulse with different persistence
dYs = G @ dGs # simple command obtains impulses to all these simultaneously
plt.plot(dYs[:25])
plt.legend([fr'$\rho={rho}$' for rho in rhos])
plt.title('First-order % response of $Y_t$ to different 1% AR(1) G shocks')
plt.savefig('first_order_response_ar1.pdf')
plt.show()

# %%
# plot 3 plots besides each other
# (1) percentage point deviation from steady state for G
# (2) percentage point deviation from steady state for Y
# (3) percentage point deviation from steady state for goods_market_clearing
fix, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(dGs[:50,1])
ax[0].set_title('G')
ax[0].set_xlabel('time')
ax[0].set_ylabel('percentage point deviation from steady state')
ax[1].plot(dYs[:50,1])
ax[1].set_title('Y')
ax[1].set_xlabel('time')
#fig[1].set_ylabel('percentage point deviation from steady state')
ax[2].plot(dGs[:50,1] - dYs[:50,1])
ax[2].set_title('goods market clearing')
ax[2].set_xlabel('time')
#fig[2].set_ylabel('percentage point deviation from steady state')
plt.savefig('first_order_response_ar1.pdf')
plt.show()

# %%
_, _, deficitBalancedBudget = fiscal(SS['B'], SS['r'], SS['G']+dGs[:50,1], SS['Y']+dYs[:50,1])
#print('Defincit deviation from Steady state: {}'.format(deficitBalancedBudget-SS['deficit']))

# plot deficit deviation from steady state
plt.plot(deficitBalancedBudget-SS['deficit'])
plt.title('Deficit (balanced budget)')
plt.xlabel('time')
plt.ylabel('deviation from steady state')
plt.show()
# %% The rest of the code just reiterates everything with the balanced budget assumption



"""
How do the defincit financed application, since dY is also changing
in the fiscal function.
"""
dGs_balancedBudget = dGs[:,1]
dYs_balancedBudget = dYs[:,1]

# %% % debt finance dG
rho_B = rho_G
dB = np.cumsum(dG) * rho_B ** np.arange(T)
dT, dZ, ddeficit = fiscal(SS['B']+dB, r, SS['G']+dG, SS['Y'])

Js = jacobian(SS, {
                    'Z': {'Z': dZ}, # since we want to compute H^{C,Z}
                    }, 
            T)

G = -np.linalg.inv(- np.identity(T) + Js['C']['Z']) 
rhos = np.array([0.5, 0.8, 0.9, 0.95, 0.975])
dGs = rhos**np.arange(T)[:, np.newaxis] # each column is a dG impulse with different persistence
dYs = G @ dGs # simple command obtains impulses to all these simultaneously!


# %% save the result for rho = 0.8 again for deficit financed
dGs_deficitFinanced = dGs[:,1].copy()
dYs_deficitFinanced  = dYs[:,1].copy()
deficit_deficitFinanced = (ddeficit)
# plot 3 plots besides each other
# (1) percentage point deviation from steady state for G, for the balanced budget case and the deficit financed case
# (2) percentage point deviation from steady state for Y, for the balanced budget case and the deficit financed case
# (3) percentage point deviation from steady state for deficit, for the balanced budget case and the deficit financed case
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(dGs_balancedBudget[:50] , label='balanced budget')
ax[0].plot(dGs_deficitFinanced[:50] , label='deficit financed', linestyle='--')
ax[0].set_title('Government spending G')
ax[0].set_xlabel('Quarter')
ax[0].set_ylabel('percentage point deviation from steady state')
#ax[0].legend()
ax[1].plot(dYs_balancedBudget[:50]  , label='balanced budget')
ax[1].plot(dYs_deficitFinanced[:50] , label='deficit financed', linestyle='--')
ax[1].set_title('Output Y')
ax[1].set_xlabel('Quarter')
#fig[1].set_ylabel('percentage point deviation from steady state')
#ax[1].legend()
ax[2].plot(deficitBalancedBudget-SS['deficit'], label='balanced budget')
ax[2].plot((deficit_deficitFinanced[:50]-SS['deficit'])/SS['deficit'], label='deficit financed', linestyle='--')
ax[2].set_title('Government deficit')
ax[2].set_xlabel('Quarter')
#fig[2].set_ylabel('percentage point deviation from steady state')
ax[2].legend()
plt.savefig('ComparisionDeficitAndBalancedBudgetShock.pdf')
plt.show()