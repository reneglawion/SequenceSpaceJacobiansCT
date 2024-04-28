# %%
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy import linalg
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib import cm
import time 

 # Settings
maxIterations = 100
Delta=1000
dt=1/10

# Shock
T = 100 
Zs = 0.01*0.95**np.arange(T)

#--------------------------------------------------
#PARAMETERS
ga = 2                                                  # CRRA utility with parameter gamma
rho = 0.08                                              # discount rate beta in discrete time
K = 3.5                                                 # capital
L = 1.                                                  # labor wlog
alpha = 0.2                                             # capital share
delta = .025                                            # depreciation rate

print('r = ', alpha  * (K / L) ** (alpha-1) - delta)
print('w = ', (1-alpha) * (K / L) ** alpha)

relax = 0.99                                            # relaxation parameter 
amin = 0.                                               # borrowing constraint    
amax = 100                                              # range a
I = 200  
J = 2

# set up Poisson process for productivity
z1 = .1
z2 = .2
z = np.asarray([z1,z2])
la1 = 1
la2 = 2
la = np.asarray([la1,la2])

#simulation parameters
maxit  = 100                                            # maximum number of iterations in the HJB loop
maxitK = 100                                            # maximum number of iterations in the K loop
crit = 10**(-6)                                         # criterion HJB loop
critK = 1e-5                                            # criterion K loop

# Utility function
def U(cc):
  return cc**(1-ga)/(1-ga)

def dUinv(vv):
  return vv**(-1/ga)

#--------------------------------------------------
# Grid 
a = np.linspace(amin,amax,num=I, endpoint=True)         #wealth vector
da = (amax-amin)/(I-1)  
da2 = da**2    

# put the variable a with dimension (I,) to a variable aa with deimension (I,J)
aa = np.zeros((I,J))
for i in range(I):
    for j in range(J):
        aa[i,j]=a[i]
zz = np.ones((I,1))*z

# Save identity matrix for later
Id = sparse.eye(I*J)

# production function
def firm(K, L, Z, alpha, delta):
    r = alpha * np.exp(Z) * (K / L) ** (alpha-1) - delta
    w = (1 - alpha) * np.exp(Z) * (K / L) ** alpha
    Y = np.exp(Z) * K ** alpha * L ** (1 - alpha)
    return r, w, Y

def dfirm_dZ(K, L, Z, alpha):
    dr = np.exp(Z) * alpha * (K / L) ** (alpha-1) 
    dw = (1 - alpha) * np.exp(Z) * (K / L) ** alpha 
    dY = np.exp(Z) * K ** alpha * L ** (1 - alpha) 
    return dr, dw, dY

def dfirm_dK(K, L, Z, alpha):
    dr = alpha * (alpha-1) * np.exp(Z) * K**(alpha-2) * L**(alpha-1) 
    dw = (1 - alpha) * alpha * np.exp(Z) * (-K**(alpha-1)) * L**(-alpha) 
    dY = alpha * np.exp(Z) * K ** (alpha-1) * L ** (1 - alpha) 
    return dr, dw, dY

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
    speye = sparse.eye(I)

    Aswitch = sparse.bmat([[-speye * la1, speye * la1], [speye * la2, -speye * la2]])

    return Aswitch

def backwardIteration(V, Aswitch, r, w, rho):
    """ Performs a backward iteration of the HJB equation

    Parameters
    ----------
    V : dense matrix
        value function in time t+1 (or current guess for SS iteration)
    Aswitch: sparse matrix
        matrix summarizing evolution of z
    r : float
        interest rate
    w : float
        wage
    rho : float
        discount rate

    Returns
    -------
    V : dense matrix
        value function in time t
    Va_Upwind : dense matrix
        costate in time t
    c : dense matrix
        consumption policy in time t
    s : dense matrix
        savings policy in time t 
    A : sparse matrix
        transition matrix in time t       
    """

    sA = sparse.lil_matrix((I*J, I*J))

    # Finite difference approximation of the partial derivatives
    Vaf = np.zeros((I,J))
    Vab = np.zeros((I,J))

    # forward difference 
    Vaf[0:-1,:] = (V[1:,:] - V[0:-1,:])/da 
    Vaf[-1,:] = (z*w + r*amax)**(-ga) 
    # backward difference
    Vab[1:, :] = (V[1:,:] - V[0:-1,:])/da  
    Vab[0, :] = (z*w + r*amin)**(-ga) # state constraint boundary condition

    # indicator whether value function is concave (problems arise if this is not the case)
    I_concave = Vab > Vaf        
    # consumption and savings with forward difference
    cf = Vaf**(-1/ga)
    sf = zz*w +aa*r-cf
    # consumption and savings with backward difference
    cb = Vab**(-1/ga)
    sb = zz*w+aa*r-cb
    # consumption and derivative of value function at steady state
    c0 = zz*w+aa*r
    Va0 = c0**(-ga)

    # dV_upwind makes a choice of forward or backward differences based on
    # the sign of the drift
    If = sf > 0 # positive drift --> forward difference
    Ib = sb < 0 # negative drift --> backward difference
    I0 = (1-If-Ib) # at steady state

    Va_Upwind = Vaf*If + Vab*Ib + Va0*I0 # important to include third term

    c = Va_Upwind**(-1/ga)
    s = zz*w + aa*r - c 
    u = c**(1-ga)/(1-ga) 

    # Construct matrix A
    X = - np.where(sb<0,sb,0)/da
    Z = np.where(sf>0,sf,0)/da  
    Y = -X-Z    

    A_diag_0=np.reshape(np.transpose(Y),(I*J)) # the transpose is really important here 
                                                # (in contrast to the Matlab implentation
                                                # by Achdou et al (2022))

    # updiag 
    A_diag_p1 = []  # Matlab does need a zero here - python does not 
    for j in range(J):
        A_diag_p1 = np.append(A_diag_p1, Z[0:-1,j]) 
        A_diag_p1 = np.append(A_diag_p1, 0)

    # lowdiag 
    A_diag_m1 =  X[1:, 0]
    for j in range(1, J):
        A_diag_m1 = np.append(A_diag_m1, 0)
        A_diag_m1 = np.append(A_diag_m1, X[1:,j])

    sA.setdiag(A_diag_0)         
    sA.setdiag(A_diag_p1,k=1)   
    sA.setdiag(A_diag_m1,k=-1)  

    A = Aswitch + sA

    u = U(np.reshape(np.transpose(c),I*J))
    v_stacked = np.reshape(np.transpose(V),I*J)
        
    # V = (sparse.linalg.inv((1/Delta+rho)*Id-A)).dot(u+v_stacked/Delta) # to check but slow
    BB = (1/Delta+rho)*Id-A
    b = u+v_stacked/Delta
    V = spsolve(BB, b)

    V=np.reshape(V,(J,I)).T

    return V, Va_Upwind, c, s, A

def policy_ss(r, w, rho):
    """
    Computes the steady state policy functions

    Parameters
    ----------
    r : float
        interest rate
    w : float
        wage
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
            v[i, j] = U(w*z[j]+r*a[i])/rho

    # Steady state iteration until convergence
    for i in range(maxit):

        vold = v.copy()

        v, Va, c, s, A = backwardIteration(vold, Aswitch, r, w, rho)

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

    try:
        # should always work  
        [eigenvalue, eigenvec] = sparse.linalg.eigs(
                AT, k=1, sigma=0., return_eigenvectors=True)
            
        #print("Eigenvalue = {}".format(eigenvalue[0]))

        for i in range(I):

            for j in range(J):

                pdf[i, j] = abs(eigenvec[j*I+i][0])
    except:
        # However, if not use 
        try: # dirty trick
            b = np.zeros((I*J,1))
            b[0] = 0.1
            AT.data[1:AT.indptr[1]] = 0
            AT.data[0] = 1.0
            AT.indices[0] = 0
            AT.eliminate_zeros()
            pdf = spsolve(AT,b).reshape(I, J)
        except:
            # solve perturbed system if dirty trick also fails (if you land here, check calibration)

            AT.data[1:AT.indptr[1]] = 0
            AT.data[0] = 1.0
            AT.indices[0] = 0
            AT.eliminate_zeros()

            [eigenvalue, eigenvec] = sparse.linalg.eigs(
                    AT, k=1, sigma=0., return_eigenvectors=True)
                
            #print("Eigenvalue pertrubed system = {}".format(eigenvalue[0]))
            for i in range(I):

                for j in range(J):

                    pdf[i, j] = abs(eigenvec[j*I+i][0])
    
    # normalize pdf to sum to 1
    M=np.zeros(J)
    for j in range(J):
        M[j]=da*np.sum(pdf[:,j])

    Mass=np.sum(M)
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
        
    S=np.sum(S0)
    C=np.sum(C0)
    Assets=np.sum(Assets0)
    
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


def steady_state(K,Z,alpha,delta,rho):
    """
    Computes the steady state distribution of agents

    Parameters
    ----------
    K : float
        capital
    Z : float
        TFP
    alpha : float
        factor share of capital
    delta : float
        depreciation rate
    rho : float
        discount rate
    
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
        'Y' : steady state value of output
        'K' : steady state amount of capital
        'L' : steady state amount of labor
        'alpha' : steady state capital share
        'delta' : steady state depreciation rate
        'asset_mkt_error' : steady state asset market clearing error
        'Zs' : possible path of shocks
    """

    # Fix L=1 wlog and Z=1 in steady state
    asset_mkt_error = 1.
    while asset_mkt_error > 1e-4:
        r, w, Y = firm(K, 1., Z, alpha, delta)
        V, Va, c, s, A, Aswitch = policy_ss(r, w, rho)
        g = distribution_ss(A)

        C, S, Assets = getAggregateConsumptionAndSavings(g, c, s)

        asset_mkt_error = np.abs(Assets-K)
        K = relax*K + (1-relax) * Assets
    
    return dict(g=g, Va=Va, V=V, A=A, Aswitch=Aswitch,
                a=a, c=c, s=s,
                S=S, C=C, Assets = Assets,
                r=r, w=w, rho=rho, Y=Y,
                K=K, L=1., Z=Z,
                alpha=alpha, delta=delta,
                asset_mkt_error = np.abs(Assets-K),
                Zs=1.)

SS = steady_state(K,0.,alpha,delta,rho)

# %% Calibration GE

def SS_Z(curr_guess):
    SStmp = steady_state(SS['K'],np.squeeze(curr_guess),SS["alpha"],SS['delta'],SS['rho'])
    return SStmp["asset_mkt_error"] + (SStmp["Y"]-1.)**2

Z_calib = optimize.fsolve(
                SS_Z,
                SS["Z"]-0.1447998
                )
# %%
Z_calib = np.squeeze(Z_calib)
print("Z_calib = {}".format(Z_calib))
SS = steady_state(SS["K"],Z_calib,SS["alpha"],SS["delta"],SS["rho"])
print("Asset Market error = {}".format(SS["asset_mkt_error"]))
print("Total savings = {}".format(SS["S"]))
print("Consumption = {}".format(SS["C"]))
print("Total Production = {}".format(SS["Y"]))

# Plot the value function v and the stationary distribution pdf in one plot beside each other as surface plots
# below that plot the consumption policy function c and besides that the savings policy function s 
fig = plt.figure(figsize=(12, 12))
fig.suptitle("General Equilibrium Steady State", fontsize=16)
fig, ax = plt.subplots(2, 2, figsize=(12, 6))
ax[0,0].plot(a, SS["V"][:,0], label="z=0")
ax[0,0].plot(a, SS["V"][:,1], label="z=1")
ax[0,0].set_xlabel("Assets")
ax[0,0].set_ylabel("Value Function")
ax[0,0].set_title("Value Function")
ax[0,0].legend()
ax[0,1].plot(a, SS["g"][:,0], label="z=0")
ax[0,1].plot(a, SS["g"][:,1], label="z=1")
ax[0,1].set_xlabel("Assets")
ax[0,1].set_ylabel("Density")
ax[0,1].set_title("Stationary Distribution")
ax[0,1].legend()
ax[1,0].plot(a, SS["c"][:,0], label="z=0")
ax[1,0].plot(a, SS["c"][:,1], label="z=1")
ax[1,0].set_xlabel("Assets")
ax[1,0].set_ylabel("Consumption")
ax[1,0].set_title("Consumption Policy Function")
ax[1,0].legend()
ax[1,1].plot(a, SS["s"][:,0], label="z=0")
ax[1,1].plot(a, SS["s"][:,1], label="z=1")
ax[1,1].set_xlabel("Assets")
ax[1,1].set_ylabel("Savings")
ax[1,1].set_title("Savings Policy Function")
ax[1,1].legend()
plt.tight_layout()

# safve the figure with the title as name as a pdf file
fig.savefig('GeneralEquilibriumSteadyState.pdf')
plt.show()

# %% Fake News Algorithm to get Jacobian J

# Shock
T = 100 # so 40 quarters
Zs =  0.01*0.95**np.arange(T)

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
    ss_inputs = {k: ss[k] for k in ('Aswitch', 'r', 'w', 'rho')} 
    
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
        
        S0, C0 = np.zeros(J), np.zeros(J)
        
        for j in range(J):
            S0[j] = da*np.sum(SS["g"][:,j]*(savings[:,j]-SS["s"][:,j]))
            C0[j] = da*np.sum(SS["g"][:,j]*(c[:,j]-SS["c"][:,j]))
        
        S=np.sum(S0)
        C=np.sum(C0)

        curlyY['S'][s] = S/h
        curlyY['C'][s] = C/h
        
        # what is effect on one-period-ahead distribution
        curlyg[s] = (forwardIteration(SS["g"], SS["A"], dt) - g1_noshock) / h
        
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
    #for i, shock in shocks.items():
    for i, shock in shocks.items():
        curlyYi, curlyg[i] = step1_backward(ss, shock, T, 1E-4)
        curlyY['S'][i], curlyY['C'][i] = curlyYi['S'], curlyYi['C']
    
    # step 2 for all outputs o of interest (here A and C)
    curlyE = {}
    for o in ('S', 'C'):
        # Syntax is: expectationVectors(X, A, dt, T):
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

# %%  time the first impulse reponse 
start_time = time.time()

# % get J_ha
J_ha = jacobian(SS, {
                    'r': {'r': Zs}, 
                    'w': {'w': Zs}, 
                    }, 
            T)

# % get J_firm
J_firm = {
        'r': {'K' : np.empty_like(Zs), 'Z' : np.empty_like(Zs)},
        'w': {'K' : np.empty_like(Zs), 'Z' : np.empty_like(Zs)},
        'Y': {'K' : np.empty_like(Zs), 'Z' : np.empty_like(Zs)}
        }

J_firm['r']['Z'], J_firm['w']['Z'], J_firm['Y']['Z'] = dfirm_dZ(SS['K'], SS['L'], SS['Z'], SS['alpha'])
J_firm['r']['K'], J_firm['w']['K'], J_firm['Y']['K'] = dfirm_dK(SS['K'], SS['L'], SS['Z'], SS['alpha'])

# % scale all firm Jacobians by np.eye(T) to get the right dimensions
for o in J_firm:
    for i in J_firm[o]:
        J_firm[o][i] = np.eye(T)*J_firm[o][i]

# % Get general equilibirum Jacobian

J_curlyK_K = J_ha['S']['r'] @ J_firm['r']['K'] + J_ha['S']['w'] @ J_firm['w']['K']
J_curlyK_Z = J_ha['S']['r'] @ J_firm['r']['Z'] + J_ha['S']['w'] @ J_firm['w']['Z']

H_K = J_curlyK_K - np.eye(T)
H_Z = J_curlyK_Z

Jac = J_firm.copy()
Jac['curlyK'] = {'K' : J_curlyK_K, 'Z' : J_curlyK_Z}

G = {'K': -np.linalg.solve(H_K, H_Z)}
end_time = time.time()
print('Time elapsed for first impulse response: {} seconds'.format(end_time - start_time))

# time this part of the code to see how long computing the additional impulse responses take
start_time = time.time()
# get all other impulses
G['r'] = Jac['r']['Z'] + Jac['r']['K'] @ G['K']
G['w'] = Jac['w']['Z'] + Jac['w']['K'] @ G['K']
G['Y'] = Jac['Y']['Z'] + Jac['Y']['K'] @ G['K']
G['C'] = J_ha['C']['r'] @ G['r'] + J_ha['C']['w'] @ G['w']

rhos = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
dZ = 0.01 * rhos ** (np.arange(T)[:, np.newaxis]) # get T*5 matrix of dZ

dY = G['Y'] @ dZ
dC = G['C'] @ dZ
dr = G['r'] @ dZ
end_time = time.time()
print('Time elapsed for addtiomal impulse responses: {} seconds'.format(end_time - start_time))

# %% 2x2 plot of dZ, dY, dC, dr
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
#plt.tight_layout()
#plt.gca().set_facecolor('none')
axs[0, 0].plot(dZ[:50, :])
axs[0, 0].set_title(r'TFP $Z$')
axs[0, 0].set_ylabel(r'deviation from ss')
axs[0, 0].set_xlabel(r'quarters')
axs[0, 0].legend([r'$\rho={}$'.format(rho) for rho in rhos])
axs[0, 1].plot(dY[:50, :])
axs[0, 1].set_title(r'Output $Y$')
axs[0, 1].set_ylabel(r'deviation from ss')
axs[0, 1].set_xlabel(r'quarters')
axs[1, 0].plot(dC[:50, :])
axs[1, 0].set_title(r'Aggregate consumption $C$')
axs[1, 0].set_ylabel(r'deviation from ss')
axs[1, 0].set_xlabel(r'quarters')
axs[1, 1].plot(10000*dr[:50, :])
axs[1, 1].set_title(r'Interest rate $r$')
axs[1, 1].set_ylabel(r'basis points deviation from ss')
axs[1, 1].set_xlabel(r'quarters')
# save the figure as pdf
plt.savefig('ResultsKS.png', bbox_inches='tight')
plt.show()

# %% Estimation part
"""
Here we use the code of Auclert et al. (2021) to match the discretized version
of our continuous time model, so that we can directly use their SSJ Toolbox
to estimate the model. The relevant code of their package is in the estimation file.
The interesting observation is that we can just copy/paste their code with the matched
continuous time model.
"""

import estimation
rho = 0.9
sigma_persist = 0.1
sigma_trans = 0.2

dZ1 = rho**(np.arange(T))
dY1, dC1, dK1 = G['Y'] @ dZ1, G['C'] @ dZ1, G['K'] @ dZ1
dX1 = np.stack([dZ1, dY1, dC1, dK1], axis=1)

dZ2 = np.arange(T) == 0
dY2, dC2, dK2 = G['Y'] @ dZ2, G['C'] @ dZ2, G['K'] @ dZ2
dX2 = np.stack([dZ2, dY2, dC2, dK2], axis=1)

dX = np.stack([dX1, dX2], axis=2)
dX.shape

sigmas = np.array([sigma_persist, sigma_trans])
Sigma = estimation.all_covariances(dX, sigmas) # burn-in for jit


sd = np.sqrt(np.diag(Sigma[0, ...]))
correl = (Sigma/sd)/(sd[:, np.newaxis])


ls = np.arange(-50, 51)
corrs_l_positive = correl[:51, 0, :]
corrs_l_negative = correl[50:0:-1, :, 0]
corrs_combined = np.concatenate([corrs_l_negative, corrs_l_positive])

plt.plot(ls, corrs_combined[:, 0], label='dZ')
plt.plot(ls, corrs_combined[:, 1], label='dY')
plt.plot(ls, corrs_combined[:, 2], label='dC')
plt.plot(ls, corrs_combined[:, 3], label='dK')
plt.legend()
plt.title(r'Corr of $dZ_t$ and $X_{t+l}$ for various series $X$')
plt.xlabel(r'Lag $l$')
plt.show()

# %% Log likelihood
# random 100 observations
Y = np.random.randn(100, 4)

# 0.05 measurement error in each variable
sigma_measurement = np.full(4, 0.05)

# calculate log-likelihood
estimation.log_likelihood(Y, Sigma, sigma_measurement)

def log_likelihood_from_parameters(rho, sigma_persist, sigma_trans, sigma_measurement, Y):
    # impulse response to persistent shock
    dZ1 = rho**(np.arange(T))
    dY1, dC1, dK1 = G['Y'] @ dZ1, G['C'] @ dZ1, G['K'] @ dZ1
    dX1 = np.stack([dZ1, dY1, dC1, dK1], axis=1)
    
    # since transitory shock does not depend on any unknown parameters,
    # except scale sigma_trans, we just reuse the dX2 already calculated earlier!
    
    # stack impulse responses together to make MA(T-1) representation 'M'
    M = np.stack([dX1, dX2], axis=2)
    
    # calculate all covariances
    Sigma = estimation.all_covariances(M, np.array([sigma_persist, sigma_trans]))
    
    # calculate log=likelihood from this
    return estimation.log_likelihood(Y, Sigma, sigma_measurement)


# stack covariances into matrix using helper function, then do a draw using NumPy routine
V = estimation.build_full_covariance_matrix(Sigma, sigma_measurement, 100)
Y = np.random.multivariate_normal(np.zeros(400), V).reshape((100, 4))


sigma_persist_values = np.linspace(0.05, 0.2, 100)
lls = np.array([log_likelihood_from_parameters(rho, sigma_persist, sigma_trans, sigma_measurement, Y) for sigma_persist in sigma_persist_values])


plt.plot(sigma_persist_values, lls)
plt.axvline(0.1, linestyle=':', color='gray')
plt.title(r'Log likelihood of simulated data as function of $\sigma_{persist}$')
plt.show()