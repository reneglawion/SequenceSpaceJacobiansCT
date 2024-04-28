# %%
"""
Implement the Version from the pdf 
1. Labor earnings z follow diffusion process
2. savings function is 
    s = (Z-tau)*w*z + (1+r)*k - c
3. Governemnt collects tax tau and has budget constraint
    0 = G = (1+r)*B + tau*w*z
    set taxes tau_t=t_T*B_t/e, where e is average endowment, so mu=1 
    => tau_t = r_T*B_t
"""

# %%
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
dt=1/10

# Shock
T = 100 # so 40 quarters
Zs = 1 + 0.01*0.95**np.arange(T)

#--------------------------------------------------
#PARAMETERS
ga = 2       # CRRA utility with parameter gamma
w = 1.      # mean O-U process (in levels). This parameter has to be adjusted to ensure that the mean of z (truncated gaussian) is 1.
r = 0.03
Corr = .9
the = -np.log(Corr)
sig2 = 0.05;  # sigma^2 O-U
rho = 0.05;   # discount rate beta in discrete time

relax = 0.999 # relaxation parameter 
zmin = .5   # Range z
zmax = 1.5
amin = -2.     # borrowing constraint     HIER SCHAUEN
amax = 30    # range a
I = 40  
J = 10

#simulation parameters
maxit  = 100;     #maximum number of iterations in the HJB loop
maxitK = 100;    #maximum number of iterations in the K loop
crit = 10^(-6); #criterion HJB loop
critK = 1e-5;   #criterion K loop

def U(cc):
  return cc**(1-ga)/(1-ga)

def dUinv(vv):
  return vv**(-1/ga)

def beta(rr):
    return 1.

#--------------------------------------------------
# Grid 
a = np.linspace(amin,amax,num=I, endpoint=True) #tf.reshape(tf.linspace(amin,amax,I),(I,1))  #wealth vector
da = (amax-amin)/(I-1)  
da2 = da**2    
z = np.linspace(zmin,zmax,num=J, endpoint=True)   # productivity vector
dz = (zmax-zmin)/(J-1)
dz2 = dz**2
aa = np.reshape(a,(I,1))*np.ones((1,J)) # solle so stimmen
zz = np.ones((I,1))*z

mu = the*(w - z)        #DRIFT (FROM ITO'S LEMMA)
s2 = np.squeeze(sig2*np.ones((1,J)))

Id = sparse.eye(I*J)

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
    # CONSTRUCT MATRIX Aswitch SUMMARIZING EVOLUTION OF z
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

def backwardIteration(V, Aswitch, r, w, rho, B=0., tau=0., Zs=1.):
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
    B : float
        government debt, default 0.
    tau : float
        tax rate, default 0.
    Z : float
        current productivity, default = 1.

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
    Vaf[-1,:] = ((Zs-tau)*z*w + r*amax)**(-ga) 
    # backward difference
    Vab[1:, :] = (V[1:,:] - V[0:-1,:])/da  
    Vab[0, :] = ((Zs-tau)*z*w + r*amin)**(-ga) #  %state constraint boundary condition

    # indicator whether value function is concave (problems arise if this is not the case)
    I_concave = Vab > Vaf        
    # consumption and savings with forward difference
    cf = Vaf**(-1/ga)
    sf = (Zs-tau)*zz*w +aa*r-cf
    # consumption and savings with backward difference
    cb = Vab**(-1/ga)
    sb = (Zs-tau)*zz*w+aa*r-cb
    # consumption and derivative of value function at steady state
    c0 = (Zs-tau)*zz*w+aa*r
    Va0 = c0**(-ga)

    # dV_upwind makes a choice of forward or backward differences based on
    # the sign of the drift
    If = sf > 0 #positive drift --> forward difference
    Ib = sb < 0#negative drift --> backward difference
    I0 = (1-If-Ib) #at steady state

    Va_Upwind = Vaf*If + Vab*Ib + Va0*I0 #important to include third term # check‚‚

    c = Va_Upwind**(-1/ga)
    s = (Zs-tau)*zz*w + aa*r - c 
    u = c**(1-ga)/(1-ga) 

    # CONSTRUCT MATRIX A
    X = - np.where(sb<0,sb,0)/da
    Z = np.where(sf>0,sf,0)/da  
    Y = -X-Z    

    A_diag_0=np.reshape(np.transpose(Y),(I*J)) # the transpose is really important here 
                                                # (in contrast to the Matlab implentation
                                                # by Achdou et al (2022))

    # updiag 
    A_diag_p1 = []  # matlab need a zero here - python not 
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
        
    # V = (sparse.linalg.inv((1/Delta+rho)*Id-A)).dot(u+v_stacked/Delta)
    BB = (1/Delta+rho)*Id-A
    b = u+v_stacked/Delta
    V = spsolve(BB, b)

    V=np.reshape(V,(J,I)).T

    return V, Va_Upwind, c, s, A

def policy_ss(r, w, rho, B=0, tau=0):
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
    B : float
        government debt
    tau : float
        tax rate

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

        v, Va, c, s, A = backwardIteration(vold, Aswitch, r, w, rho, B, tau)

        if np.max(np.max(np.abs(v-vold))) < crit:
            break

    return v, Va, c, s, A, Aswitch

def distribution_ss(A):
    """
    Computes the steady state distribution of agents

    input:  A = infenitesimal operator of the HJB equation (operator curlyA in the paper)

    output: D_ss = steady state distribution of agents
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

def getAggregateConsumptionAndSavings(D, c, s):
    """
    Computes aggregate consumption and savings

    input:  D =  distribution of agents
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

        S0[j] = da*np.sum(D[:,j]*s[:,j]) # why a and not s[:,j]?
        C0[j] = da*np.sum(D[:,j]*c[:,j])
        Assets0[j] = da*np.sum(D[:,j]*a[:])
        
    S=dz*np.sum(S0)
    C=dz*np.sum(C0)
    Assets=dz*np.sum(Assets0)
    
    return C, S, Assets

def forwardIteration(D_t, A, dt):
    """
    Computes the forward iteration for the FPE 

    Parameters
    ----------
    D_t : dense matrix
        distribution at time t
    A : sparse matrix
        Generator of the HJB equation (operator curlyA in the paper)
    dt : float
        time step size

    Returns
    -------
    D_tplus1 : dense matrix
        distribution at time t+dt
    """
    I, J = D_t.shape
    D_t = np.reshape(D_t.T, (I*J, 1))
    D_tplusdt = spsolve(Id - A.T*dt, D_t) # solve linear system with A^T
    D_tplusdt = np.reshape(D_tplusdt,(J,I)).T
    return D_tplusdt

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
    IS THIS CORRECT??

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


def steady_state(r,w,rho,B=0, tau=0):
    """
    Computes the steady state distribution of agents

    Parameters
    ----------
    r : float
        interest rate
    w : float
        wage
    rho : float
        discount rate
    B : float
        government debt, default 0
    tau : float
        tax rate, default 0
    
    Returns
    -------
    d : dictionary
        steady state values of all variables with keys:
        'D' : steady state distribution of agents
        'C' : steady state aggregate consumption
        'S' : steady state aggregate savings
        'v' : steady state value function
        'Va' : steady state derivative of value function (costate)
        'c' : steady state consumption
        's' : steady state savings
        'A' : steady state generator of the HJB equation
    """

    V, Va, c, s, A, Aswitch = policy_ss(r, w, rho, B, tau)
    D = distribution_ss(A)

    C, S, Assets = getAggregateConsumptionAndSavings(D, c, s)
    
    return dict(D=D, Va=Va, V=V, A=A, Aswitch=Aswitch,
                a=a, c=c, s=s,
                S=S, C=C, Assets = Assets,
                r=r, w=w, rho=rho, B=B, tau= tau,
                asset_mkt_error = np.abs(Assets-B),
                Zs=1.)

SS = steady_state(0.02, 1., 0.05, B=0)

# Plot the value function v and the stationary distribution pdf in one plot beside each other as surface plots
# below that plot the consumption policy function c and besides that the savings policy function s 
fig = plt.figure(figsize=(12, 12))
# give the plot the the title "Partial Equilibrium Steady State"
fig.suptitle("Partial Equilibrium Steady State", fontsize=16)
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
ax2.plot_surface(X, Y, SS['D'].T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
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
fig.savefig('PartialEquilibriumSteadyState.pdf')
plt.show()

# %% Calibration GE
B = 5.6
tau = r*B

def SS_rho(rho_guess):
    SS = steady_state(r,w,np.squeeze(rho_guess),B=B, tau=tau)
    return SS["asset_mkt_error"]

rho_calib = optimize.fsolve(
                SS_rho,
                0.04)
rho_calib = np.squeeze(rho_calib)
print("rho_calib = {}".format(rho_calib))
SS = steady_state(r, w, rho_calib, B=B, tau=tau)
print("Asset Market error = {}".format(SS["asset_mkt_error"]))
print("Total savings = {}".format(SS["S"]))
print("Consumption = {}".format(SS["C"]))

# Plot the value function v and the stationary distribution pdf in one plot beside each other as surface plots
# below that plot the consumption policy function c and besides that the savings policy function s 
fig = plt.figure(figsize=(12, 12))
# give the plot the the title "General Equilibrium Steady State"
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
ax2.plot_surface(X, Y, SS['D'].T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
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

# %% test with Consumption
ctilde = SS["c"] - SS["C"] # demeaned consumption
E_ctilde = expectationVectors(ctilde, SS["A"], dt, T) # expe
#E = expectationVectors(SS["c"], SS["A"], 1, 30)
Autocov_c = np.empty(T)
for j in range(T):
    Autocov_c[j] = np.vdot(SS["D"], ctilde * E_ctilde[j])

Autocorr_c = Autocov_c / Autocov_c[0]

Autocov_c_alt = E_ctilde.reshape((T, -1)) @ (ctilde*SS["D"]).ravel()
#assert np.max(np.abs(Autocov_c_alt - Autocov_c)) < 1E-15 # verify this is right

# Plot autocorrelation of consumption and scale the x axes with dt so that T is in quarters
plt.plot(np.arange(T)*dt, Autocorr_c)
plt.title('Autocorrelation of consumption')
plt.xlabel('Horizon in quarters')
plt.ylabel('Correlation')
plt.tight_layout()
# save the figure with the title as name as a pdf file
plt.savefig('AutocorrelationConsumption.pdf')
plt.show()

# %% Fake News Algorithm to get Jacobian J

def J_from_F(F):
    """
    Computes the Jacobian J from  curlyF_{t,s} dx = curlyE_{t-1} dD_1^s

    Parameters
    ----------
    F : sequence of dense matrices
        curlyF_{t,s} dx = curlyE_{t-1} dD_1^s
    
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
    curlyD : sequence of dense matrices
        Disrtributions in each time step t after shock
    """

    # preliminaries: D_1 with no shock, ss inputs to backward_iteration
    D1_noshock = forwardIteration(ss["D"], ss["A"], dt)
    ss_inputs = {k: ss[k] for k in ('Aswitch', 'r', 'w', 'rho', 'B', 'tau')} 
    
    # allocate space for results
    curlyY = {'C': np.empty(T), 'S': np.empty(T)}
    curlyD = np.empty((T,) + ss['D'].shape)

    # syntax for backward Iteration is:
    # backwardIteration(v, Aswitch, r, w, rho, B=0, tau=0)
    
    V = ss['V'] # initialize with steady-state value function
    # backward iterate
    for s in range(T):
        if s == 0:
            # at horizon of s=0, 'shock' actually hits, override ss_inputs with shock
            shocked_inputs = {k: ss[k] + h*shock[k][s] for k in shock}
            #shocked_inputs = {'w': shock[s]}
            V, _, c, savings, Atmp = backwardIteration(**{'V' : V, **ss_inputs, **shocked_inputs})
        else:
            # now the only effect is anticipation, so it's just Va being different
            V, _, c, savings, Atmp = backwardIteration(**{**ss_inputs, 'V': V})
        
        # shocked_inputs = {'w': shock[s]}
        # V, _, c, savings, Atmp = backwardIteration(**{'V' : V, **ss_inputs, **shocked_inputs})

        # aggregate effects on A and C
        # curlyY['A'][s] = np.vdot(ss['D'], a - ss['a']) / h
        # curlyY['C'][s] = np.vdot(ss['D'], c - ss['c']) / h
        S0, C0 = np.zeros(J), np.zeros(J)
        
        for j in range(J):
            # aggregate effects on S do not really make sense, since 'a' is not a control here (only consumption c)
            S0[j] = da*np.sum(SS["D"][:,j]*(savings[:,j]-SS["s"][:,j]))
            C0[j] = da*np.sum(SS["D"][:,j]*(c[:,j]-SS["c"][:,j]))
        
        S=dz*np.sum(S0)
        C=dz*np.sum(C0)

        curlyY['S'][s] = S/h
        curlyY['C'][s] = C/h
        
        # what is effect on one-period-ahead distribution?
        # a_i_shocked, a_pi_shocked = sim.interpolate_lottery_loop(a, ss['a_grid'])
        # hier stand D[s] anstatt SS["D"]
        curlyD[s] = (forwardIteration(SS["D"], Atmp, dt) - D1_noshock) / h
        
    return curlyY, curlyD

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
    # step 1 for all shocks i, allocate to curlyY[o][i] and curlyD[i]
    curlyY = {'S': {}, 'C': {}}
    curlyD = {}
    #for i, shock in shocks.items():
    for i, shock in shocks.items():
        curlyYi, curlyD[i] = step1_backward(ss, shock, T, 1E-4)
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
            # überprüfen ob dieser reshape auch in CT stimmt
            # Ist curlyE'_{t-1}*curlyD_s im Paper
            F[1:, :] = curlyE[o].reshape(T-1, -1) @ curlyD[i].reshape(T, -1).T
            Js[o][i] = J_from_F(F)
    
    return Js

Js = jacobian(SS, {
                    'r_direct': {'r': np.ones_like(Zs)}, 
                    'Z': {'Zs': Zs}, 
                    #'T': {'Zs': np.ones_like(Zs)}
                    }, 
            T)

# %% Get general equilibirum Jacobian
G = -np.linalg.solve(Js['S']['r_direct'][:-1, 1:] - 0*Js['C']['Z'][:-1, 1:], Js['S']['Z'][:-1, :-1]) 
# %%
plt.plot(G[:50, [0, 10, 20]])
plt.legend(['s=0', 's=10', 's=20'])
plt.title('First-order response of $r^{ante}_t$ to $Z_s$ shocks')
# save the figure to a file with the name 'first_order_response.pdf'
plt.savefig('first_order_response.pdf')
plt.show()

# %%
rhos = np.array([0.5, 0.8, 0.9, 0.95, 0.975])
dZs = rhos**np.arange(T-1)[:, np.newaxis] # each column is a dZ impulse with different persistence
drs = G @ dZs # simple command obtains impulses to all these simultaneously!
plt.plot(drs[:50])
plt.legend([fr'$\rho={rho}$' for rho in rhos])
plt.title('First-order % response of $r^{ante}_t$ to different 1% AR(1) Z shocks')
plt.savefig('first_order_response_ar1.pdf')
plt.show()
# %%
