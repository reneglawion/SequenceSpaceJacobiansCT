# Source: https://github.com/shade-econ/nber-workshop-2022/blob/main/Tutorials/Tutorial%201%20Fiscal%20Policy.ipynb
# %%
import numpy as np  # numpy helps us perform linear algebra calculations
import matplotlib.pyplot as plt  # helps us plot
import sequence_jacobian as sj  # SSJ will allow us to define blocks, models, compute IRFs, etc

calibration = {'eis': 0.5,  # EIS
               'rho_e': 0.9,  # Persistence of idiosyncratic productivity shocks
               'sd_e': 0.92,  # Standard deviation of idiosyncratic productivity shocks
               'G': 0.2,  # Government spending
               'B': 0.8,  # Government debt
               'Y': 1.,  # Output
               'min_a': 0.,  # Minimum asset level on the grid
               'max_a': 40,  # Maximum asset level on the grid
               'n_a': 200,  # Number of asset grid points
               'n_e': 10}  # Number of productivity grid points

# initialize
def hh_init(a_grid, z, r, eis):
    coh = (1 + r) * a_grid[np.newaxis, :] + z[:, np.newaxis]
    Va = (1 + r) * coh ** (-1 / eis)
    return Va

# backward step
@sj.het(exogenous='Pi',  # <-- this means our transition matrix will be fed into the model as Pi (use this for forward iteration)
        policy='a',  # <-- this means our endogenous state variable is a, defined over grid a_grid (we use this to check convergence)
        backward='Va',  # <-- this means we're iterating over variable Va, whose future value is Va_p (solver needs to know this to iterate!)
        backward_init=hh_init)
def hh(Va_p, a_grid, z, r, beta, eis):
    uc_nextgrid = beta * Va_p  # u'(c') on tomorrow's grid
    c_nextgrid = uc_nextgrid ** (-eis)  # c' on tomorrow's grid
    coh = (1 + r) * a_grid[np.newaxis, :] + z[:, np.newaxis]  # cash on hand on today's grid
    a = sj.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)  # this plots (c_next + a', a') pairs and computes policy a' from interpolation on coh
    sj.misc.setmin(a, a_grid[0])  # impose borrowing constraint
    c = coh - a  # back out consumption
    Va = (1 + r) * c ** (-1 / eis)  # V'(a)
    return Va, a, c

print(hh)
print('It has inputs: ' + str(hh.inputs))
print('It has outputs: ' + str(hh.outputs))

def make_grids(rho_e, sd_e, n_e, min_a, max_a, n_a):
    e_grid, _, Pi = sj.grids.markov_rouwenhorst(rho_e, sd_e, n_e)
    a_grid = sj.grids.asset_grid(min_a, max_a, n_a)
    return e_grid, Pi, a_grid

def income(Z, e_grid):
    z = Z * e_grid
    return z

hh_extended = hh.add_hetinputs([make_grids, income])

print(hh_extended)
print('It has inputs: ' + str(hh_extended.inputs))
print('It has outputs: ' + str(hh_extended.outputs))

@sj.simple
def fiscal(B, r, G, Y):
    T = (1 + r) * B(-1) + G - B  # total tax burden
    Z = Y - T  # after tax income
    deficit = G - T
    return T, Z, deficit

@sj.simple
def mkt_clearing(A, B, Y, C, G):
    asset_mkt = A - B
    goods_mkt = Y - C - G
    return asset_mkt, goods_mkt

ha = sj.create_model([hh_extended, fiscal, mkt_clearing], name="Simple HA Model")

print(ha)
print('It has inputs: ' + str(ha.inputs))
print('It has outputs: ' + str(ha.outputs))

calibration['r'] = 0.03
calibration['beta'] = 0.85

ss = ha.steady_state(calibration)
ss['asset_mkt']

calibration['r'] = 0.03
calibration['beta'] = 0.85

ss = ha.steady_state(calibration)
ss['asset_mkt']

unknowns_ss = {'beta': (0.75, 0.9)}  # provide bounds on beta for the solver
targets_ss = ['asset_mkt']  # set the ss target

ss = ha.solve_steady_state(calibration, unknowns_ss, targets_ss)

print(f"Goods market clearing: {ss['goods_mkt']}")

print(hh_extended.outputs)

def compute_weighted_mpc(c, a, a_grid, r, e_grid):
    """Approximate mpc out of wealth, with symmetric differences where possible, exactly setting mpc=1 for constrained agents."""
    mpc = np.empty_like(c)
    post_return = (1 + r) * a_grid
    mpc[:, 1:-1] = (c[:, 2:] - c[:, 0:-2]) / (post_return[2:] - post_return[:-2])
    mpc[:, 0] = (c[:, 1] - c[:, 0]) / (post_return[1] - post_return[0])
    mpc[:, -1] = (c[:, -1] - c[:, -2]) / (post_return[-1] - post_return[-2])
    mpc[a == a_grid[0]] = 1
    mpc = mpc * e_grid[:, np.newaxis]
    return mpc

hh_extended = hh_extended.add_hetoutputs([compute_weighted_mpc])

ha = sj.create_model([hh_extended, fiscal, mkt_clearing], name="Simple HA Model")

print(hh_extended.outputs)

hh_extended.steady_state(ss)['MPC']

unknowns_ss = {'beta': 0.8, 'B': 0.4}
targets_ss = {'asset_mkt': 0., 'MPC': 0.25}  # <-- with a dict rather than a list, we can specify specific targets for output variables

ss_mpc = ha.solve_steady_state(calibration, unknowns_ss, targets_ss)

print(f"To achieve an MPC of 0.25, we had to reduce the available gov't debt from {ss['B']} to {ss_mpc['B']}")

ss = ss_mpc

ss['Z'], ss['T'], ss['G'], ss['A']

# for each key in ss print the value
for key, value in ss.items():
    print(f"{key} = {value}")

D = ss.internals['hh']['D'].sum(axis=0)
a_grid = ss.internals['hh']['a_grid']
plt.plot(a_grid, D.cumsum())
plt.ylim([0.2, 1])
plt.xlim([0, 5])
plt.xlabel('Assets')
plt.ylabel('Cumulative distribution')
plt.show()

# %% Transitional Dynamics
T = 300  # <-- the length of the IRF
rho_G = 0.8
dG = 0.01 * rho_G ** np.arange(T)
shocks = {'G': dG}

unknowns_td = ['Y']
targets_td = ['asset_mkt']

irfs = ha.solve_impulse_linear(ss, unknowns_td, targets_td, shocks)

def show_irfs(irfs_list, variables, labels=[" "], ylabel=r"Percentage points (dev. from ss)", T_plot=50, figsize=(18, 6)):
    if len(irfs_list) != len(labels):
        labels = [" "] * len(irfs_list)
    n_var = len(variables)
    fig, ax = plt.subplots(1, n_var, figsize=figsize, sharex=True)
    for i in range(n_var):
        # plot all irfs
        for j, irf in enumerate(irfs_list):
            ax[i].plot(100 * irf[variables[i]][:50], label=labels[j])
        ax[i].set_title(variables[i])
        ax[i].set_xlabel(r"$t$")
        if i==0:
            ax[i].set_ylabel(ylabel)
        ax[i].legend()
    plt.show()

show_irfs([irfs], ['G', 'Y', 'goods_mkt'])

rho_B = 0.9
# dB = dG[0] * rho_B ** np.arange(T)
dB = np.cumsum(dG) * rho_B ** np.arange(T)
shocks_B = {'G': dG, 'B': dB}

irfs_B = ha.solve_impulse_linear(ss, unknowns_td, targets_td, shocks_B)

show_irfs([irfs, irfs_B], ['G', 'Y', 'deficit'], labels=["balanced budget", "deficit financed"])
# %%


