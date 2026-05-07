import numpy as np
from matplotlib import pyplot as plt
import cvxpy as cp
from tqdm import tqdm

from defdap.utils import subplot_grid

def calc_schmid_tensors(slip_systems, ori):
    """
    Calculate in-plane components of Schmid tensor for a set of slip systems
    rotated in the sample reference frame.

    Parameters
    ----------
    slip_systems : list[crystal.SlipSystem]
        List of slip systems.
    ori : Quat
        Orientation of the grain.
    
    Returns
    -------
    np.ndarray of shape (4, n)
        Flattened in-plane components of Schmid tensor

    """
    return np.array([
        np.outer(
            ori.conjugate.transform_vector(ss.slip_dir), 
            ori.conjugate.transform_vector(ss.slip_plane)
        )[0:2, 0:2].flatten() 
        for ss in slip_systems
    ]).T

def run_sslip(def_grad, ori, slip_systems, threshold=0.01):
    """
    Calculate slip amplitudes by minimizing L1 norm of slip systems.
    Uses convex optimization (CVXPY) to find slip amplitudes that minimize
    the sum of absolute values subject to a constraint on the L2 norm of
    the residual between theoretical and experimental displacement gradients.

    Parameters
    ----------
    def_grad : np.ndarray of shape (2, 2, n)
        Deformation gradient tensor at each of n points in the grain.
    ori : Quat
        Reference orientation (grain orientation) used to rotate slip systems
        to the sample frame.
    slip_systems : list[crystal.SlipSystem]
        List of slip systems to optimize over (length n_ss).
    threshold : float, optional
        Maximum tolerance for the L2 norm of residual displacement gradient,
        by default 0.01.

    Returns
    -------
    np.ndarray of shape (n_ss, n)
        Slip amplitudes for each slip system at each grain point.

    """
    schmid_tensors = calc_schmid_tensors(
        slip_systems, 
        ori
    )
    
    n = def_grad.shape[-1]
    n_ss = len(slip_systems)
    slip_amplitudes = np.empty((n_ss, n))

    # Convert deformation gradient to displacement gradient
    disp_grad = def_grad - np.eye(2)[:, :, np.newaxis]
    disp_grad[0, 1] *= -1
    disp_grad[1, 0] *= -1
    disp_grad = disp_grad.reshape(4, -1)

    # Setting up the variables for the optimisation
    x = cp.Variable(n_ss)
    disp_grad_i = cp.Parameter((4, ))
    constraints = [cp.norm(schmid_tensors @ x - disp_grad_i, 2) <= threshold]
    objective = cp.Minimize(cp.sum(cp.abs(x)))
    prob = cp.Problem(objective, constraints)

    # Solve for each point
    for i in tqdm(range(disp_grad.shape[1])):
        disp_grad_i.value = disp_grad[:, i]
        prob.solve()
        slip_amplitudes[:, i] = x.value
           
    total_slip_sys_ampl = np.sum(np.abs(slip_amplitudes), axis=1)
    total = total_slip_sys_ampl.sum()
    print('SSLIP complete. Slip system amplitudes:\n')
    print('Slip System\tAmplitude\t(Percentage)')
    for slip_amp, ss in zip(total_slip_sys_ampl, slip_systems):
        print(f'{ss.slip_plane_label}, {ss.slip_dir_label} \t{slip_amp:.2f}'
              f'    \t({slip_amp / total * 100:.1f} %)')

    return slip_amplitudes

def plot_sslip_all(
        dic_grain, 
        slip_amplitudes, 
        absolute_amplitudes=True,
        slip_systems=None, 
        slip_traces=None, 
        vmax=None, 
        layout=None
    ):
    """
    Plot SSLIP results for all slip systems in a grid layout.
    Each subplot shows slip amplitude distribution as a heatmap with slip trace
    overlaid.
    
    Parameters
    ----------
    dic_grain : defdap.hrdic.Grain
        DIC grain object containing the grain geometry and data.
    slip_amplitudes : np.ndarray
        Array of shape (n_slip_systems, n_grain_points) containing the 
        calculated slip amplitudes for each slip system at each point in 
        the grain.
    absolute_amplitudes : bool, optional
        If True, plot absolute values. If False, plot signed values.
        Default is True.
    slip_systems : list[crystal.SlipSystem], optional
        List of slip system objects. If None, computed from grain's EBSD data.
    slip_traces : list or np.ndarray, optional
        Slip trace angles in degrees (counter-clockwise from vertical) for each
        slip system. If None, computed from grain's EBSD data.
    vmax : float, optional
        Maximum value for the color scale. If None, set to max of slip_amplitudes.
    layout : tuple, optional
        Subplot grid layout (rows, cols). If None, computed to form a compact grid.

    """

    if slip_systems is None:
        dic_grain.ebsd_grain.calc_average_ori()
        slip_systems = sum(dic_grain.ebsd_grain.phase.slip_systems, start=[])
    if slip_traces is None:
        slip_traces = []
        for i, group in enumerate(dic_grain.ebsd_grain.phase.slip_systems):
                for ss in group:
                    slip_traces.append(dic_grain.ebsd_grain.slip_traces[i])

    if vmax is None:
        vmax = np.max(np.abs(slip_amplitudes))
    if layout is None:
        layout = subplot_grid(len(slip_systems))

    fig, axes = plt.subplots(*layout, figsize=(8, 8), sharex=True, sharey=True,
        constrained_layout=True)
    axes = axes.ravel()

    total_slip_sys_ampl = np.sum(np.abs(slip_amplitudes), axis=1)
    total = total_slip_sys_ampl.sum()

    for ax, slip_amp, ss, t in zip(axes, slip_amplitudes, slip_systems, slip_traces):
        perc = np.sum(np.abs(slip_amp)) / total * 100
        ax.set_title(str(ss) + '\n({0:.1f} %)'.format(perc))
        
        if absolute_amplitudes == True:
            slip_amp = np.abs(slip_amp)
            cmap = 'viridis'
            vmin = 0 
            label = 'Absolute Slip Amplitude'
            trace_colour = 'r'
        elif absolute_amplitudes == False:
            cmap = 'seismic'
            vmin = -vmax            
            label = 'Slip Amplitude'
            trace_colour = 'k'

        plot = dic_grain.plot_grain_data(grain_data=slip_amp, 
            ax=ax, fig=fig, vmin=vmin, vmax=vmax, cmap=cmap)
        
        plot.add_traces(angles=[t], colours=[trace_colour], linewidths=[2.0])

    fig.colorbar(plot.img_layers[0], label = label, 
                    ax=axes, shrink=0.6, location='bottom', pad=0.04)

    
