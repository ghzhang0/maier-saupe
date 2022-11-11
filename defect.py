import numpy as np
from typing import List
from math import *
import matplotlib
import matplotlib.pyplot as plt
import maiersaupe as ms
cmap = matplotlib.cm.get_cmap('viridis_r')

def plot_defect(m0:np.ndarray, cone: ms.Cone, p: int, ax: matplotlib.axes.Axes,
                plot: bool = True) -> np.ndarray:
    '''
    * Calculate and return \int_0^{2 \pi} \nabla \vartheta dl centered about 
    each lattice site.  
    * Plot 1. p-atic texture, and 2. lattice sites colored by vorticity. (Gray: 
    edge sites. Yellow: normal bulk sites without defects. Blue: bulk site with 
    defect.)
    '''

    N = len(m0)
    vorticity = np.zeros(N)
    hx = 0.35 # Size of p-atic molecules
    for n in range(N):

        xs = [cone.lattice.real[n] + hx*np.cos(m0[n]+2*pi/p*i)
              for i in range(p)]
        ys = [cone.lattice.imag[n] + hx*np.sin(m0[n]+2*pi/p*i)
              for i in range(p)]

        if n == cone.ap_ind: # Leave out apex site 
            rgba = 'white'
        # Leave out apex's nearest neighbors (apex orientation not well-defined)
        elif np.abs(cone.lattice[n])< 1.1: 
            rgba = cmap(0)
        elif n in cone.bulk_indices:        
            nn_ind = np.where(cone.A[n]>0.5)[0] # Find neighbors of n-th site
            nn_inds = []
            for m in nn_ind: # Eliminate self
                if m != n:
                    nn_inds.append(m)
            if len(nn_inds)==0:
                continue
            nn_inds = np.array(nn_inds)
            # Sort neighbors in CCW direction relative to center site
            angles = np.angle(cone.lattice[nn_inds] - cone.lattice[n])%(2*pi)
            nang_inds = nn_inds[np.argsort(angles)]
            # Modulo all orietation angles by 2 pi / p
            nn_m0 = m0[nang_inds] % (2 * pi / p)
            # Construct vector of (\nabla \vartheta)_i at all neighboring sites
            diffs = [] 
            for i in range(len(nn_m0)-1):
                # Find all possible interpolations from the angle of one
                # orientation vector to its neighbor's
                poss = [nn_m0[i+1] - nn_m0[i] + k*2*pi/p for k
                        in np.arange(-int(p/2)-1, int(p/2)+2)]
                # Pick the interpolation that minimizes the difference between 
                # the two neighboring angles
                dth = poss[np.argsort(np.abs(poss))[0]]
                diffs.append(dth) 
            # Interpolate between the first and the last neighboring sites
            poss = [nn_m0[0] - nn_m0[-1] + k*2*pi/p for k
                    in np.arange(-int(p/2)-1, int(p/2)+2)]
            dth = poss[np.argsort(np.abs(poss))[0]]
            diffs.append(dth) 
            # Summing up \nabla \vartheta
            nns = np.sum(diffs)/(2*pi/p)
            vorticity[n] = nns
            rgba = cmap(nns)
        else: # edge sites (fixed) are colored gray
            rgba = 'lightgray' 

        if plot:
            if p > 2:
                ax.fill(xs, ys, color = rgba)
            else:
                ax.plot(cone.lattice.real[n], cone.lattice.imag[n],
                        'o', color = rgba)

    plt.tight_layout()
    return vorticity

def cone_plot(m0: np.ndarray, cone: ms.Cone, p: int, plot = True,
              lim: float = 14) -> np.ndarray:
    '''Plot and identify defects in p-atic textures on a cone.'''
    fig, ax = plt.subplots(2)
    fig.set_figwidth(5)
    fig.set_figheight(10)

    for i in range(p):
        ax[0].quiver(cone.lattice.real, cone.lattice.imag, 
                     np.cos(m0+i*2*pi/p), np.sin(m0+i*2*pi/p), 
                     headwidth = 0.5, scale = 35, pivot = 'middle')
    vorticity = plot_defect(m0, cone, p, ax[1])

    for i in range(2):
        ax[i].axis('Equal')
        ax[i].set_xticks([])
        ax[i].set_xlim([-lim, lim])
        ax[i].set_ylim([-lim, lim])
        ax[i].set_yticks([])
    
    return vorticity


def hyp_plot(m0: np.ndarray, hyp: ms.Hyperbolic, p: int,
             plot: bool = True, lim: float = 14) -> List[np.ndarray]:
    '''Plot and identify defects in p-atic textures on a hyperbolic cone.'''
    fig, ax = plt.subplots(2, 2)
    fig.set_figwidth(10)
    fig.set_figheight(10)

    for i in range(p):
        ax[0, 0].quiver(hyp.top.lattice.real, hyp.top.lattice.imag,
                        np.cos(m0[:len(hyp.top.lattice)]+i*2*pi/p),
                        np.sin(m0[:len(hyp.top.lattice)]+i*2*pi/p),
                        headwidth = 0.5, scale = 35, pivot = 'middle')
        ax[1, 0].quiver(hyp.bottom.lattice.real[:-hyp.r], 
                        hyp.bottom.lattice.imag[:-hyp.r], 
                        np.cos(m0[len(hyp.top.lattice):]+i*2*pi/p)[:-hyp.r],
                        np.sin(m0[len(hyp.top.lattice):]+i*2*pi/p)[:-hyp.r],
                        headwidth = 0.5, scale = 35, pivot = 'middle')
    
    vort_top = plot_defect(m0[:len(hyp.top.lattice)], hyp.top,
                           p, ax[0, 1], plot = plot)
    vort_seam = plot_hyperbolic_seam(m0, hyp, p, ax[0, 1], plot = plot)
    vort_bottom = plot_defect(m0[len(hyp.top.lattice):], hyp.bottom,
                              p, ax[1, 1], plot = plot)

    for i in range(2):
        for j in range(2):
            ax[i, j].plot([0], [0], 's', markersize = 10, color = 'white')
            ax[i, j].plot([0, lim], [-1/2, -1/2], '--', color = 'gray',
                          zorder = -10)
            ax[i, j].axis('Equal')
            ax[i, j].set_xticks([])
            ax[i, j].set_xlim([-lim, lim])
            ax[i, j].set_ylim([-lim, lim])
            ax[i, j].set_yticks([])
    
    return [vort_top + vort_seam, vort_bottom]


def plot_hyperbolic_seam(m0: np.ndarray, hyp: ms.Hyperbolic, p: int,
                         ax: matplotlib.axes.Axes,
                         plot: bool = True) -> np.ndarray:
    '''Identify defects in the seam between the top and bottom layers of an 
    unrolled hyperbolic cone.'''
    m = len(hyp.top.lattice)
    seam_inds = list(hyp.top.start_seam_inds)
    seam_patch_inds = seam_inds + list(
        np.where(((np.abs(hyp.top.lattice.imag) > 0.5) & 
                  (np.abs(hyp.top.lattice.imag) < 1.5) & 
                  (hyp.top.lattice.real > 0)))[0]) + list(
                    m + np.where(((hyp.bottom.lattice.imag < -0.5) &
                                  (hyp.bottom.lattice.imag > -1.5) &
                                  (hyp.bottom.lattice.real > 0)))[0])
    
    A = hyp.A
    for n in seam_inds:
        for i in seam_patch_inds:
            if i != n and abs(hyp.lattice[n] - hyp.lattice[i]) < 1.5:
                A[n, i] = 1

    vorticity = np.zeros(m)
    hx = 0.35
    for n in seam_inds:

        xs = [hyp.lattice.real[n] + hx*np.cos(m0[n]+2*pi/p*i) for i in range(p)]
        ys = [hyp.lattice.imag[n] + hx*np.sin(m0[n]+2*pi/p*i) for i in range(p)]

        # Leave out apex's nearest neighbors (apex orientation not well-defined)
        if (np.abs(hyp.lattice[n]) < 1.1
            or np.abs(hyp.lattice[n]) > (hyp.r - 1.1)):
            rgba = cmap(0)
            trans = 0
        else:        
            nn_ind = np.where(A[n]>0.5)[0] # Find neighbors of the n-th site
            nn_inds = []
            for m in nn_ind:               # Eliminate self
                if m != n:
                    nn_inds.append(m)
            if len(nn_inds)==0:
                continue
            nn_inds = np.array(nn_inds)
            # Sort neighbors in CCW direction relative to center site
            angles = np.angle(hyp.lattice[nn_inds] - hyp.lattice[n])%(2*pi)
            nang_inds = nn_inds[np.argsort(angles)]
            # Modulo all orietation angles by 2 pi / p
            nn_m0 = m0[nang_inds]%(2*pi/p)
            diffs = [] 
            for i in range(len(nn_m0)-1):
                poss = [nn_m0[i+1] - nn_m0[i] + k*2*pi/p for k
                        in np.arange(-int(p/2)-1, int(p/2)+2)]
                dth = poss[np.argsort(np.abs(poss))[0]]
                diffs.append(dth) 
            poss = [nn_m0[0] - nn_m0[-1] + k*2*pi/p for k
                    in np.arange(-int(p/2)-1, int(p/2)+2)]
            dth = poss[np.argsort(np.abs(poss))[0]]
            diffs.append(dth) 
            nns = np.sum(diffs)/(2*pi/p)
            vorticity[n] = nns
            rgba = cmap(nns)
            trans = 1

        if plot:
            if p > 2:
                ax.fill(xs, ys, color = rgba)
            else:
                ax.plot(hyp.lattice.real[n], hyp.lattice.imag[n], 'o',
                        color = rgba, alpha = trans)

    plt.tight_layout()
    return vorticity

def hyp_lattice_plot(hyp: ms.Hyperbolic, lim: float = 12):
    '''Plot the unrolled lattice of a hyperbolic cone.'''
    fig, ax = plt.subplots(2, 1)

    fig.set_figwidth(5)
    fig.set_figheight(10)

    ax[0].scatter(hyp.top.lattice.real, hyp.top.lattice.imag,
                  s = 10, c = 'gray')
    ax[1].scatter(hyp.bottom.lattice.real[:-hyp.r], 
                hyp.bottom.lattice.imag[:-hyp.r], s = 10, c = 'gray')
    ax[0].plot([-0.5, lim * np.cos(hyp.alpha)-0.5],
               [0, lim * np.sin(hyp.alpha)], '--', color = 'blue', zorder = 10)
    ax[0].plot([0, lim], [-1/2, -1/2], '--', color = 'red', zorder = 10)
    ax[1].plot([0, lim], [-0.375, -0.375], '--', color = 'blue', zorder = 10)
    ax[1].plot([0, lim], [-0.6, -0.6], '--', color = 'red', zorder = 10)
    for i in range(2):
        ax[i].plot([0], [0], 's', markersize = 10, color = 'white')
        ax[i].axis('Equal')
        ax[i].set_xticks([])
        ax[i].set_xlim([-lim, lim])
        ax[i].set_ylim([-lim, lim])
        ax[i].set_yticks([])