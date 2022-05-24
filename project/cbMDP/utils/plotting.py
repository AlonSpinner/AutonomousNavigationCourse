import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_cov_ellipse(pos, cov, nstd=1, ax=None, facecolor = 'none',edgecolor = 'b' ,  **kwargs):
        #slightly edited from https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
        '''
        Plots an `nstd` sigma error ellipse based on the specified covariance
        matrix (`cov`). Additional keyword arguments are passed on to the 
        ellipse patch artist.

        Parameters
        ----------
            pos : The location of the center of the ellipse. Expects a 2-element
                sequence of [x0, y0].
            cov : The 2x2 covariance matrix to base the ellipse on
            nstd : The radius of the ellipse in numbers of standard deviations.
            ax : The axis that the ellipse will be plotted on. If not provided, we won't plot.
            Additional keyword arguments are pass on to the ellipse patch.

        Returns
        -------
            A matplotlib ellipse artist
        '''
        eigs, vecs = np.linalg.eig(cov)
        theta = np.degrees(np.arctan2(vecs[1,0],vecs[0,0])) #obtain theta from first axis. second axis is just perpendicular to it

        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(eigs)
        ellip = Ellipse(xy=pos, 
                        width=width, 
                        height=height, 
                        angle=theta,
                        facecolor = facecolor, 
                        edgecolor=edgecolor, **kwargs)

        if ax is not None:
            ax.add_patch(ellip)
        
        return ellip

def plot_pose(axes , pose, axis_length: float = 0.1, covariance: np.ndarray = None):
    '''
    TAKEN FROM gtsam.utils.plot AND SLIGHTLY EDITED
    Plot a 2D pose on given axis `axes` with given `axis_length`.

    e - ego
    w - world
    '''
    # get rotation and translation (center)
    Re2w = pose.rotation().matrix() 
    t_w_w2e = pose.translation()

    graphics = []

    x_axis = t_w_w2e + Re2w[:, 0] * axis_length
    line = np.vstack((t_w_w2e,x_axis))
    graphics_line1, = axes.plot(line[:, 0], line[:, 1], 'r-')
    graphics.append(graphics_line1)

    y_axis = t_w_w2e + Re2w[:, 1] * axis_length
    line = np.vstack((t_w_w2e,y_axis))
    graphics_line2, = axes.plot(line[:, 0], line[:, 1], 'g-')
    graphics.append(graphics_line2)


    if covariance is not None:
        graphics_ellip = plot_cov_ellipse(t_w_w2e, covariance[:2,:2], nstd=1, ax=axes, facecolor = 'none', edgecolor = 'k')
        graphics.append(graphics_ellip)

    return graphics

def plot_landmark(axes, loc, cov = None, index = None,
        markerShape = '.', markerColor = 'b', markerSize = 5, textColor = 'k'):
    
    graphics = []
    graphics.append(axes.scatter(loc[0],loc[1], marker = markerShape, c = markerColor, s = markerSize))
    if cov is not None:
        graphics.append(plot_cov_ellipse(loc,cov,nstd = 1,ax = axes,edgecolor = markerColor))
    if index is not None:
        graphics.append(axes.text(loc[0],loc[1],index, color = textColor))

    return graphics

def spawnWorld(xrange = None, yrange = None,type = "world"):
    
    if type == "world":
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('x'); ax.set_ylabel('y'); 
        ax.set_aspect('equal'); ax.grid()

        if xrange is not None: ax.set_xlim(xrange)
        if yrange is not None: ax.set_ylim(yrange)

        return fig, ax
    
    elif type == "world and error":
        fig, (ax1, ax2) = plt.subplots(2,1)
        
        #axes for world
        ax1.set_xlim(xrange); ax1.set_ylim(yrange); 
        ax1.set_xlabel('x'); ax1.set_ylabel('y'); 
        ax1.set_aspect('equal'); ax1.grid()

        #axes for error tracking
        ax2.set_xlabel('time'); ax2.set_ylabel('error'); 
        ax2.grid()

        return fig, (ax1,ax2)