import numpy as np
import matplotlib.pylab as plt
from scipy.stats import f as ssf


def ellipse(X, level=0.95, method='deviation', npoints=100):
    """
    X: data, 2D numpy array with 2 columns
    level: confidence level
    method: either 'deviation' (swarning data) or 'error (swarning the mean)'
    npoints: number of points describing the ellipse
    """
    cov_mat = np.cov(X.T)
    dfd = X.shape[0]-1
    dfn = 2
    center = np.apply_along_axis(np.mean, arr=X, axis=0) # np.mean(X, axis=0)
    if method == 'deviation':
        radius = np.sqrt(2 * ssf.ppf(q=level, dfn=dfn, dfd=dfd))
    elif method == 'error':
        radius = np.sqrt(2 * ssf.ppf(q=level, dfn=dfn, dfd=dfd)) / np.sqrt(X.shape[0])
    else:
        raise ValueError("Method should be either 'deviation' or 'error'.")
    angles = (np.arange(0,npoints+1)) * 2 * np.pi/npoints
    circle = np.vstack((np.cos(angles), np.sin(angles))).T
    ellipse = center + (radius * np.dot(circle, np.linalg.cholesky(cov_mat).T).T).T
    return ellipse


def biplot(objects, eigenvectors, eigenvalues=None,
           labels=None, scaling=1, xpc=0, ypc=1,
           group=None, plot_ellipses=False, confidense_level=0.95,
           axis_label='PC', xlim=None, ylim=None):

    """
    Creates a biplot with:

    Parameters:
        objects: 2D numpy array of scores
        eigenvectors: 2D numpy array of loadings
        eigenvalues: 1D numpy array of eigenvalues, necessary to compute correlation biplot_scores
        labels: 1D numpy array or list of labels for loadings
        scaling: either 1 or "distance" for distance biplot, either 2 or "correlation" for correlation biplot
        xpc, ypc: integers, index of the axis to plot. generally xpc=0 and ypc=1 to plot the first and second components
        group: 1D numpy array of categories to color scores
        plot_ellipses: 2D numpy array of error (mean) and deviation (samples) ellipses around groups
        confidense_level: confidense level for the ellipses
        axis_label: string, the text describing the axes
    Returns:
         biplot as matplotlib object
    """


    # select scaling
    if scaling == 1 or scaling == 'distance':
        scores = objects[:, [xpc, ypc]]
        loadings = eigenvectors[[xpc, ypc], :]
    elif scaling == 2 or scaling == 'correlation':
        scores = objects.dot(np.diag(eigenvalues**(-0.5)))[:, [xpc, ypc]]
        loadings = eigenvectors.dot(np.diag(eigenvalues**0.5))
    else:
        raise ValueError("No such scaling")

    # draw the cross
    plt.axvline(0, ls='solid', c='k')
    plt.axhline(0, ls='solid', c='k')

    # draw the ellipses
    if group is not None and plot_ellipses:
        groups = np.unique(group)
        for i in range(len(groups)):
            mean = np.mean(scores[group==groups[i], :], axis=0)
            plt.text(mean[0], mean[1], groups[i],
                     ha='center', va='center', color='k', size=15)
            ell_dev = ellipse(X=scores[group==groups[i], :], level=confidense_level, method='deviation')
            ell_err = ellipse(X=scores[group==groups[i], :], level=confidense_level, method='error')
            plt.fill(ell_err[:,0], ell_err[:,1], alpha=0.6, color='grey')
            plt.fill(ell_dev[:,0], ell_dev[:,1], alpha=0.2, color='grey')

    # plot scores
    if group is None:
        plt.scatter(scores[:,xpc], scores[:,ypc])
    else:
        for i in range(len(np.unique(group))):
            cond = group == np.unique(group)[i]
            plt.plot(scores[cond, 0], scores[cond, 1], 'o')

    # plot loadings
    for i in range(loadings.shape[1]):
        plt.arrow(0, 0, loadings[xpc, i], loadings[ypc, i],
                  color = 'black', head_width=np.ptp(objects)/100)

    # plot loading labels
    if labels is not None:
        for i in range(loadings.shape[1]):
            plt.text(loadings[xpc, i]*1.2, loadings[ypc, i]*1.2, labels[i],
                     color = 'black', ha = 'center', va = 'center', fontsize=20)

    # axis labels
    plt.xlabel(axis_label + str(xpc+1))
    plt.ylabel(axis_label + str(ypc+1))

    # axis limit
    if xlim is None:
        xlim = [np.hstack((loadings[xpc, :], scores[:,xpc])).min(),
                np.hstack((loadings[xpc, :], scores[:,xpc])).max()]
        margin_x = 0.05*(xlim[1]-xlim[0])
        xlim[0]=xlim[0]-margin_x
        xlim[1]=xlim[1]+margin_x
    if ylim is None:
        ylim = [np.hstack((loadings[ypc, :], scores[:,ypc])).min(),
                np.hstack((loadings[ypc, :], scores[:,ypc])).max()]
        margin_y = 0.05*(ylim[1]-ylim[0])
        ylim[0]=ylim[0]-margin_y
        ylim[1]=ylim[1]+margin_y
    plt.xlim(xlim)
    plt.ylim(ylim)


def triplot(objects, eigenvectors, species, eigenvalues=None,
           labels=None, scaling=1, xpc=0, ypc=1,
           axis_label='PC'):
    """
    objects, species and eigenvectors are pandas.DataFrames
    """

    site_scores = objects.iloc[:, [xpc, ypc]]
    species_scores = species.iloc[:, [xpc, ypc]]
    loadings = eigenvectors


    # draw the cross
    plt.axvline(0, ls='solid', c='k')
    plt.axhline(0, ls='solid', c='k')

    # plot scores
    ## sites

    for i in range(site_scores.shape[0]):
        plt.text(x=site_scores.iloc[i,0],
                 y=site_scores.iloc[i,1],
                 s=site_scores.index.values[i],
                 color='black')
    ## species
    for i in range(species_scores.shape[0]):
        plt.text(x=species_scores.iloc[i,0],
                 y=species_scores.iloc[i,1],
                 s=species_scores.index.values[i],
                 color='red')

    # plot loadings
    expand_scores = 3
    margin_score_labels = 0.3
    for i in range(loadings.shape[0]):
        plt.arrow(0, 0,
              loadings.iloc[i,0]*expand_scores,
              loadings.iloc[i,1]*expand_scores,
              color = 'blue', head_width=.1)
        plt.text(x=loadings.iloc[i,0]*(expand_scores + margin_score_labels),
             y=loadings.iloc[i,1]*(expand_scores + margin_score_labels),
             s = loadings.index.values[i],
             color='blue')

    # axis labels
    plt.xlabel(axis_label + str(xpc+1))
    plt.ylabel(axis_label + str(ypc+1))

    #
    # amalgamate all the values to define the limits in X and Y
    allX = np.hstack((species_scores.iloc[:,0],
                      site_scores.iloc[:,0],
                      loadings.iloc[:,0]*(expand_scores + margin_score_labels)))
    allY = np.hstack((species_scores.iloc[:,1],
                      site_scores.iloc[:,1],
                      loadings.iloc[:,1]*(expand_scores + margin_score_labels)))
    margin_plot = 0.5
    plt.xlim([np.min(allX)-margin_plot, np.max(allX)+margin_plot])
    plt.ylim([np.min(allY)-margin_plot, np.max(allY)+margin_plot])
