"""
Estimate the generalization error rate for kernel regression under extrapolation,
using cross-validation with excluded spheres around the test points.
"""

import pandas as pd
import numpy as np
from locpoly import locpoly
from get_data import dx, vnx, mnx

# 0 gives main effects only, 1 also includes interactions
ixl = 0

# a bit of ridging helps
ri = 0.0001

def lpreg_lno_cv(yvec, xmat, bw, radius, nbag=10, ntest=10):
    """
    Estimate the leave-neighborhood-out RMSE.

    Parameters
    ----------
    yvec : array-like
        The dependent variable (y)
    xmat : array-like
        The independent variables (x), do not include an intercept.
    bw : array-like
        The bandwidths, to be adjusted for sample size.
    radius : float64
        The radius for the excluded ball around each point
    nbag : integer
        The number of bagging iterations to stabilize the bandwidth
        selection.
    ntest : integer
        The number of cases held out of the training set for tuning.

    Returns
    -------
    The estimated RMSE for the given cluster, and the average number of points
    in the excluded ball.
    """

    # The sample size and dimension
    n, d = xmat.shape

    mse = 0
    n_mask = []

    for ir in range(n):

        # Mask out the test case and its neighbors
        dist = xmat - xmat[ir, :]
        dist = np.sqrt((dist**2).sum(1))
        n_mask.append(np.sum(dist <= radius))
        keep = np.flatnonzero(dist > radius)
        omit = np.flatnonzero(dist <= radius)
        if keep.size < 100:
            # radius is too big
            1/0

        # Bagging iterations to select the bandwidth
        lbw = np.zeros((nbag, len(bw), ntest))
        xkeep = xmat[keep, :]
        ykeep = yvec[keep]

        for k in range(nbag):

            # Choose ntest points to omit at random, then also omit
            # their radius-neighborhoods.
            omit1 = np.random.permutation(len(keep))[0:ntest]
            keep1 = np.ones(len(keep), dtype=np.bool)
            for j in omit1:
                dist = xkeep - xkeep[j, :]
                dist = np.sqrt((dist**2).sum(1))
                keep1[dist <= radius] = False

            if keep1.sum() < 100:
                # radius is too big
                1/0

            # Bagging subsampling
            ii = np.random.permutation(len(keep1))
            keep1[ii[0:len(ii)//2]] = False
            keep1 = np.flatnonzero(keep1)

            # The bandwidth multiplier for the training set
            fx = len(keep1) ** (1 / (d + 4))

            for kw, w in enumerate(bw):

                # Train on half of the training set
                f = locpoly(ykeep[keep1], xkeep[keep1, :], w/fx, ixl, ri)

                # Evaluate on the test set
                yhat = [f(xkeep[i, :], None)[0] for i in omit1]
                yhat = np.asarray(yhat)
                resid = ykeep[omit1] - yhat
                lbw[k, kw, :] = resid**2

        lbw = np.asarray(lbw)

        # Get the optimal bandwidth using only the first part of the
        # test cluster
        jj = lbw.mean(0).mean(1).argmin()

        # The bandwidth multiplier for half of the training set
        fx = len(keep) ** (1 / (d + 4))

        # Re-train on the whole training set.
        f = locpoly(ykeep, xkeep, bw[jj]/fx, ixl, ri)

        # Evaluate on the held-out case.
        mse += (yvec[ir] - f(xmat[ir, :], None)[0])**2

    rmse = np.sqrt(mse / n)

    return rmse, np.mean(n_mask)

# The data we are working with (melting point/liquidus temperature)
yvec = np.asarray(dx.Tliq_C)
xmat = np.asarray(dx.loc[:, vnx])

# Radii of the balls to exclude around each point when training
radius_range = np.linspace(0.1, 2, 20)

bw_range = np.linspace(0.1, 2, 10)

# Drop ...
for radius in radius_range:
    rmse, n_nbs = lpreg_lno_cv(yvec, xmat, bw_range, radius, nbag=10)
    print("Radius %f: RMSE=%8.3f, neighbors=%d" % (radius, rmse, n_nbs))
