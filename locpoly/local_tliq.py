import pandas as pd
import numpy as np
from locpoly import locpoly
from get_data import dx, vnx, mnx

# 0 gives main effects only, 1 also includes interactions
ixl = 0

# a bit of ridging helps
ri = 0.0001

def lpreg_lco_cv(yvec, xmat, bw, clust, nbag=10, nrep=200):
    """
    Estimate the leave-cluster-out RMSE.

    Parameters
    ----------
    yvec : array-like
        The dependent variable (y)
    xmat : array-like
        The independent variables (x), do not include an intercept.
    bw : array-like
        The bandwidths, to be adjusted for sample size.
    clust : integer
        The cluster to be omitted
    nbag : integer
        The number of bagging iterations to stabilize the bandwidth
        selection.
    nrep : integer
        The number of random test-set splits to stabilize the MSE
        estimation.

    Returns
    -------
    The estimated RMSE for the given cluster.

    Notes
    -----
    A large values of nbag is expensive for computation, but increasing nrep
    is cheap.
    """

    # Drop one cluster when training
    omit0 = np.flatnonzero(dx.cluster_id == clust)
    keep = np.flatnonzero(dx.cluster_id != clust)

    xmat = xmat.copy()
    xkeep = xmat[keep, :]
    xmn = xkeep.mean(0)
    xsd = xkeep.std(0)
    xmat -= xmn
    xmat /= xsd

    # The dimension
    d = xmat.shape[1]

    # Pre-compute the errors using the whole training set at
    # each bandwidth
    pce = np.zeros((len(bw), len(omit0)))
    fxa = len(keep) ** (1 / (d + 4)) # bandwidth multiplier
    for kw, w in enumerate(bw):

        # Train on half of the training set
        f = locpoly(yvec[keep], xmat[keep, :], w/fxa, ixl, ri)

        # Evaluate on the test set
        yhat = [f(xmat[i, :], None)[0] for i in omit0]
        yhat = np.asarray(yhat)
        resid = yvec[omit0] - yhat
        pce[kw, :] = resid**2

    # Bagging iterations to select the bandwidth
    lbw = np.zeros((nbag, len(bw), len(omit0)))
    for k in range(nbag):

        # Train on half the training set.
        ii = np.random.permutation(len(keep))
        keep1 = keep[ii[0:len(ii)//2]]

        # The bandwidth multiplier for half of the training set
        fx = len(keep1) ** (1 / (d + 4))

        for kw, w in enumerate(bw):

            # Train on half of the training set
            f = locpoly(yvec[keep1], xmat[keep1, :], w/fx, ixl, ri)

            # Evaluate on the test set
            yhat = [f(xmat[i, :], None)[0] for i in omit0]
            yhat = np.asarray(yhat)
            resid = yvec[omit0] - yhat
            lbw[k, kw, :] = resid**2

    lbw = np.asarray(lbw)

    mse = 0
    for k in range(nrep):

        # Split the test cluster
        ii = np.random.permutation(len(omit0))
        i0 = ii[0:4*len(ii)//5]
        i1 = ii[4*len(ii)//5:]

        # Get the optimal bandwidth using only the first part of the
        # test cluster
        jj = lbw[:, :, i0].mean(0).mean(1).argmin()

        # Evaluate on the second part of the test cluster
        mse += pce[jj, i1].mean()

    rmse = np.sqrt(mse / nrep)

    return rmse

# The data we are working with (melting point/liquidus temperature)
yvec = np.asarray(dx.Tliq_C)
xmat = np.asarray(dx.loc[:, vnx])

lco_range = np.arange(0.2, 1.8, 0.05)

# Drop each cluster, one at a time.  Get the per-cluster and aggregated MSE/RMSE.
mse = 0
n = 0
for clust in range(1, 11):
    rmse1 = lpreg_lco_cv(yvec, xmat, lco_range, clust, nbag=1)
    n0 = (dx.cluster_id == clust).sum()
    print("Cluster %2d: RMSE=%8.3f   n=%4d" % (clust, rmse1, n0))
    mse += rmse1**2 * n0
    n += n0
mse /= n
rmse = np.sqrt(mse)
print("Overall:    RMSE=%8.3f   n=%4d" % (rmse, n))
