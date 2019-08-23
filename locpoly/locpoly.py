import numpy as np

def locpoly(endog, exog, lam, ixl, ri):
    """
    Local linear regression to estimate E[y | x].

    Parameters
    ----------
    endog : array-like 1d
        The dependent variable
    exog : array-like 2d
        The independent variables, do not include an intercept
    lam : scalar, non-negative
        The local regression bandwidth, using squared exponential weights
    ixl : 0, 1
        If 0, include only main effects, if 1 include main effects
        and pairwise interactions.
    ri : non-negative float
        The ridge penalty parameter

    Returns
    -------
    A function f such that f(x)[0] is the estimated mean for covariate
    vector x.  f has a keyword argument "omit" that contains a list of
    case positions that are omitted when computing the value of f.
    f(x)[1] are the coefficients used to produce the estimated mean,
    these coefficients are applied to covariate vector that have been
    centered on x.
    """

    if ixl == 0:
        # Main effects only
        p = exog.shape[1] + 1
    elif ixl == 1:
        # Main effects and pairwise interactions
        ii0, ii1 = np.tril_indices(exog.shape[1], -1)
        p = exog.shape[1] +1 + len(ii0)

    wexog = np.zeros((exog.shape[0], p))

    def f(x, omit=None):

        # Center the covariates, so the fitted regression mean
        # value is the intercept
        dx = exog - x

        # Get the weights
        di = np.sum(dx**2, 1)
        p = exog.shape[1]
        w = np.exp(-di / (2 * lam))
        w /= w.sum()
        if omit is not None:
            w[omit] = 0
        wr = np.sqrt(w)

        # Set up the design matrix
        wexog[:, 0] = wr # whitened intercept
        wexog[:, 1:p+1] = dx * wr[:, None] # whitened main effects
        if ixl == 1:
            # whitened interactions
            jj = exog.shape[1] +1
            for i, j in zip(ii0, ii1):
                wexog[:, jj] = dx[:, i] *dx[:, j] * wr

        # whitened response
        wendog = endog * wr

        # Use a bit of ridging, but not on the intercept
        e = np.ones(wexog.shape[1])
        e[0] = 0

        # Get the coefficient estimates using ridge regression
        u, s, vt = np.linalg.svd(wexog, 0)
        params = np.dot(vt.T, np.dot(u.T, wendog) * s / (s**2 + ri*e))

        return params[0], params[1:]

    return f

def test():
    n = 1000
    x = np.random.normal(size=(n, 5))
    ey = x[:, 0]**2 - x[:, 1]**2
    y = ey + np.random.normal(size=n)

    f = locpoly(y, x, 0.5, 0, 0)

    yhat = [f(x[i, :])[0] for i in range(x.shape[0])]
    yhat = np.asarray(yhat)
    print(np.corrcoef(yhat, y)[0, 1])
    print(np.corrcoef(ey, y)[0, 1])
