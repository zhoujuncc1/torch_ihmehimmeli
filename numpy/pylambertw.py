import numpy as np

M_E = 1e5  # unknow yet
# Minimum argument for the main branch of the Lambert W function.
kMinLambertArg = -1.0 / M_E
# Maximum argument for which gsl_sf_lambert_W0 produces a valid result.
kMaxLambertArg = 1.7976131e+308


kNearBranchCutoff = -0.3235
kE = 2.718281828459045


kReciprocalE = 0.36787944117
kDesiredAbsoluteDifference = 1e-3
kNumMaxIters = 10

def LambertW0InitialGuess(x):
    # Sqrt approximation near branch cutoff.
    if x < kNearBranchCutoff:
        return -1.0 + sqrt(2.0 * (1 + kE * x))
    # Taylor series between [-1/e and 1/e].
    if x > kNearBranchCutoff and x < -kNearBranchCutoff:
        return x * (1 + x * (-1 + x * (3.0 / 2.0 - 8.0 / 3.0 * x)))

    # Series of piecewise linear approximations.
    if x < 0.6:
        return 0.23675531078855933 + (x - 0.3) * 0.5493610866617109
    if x < 0.8999999999999999:
        return 0.4015636367870726 + (x - 0.6) * 0.4275644294878729
    if x < 1.2:
        return 0.5298329656334344 + (x - 0.8999999999999999) * 0.3524368357714513
    if x < 1.5:
        return 0.6355640163648698 + (x - 1.2) * 0.30099113800452154
    if x < 1.8:
        return 0.7258613577662263 + (x - 1.5) * 0.2633490154764343
    if x < 2.0999999999999996:
        return 0.8048660624091566 + (x - 1.8) * 0.2345089875713013
    if x < 2.4:
        return 0.8752187586805469 + (x - 2.0999999999999996) * 0.2116494532726034
    if x < 2.6999999999999997:
        return 0.938713594662328 + (x - 2.4) * 0.19305046534383152
    if x < 2.9999999999999996:
        return 0.9966287342654774 + (x - 2.6999999999999997) * 0.17760053566187495

    # Asymptotic approximation.
    l = np.log(x)
    ll = np.log(l)
    return l - ll + ll / l


def LambertW0InitialGuessNp(x):
    # Sqrt approximation near branch cutoff.
    if x < kNearBranchCutoff:
        return -1.0 + sqrt(2.0 * (1 + kE * x))
    # Taylor series between [-1/e and 1/e].
    if x > kNearBranchCutoff and x < -kNearBranchCutoff:
        return x * (1 + x * (-1 + x * (3.0 / 2.0 - 8.0 / 3.0 * x)))
    x1 = np.where(x<kNearBranchCutoff, -1.0 + np.sqrt(2.0 * (1 + kE * x)), x)
    x2 = np.where(x>kNearBranchCutoff and x < -kNearBranchCutoff, x * (1 + x * (-1 + x * (3.0 / 2.0 - 8.0 / 3.0 * x))), x1)
    # Series of piecewise linear approximations.
    if x < 0.6:
        return 0.23675531078855933 + (x - 0.3) * 0.5493610866617109
    if x < 0.8999999999999999:
        return 0.4015636367870726 + (x - 0.6) * 0.4275644294878729
    if x < 1.2:
        return 0.5298329656334344 + (x - 0.8999999999999999) * 0.3524368357714513
    if x < 1.5:
        return 0.6355640163648698 + (x - 1.2) * 0.30099113800452154
    if x < 1.8:
        return 0.7258613577662263 + (x - 1.5) * 0.2633490154764343
    if x < 2.0999999999999996:
        return 0.8048660624091566 + (x - 1.8) * 0.2345089875713013
    if x < 2.4:
        return 0.8752187586805469 + (x - 2.0999999999999996) * 0.2116494532726034
    if x < 2.6999999999999997:
        return 0.938713594662328 + (x - 2.4) * 0.19305046534383152
    if x < 2.9999999999999996:
        return 0.9966287342654774 + (x - 2.6999999999999997) * 0.17760053566187495

    # Asymptotic approximation.
    l = np.log(x)
    ll = np.log(l)
    return l - ll + ll / l


def LambertW0(x):
    if x <= -kReciprocalE:
        return None, False
    if x == 0.0:
        return 0, True
    if x == -kReciprocalE:
        return -1.0, True

    # Current guess.
    w_n = LambertW0InitialGuess(x)
    have_convergence = False

    # Fritsch iteration.
    for i in range(kNumMaxIters):
        z_n = np.log(x / w_n) - w_n
        q_n = 2.0 * (1.0 + w_n) * (1.0 + w_n + 2.0 / 3.0 * z_n)
        e_n = (z_n / (1.0 + w_n)) * ((q_n - z_n) / (q_n - 2.0 * z_n))
        w_n *= (1.0 + e_n)
        # Done this way as the log is the expensive part above.
        if abs(z_n) < kDesiredAbsoluteDifference:
            have_convergence = True
            break
    return w_n, have_convergence
