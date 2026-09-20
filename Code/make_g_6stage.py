import numpy as np

from Code.make_g import make_treno as make_treno_4stage


def _interpolate_stage_axis(arr, n_stages):
    old_x = np.linspace(0.0, 1.0, arr.shape[1])
    new_x = np.linspace(0.0, 1.0, n_stages)
    out = np.zeros((arr.shape[0], n_stages, arr.shape[2]), dtype=arr.dtype)
    for p in range(arr.shape[2]):
        out[0, :, p] = np.interp(new_x, old_x, arr[0, :, p])
    return out


def make_treno():
    treno4, U4, L4 = make_treno_4stage()
    n_stages = 6

    treno6 = _interpolate_stage_axis(treno4, n_stages)
    U6 = _interpolate_stage_axis(U4, n_stages)
    L6 = _interpolate_stage_axis(L4, n_stages)

    total_stage = treno4[0, :, 6].sum()
    total_upper = U4[0, :, 6].sum()
    total_lower = L4[0, :, 6].sum()

    treno6[0, :, 6] = np.maximum(treno6[0, :, 6], 1e-3)
    U6[0, :, 6] = np.maximum(U6[0, :, 6], 1e-3)
    L6[0, :, 6] = np.maximum(L6[0, :, 6], 1e-3)

    treno6[0, :, 6] *= total_stage / treno6[0, :, 6].sum()
    U6[0, :, 6] *= total_upper / U6[0, :, 6].sum()
    L6[0, :, 6] *= total_lower / L6[0, :, 6].sum()

    return treno6, U6, L6
