import numpy as np
import SimpleITK as sitk
import scipy.io as sio
from tqdm import tqdm
from numba import jit
from numba import njit, prange, cuda, set_num_threads
from numba import cuda
import numba

l2n = lambda l: np.array(l)
n2l = lambda n: list(n)

set_num_threads(15)
@njit(nopython=True, parallel=True, fastmath= True)
#@cuda.reduce
def make_transducer(ROC, width, dx, Tcenter, Tnormal):

    Tdia = width / 1000
    Troc = ROC / 1000
    pi = 3.141592653589793

    H = Troc - np.sqrt(Troc ** 2 - (0.5 * Tdia) ** 2)
    TP_dis = np.floor(H / dx)
    move_Tcenter = (Tcenter + TP_dis * Tnormal)

    nH = 2000
    nD = 2000
    nA = int(np.floor(0.5 * Tdia / dx) + 5)
    A = np.zeros((2 * nA + 1, 2 * nA + 1, 2 * nA + 1))

    Vs = np.zeros(3)
    Vs[1] = -Tnormal[2]
    Vs[2] = Tnormal[1]

    if np.all(Vs == 0):
        Vs[1] = 0.000000000001
        Vs[2] = 0.000000000001

    Vs = Vs / np.linalg.norm(Vs)
    Vt = np.cross(Tnormal, Vs)

    for i in range(nH):
        R = np.sqrt(Troc ** 2 - (Troc - H + i * H / nH) ** 2)
        for j in range(nD):
            theta = 2 * pi / nD * j
            X = int((nA + 1) + np.round(
                (-(i) * H / nH * Tnormal[0] + R * np.cos(theta) * Vs[0] + R * np.sin(theta) * Vt[0]) / dx))
            Y = int((nA + 1) + np.round(
                (-(i) * H / nH * Tnormal[1] + R * np.cos(theta) * Vs[1] + R * np.sin(theta) * Vt[1]) / dx))
            Z = int((nA + 1) + np.round(
                (-(i) * H / nH * Tnormal[2] + R * np.cos(theta) * Vs[2] + R * np.sin(theta) * Vt[2]) / dx))
            A[X, Y, Z] = 1

    nS = int(np.sum(A))
    Spos = np.zeros((nS, 4))
    idx = np.where(A == 1)

    Spos[:, 2] = idx[0] - (nA + 1) + move_Tcenter[0]
    Spos[:, 1] = idx[1] - (nA + 1) + move_Tcenter[1]
    Spos[:, 0] = idx[2] - (nA + 1) + move_Tcenter[2]

    return Spos


@jit(nopython=True, parallel=True, fastmath= True)
def make_ROI_fast(ROI_idx, p_raw, times, BP_Phase, BP_Amp, BP_step,period):
    for i in range(ROI_idx.shape[0]):
        record = p_raw[ROI_idx[i, 0], ROI_idx[i, 1], ROI_idx[i, 2], :]
        peaks = np.argmax(record)
        TOF = times[peaks]
        Amp = record[peaks]
        BP_Phase[ROI_idx[i, 0], ROI_idx[i, 1], ROI_idx[i, 2]] = 2 * np.pi * ((TOF / period) - np.floor(TOF / period))
        BP_Amp[ROI_idx[i, 0], ROI_idx[i, 1], ROI_idx[i, 2]] = Amp
        BP_step[ROI_idx[i, 0], ROI_idx[i, 1], ROI_idx[i, 2]] = np.floor(TOF / period)

    return BP_Phase, BP_Amp, BP_step


@jit(nopython=True, parallel=True, fastmath= True)
def score_fast(Tcenter, Tnormal,  PHASE, AMP, skullCrop_arr, width, ROC, dx):

    ################################################################################################
    #### make transducer
    ################################################################################################

    Tdia = width / 1000
    Troc = ROC / 1000
    pi = 3.141592653589793

    H = Troc - np.sqrt(Troc ** 2 - (0.5 * Tdia) ** 2)
    TP_dis = np.floor(H / dx)
    move_Tcenter = (Tcenter + TP_dis * Tnormal)

    nH = 1500
    nD = 1500
    nA = int(np.floor(0.5 * Tdia / dx) + 5)
    A = np.zeros((2 * nA + 1, 2 * nA + 1, 2 * nA + 1))

    Vs = np.zeros(3)
    Vs[1] = -Tnormal[2]
    Vs[2] = Tnormal[1]

    Vs = Vs / np.linalg.norm(Vs)
    Vt = np.cross(Tnormal, Vs)
    #print("find point 2", tran_idx)
    for i in range(nH):
        R = np.sqrt(Troc ** 2 - (Troc - H + i * H / nH) ** 2)
        for j in range(nD):
            theta = 2 * pi / nD * j
            X = int((nA + 1) + np.round(
                (-(i) * H / nH * Tnormal[0] + R * np.cos(theta) * Vs[0] + R * np.sin(theta) * Vt[0]) / dx))
            Y = int((nA + 1) + np.round(
                (-(i) * H / nH * Tnormal[1] + R * np.cos(theta) * Vs[1] + R * np.sin(theta) * Vt[1]) / dx))
            Z = int((nA + 1) + np.round(
                (-(i) * H / nH * Tnormal[2] + R * np.cos(theta) * Vs[2] + R * np.sin(theta) * Vt[2]) / dx))
            A[X, Y, Z] = 1

    nS = int(np.sum(A))
    Spos = np.zeros((int(nS), 4))
    idx = np.where(A == 1)

    Spos[:, 2] = (idx[0] - (nA + 1) + move_Tcenter[0])
    Spos[:, 1] = (idx[1] - (nA + 1) + move_Tcenter[1])
    Spos[:, 0] = (idx[2] - (nA + 1) + move_Tcenter[2])

    ################################################################################################
    #### Check ROI or not
    ################################################################################################

    if np.any(Spos[:, 0] >= skullCrop_arr.shape[0])\
            or np.any(Spos[:, 1] >= skullCrop_arr.shape[1])\
            or np.any(Spos[:, 2] >= skullCrop_arr.shape[2])\
            or np.any(Spos < 0):

        score = 0
        return score, Spos

    ################################################################################################
    #### Calculate score
    ################################################################################################
    check = np.zeros(Spos.shape[0])
    phase = np.zeros(Spos.shape[0])
    amp = np.zeros(Spos.shape[0])

    gather_score = []
    for i in range(len(PHASE)):

        BP_Phase = PHASE[i]
        BP_Amp = AMP[i]

        for p in range(Spos.shape[0]):
            check[p] = skullCrop_arr[int(Spos[p, 0]), int(Spos[p, 1]), int(Spos[p, 2])]
            phase[p] = BP_Phase[int(Spos[p, 0]), int(Spos[p, 1]), int(Spos[p, 2])]
            amp[p] = BP_Amp[int(Spos[p, 0]), int(Spos[p, 1]), int(Spos[p, 2])]

        if np.any(amp == 0) or np.any(check > 250):
            score = 0
            return score, Spos

        temp = np.zeros((Spos.shape[0], 2))
        temp[:,0] = np.multiply(amp,  np.cos(phase))
        temp[:,1] = np.multiply(amp,  np.sin(phase))
        temp_sum = np.sum(temp, axis=0)

        score_at1 = np.abs(np.complex(temp_sum[0], temp_sum[1]))
        gather_score.append(score_at1)

    score = np.sum(np.array(gather_score))

    return score, Spos

def neper2db(alpha, y):

    alphaDB = 20*np.log10(np.exp(1))*alpha*pow((2*np.pi*1e6), y)/100

    return alphaDB

def makeSphere(shape, radius, position):
    # assume shape and position are both a 3-tuple of int or float
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    semisizes = (radius,) * 3

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below 1
    result = (arr <= 1.0)
    result = result.astype(int)

    return result

def cordi2idx(image_array, point):

    idx = np.zeros((3))
    for i in range(3):
        idx[i] = np.abs(image_array[i] - point[i]).argmin()

    return idx


