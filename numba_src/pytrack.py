import numpy as np
import math
from numba import prange
from numba import get_num_threads, get_thread_id

def kick_py(dt, dE, voltage, omega, phi, acc_kick):
    for j in range(len(voltage)):
        dE += voltage[j] * np.sin(omega[j] * dt + phi[j])
    dE += acc_kick


def drift_py(dt, dE, T0, length_ratio, beta, energy, alpha_zero,
             alpha_one, alpha_two):

    inv_beta_sq = 1. / beta**2
    inv_ene_sq = 1. / energy**2
    for i in prange(len(dt)):
        beam_delta = np.sqrt(1. +
                             inv_beta_sq * (dE[i]*dE[i] * inv_ene_sq + 2 * dE[i]/energy)) - 1.
        dt[i] += T0 * length_ratio * (
            (1 + alpha_zero * beam_delta +
             alpha_one * (beam_delta*beam_delta) +
             alpha_two * beam_delta*beam_delta*beam_delta) *
            (1. + dE[i]/energy) / (1 + beam_delta) - 1)


def histogram_py(dt, profile, cut_left, cut_right):

    profile[:] = np.histogram(dt, bins=len(profile),
                              range=(cut_left, cut_right))[0]


def histogram_v1_py(dt, profile, cut_left, cut_right):
    profile *= 0

    profile_len = len(profile)
    inv_bin_width = profile_len / (cut_right - cut_left)
    for i in range(len(dt)):
        tbin = int(np.floor((dt[i] - cut_left) * inv_bin_width))
        if tbin >= 0 and tbin < profile_len:
            profile[tbin] += 1.0


def histogram_v2_py(dt, profile, cut_left, cut_right):
    profile[:] = 0.0

    profile_len = len(profile)
    inv_bin_width = profile_len / (cut_right - cut_left)
    target_bin = np.empty(len(dt), dtype=np.int32)
    
    for i in prange(len(dt)):
        target_bin[i] = int(np.floor((dt[i] - cut_left) * inv_bin_width))

    for tbin in target_bin:
        if tbin >= 0 and tbin < profile_len:
            profile[tbin] += 1.0



def histogram_v3_py(dt, profile, cut_left, cut_right):

    n_slices = len(profile)
    inv_bin_width = n_slices / (cut_right - cut_left)
    n_threads = get_num_threads()

    local_profile = np.zeros((n_threads, n_slices),
                                                dtype=np.int32)

    for i in prange(len(dt)):
        target_bin = math.floor((dt[i] - cut_left) * inv_bin_width)
        if target_bin >= 0 and target_bin < n_slices:
            local_profile[get_thread_id(), target_bin] += 1

    for i in prange(n_slices):
        profile[i] = 0.0
        for j in range(n_threads):
            profile[i] += local_profile[j, i]


def histogram_v4_py(dt, profile, cut_left, cut_right):

    n_slices = len(profile)
    n_parts = len(dt)

    inv_bin_width = n_slices / (cut_right - cut_left)
    n_threads = get_num_threads()

    local_profile = np.zeros((n_threads, n_slices),
                                                dtype=np.int32)

    STEP = 512
    
    total_steps = math.ceil(n_parts / STEP)
    for i in prange(total_steps):
        thr_id = get_thread_id()
        start_i = i*STEP
        loop_count = min(STEP, n_parts - start_i)
        target_bin = np.floor((dt[start_i:start_i+loop_count] - cut_left) * inv_bin_width)
        for j in range(loop_count):
            if target_bin[j] >= 0 and target_bin[j] < n_slices:
                local_profile[thr_id, int(target_bin[j])] += 1
    
    for i in prange(n_slices):
        profile[i] = 0.0
        for j in range(n_threads):
            profile[i] += local_profile[j, i]

def histogram_v5_py(dt, profile, cut_left, cut_right):

    n_slices = len(profile)
    n_parts = len(dt)
    inv_bin_width = n_slices / (cut_right - cut_left)
    n_threads = get_num_threads()

    local_profile = np.zeros((n_threads, n_slices),
                                                dtype=np.int32)

    STEP = 512
    
    local_target_bin = np.empty((n_threads, STEP), dtype=np.int32)

    total_steps = math.ceil(n_parts / STEP)
    for i in prange(total_steps):
        thr_id = get_thread_id()
        start_i = i*STEP
        loop_count = min(STEP, n_parts - start_i)
        local_target_bin[thr_id][:loop_count] = np.floor((dt[start_i:start_i+loop_count] - cut_left) * inv_bin_width)
        for j in range(loop_count):
            if local_target_bin[thr_id][j] >= 0 and local_target_bin[thr_id][j] < n_slices:
                local_profile[thr_id, local_target_bin[thr_id][j]] += 1
    
    for i in prange(n_slices):
        profile[i] = 0.0
        for j in range(n_threads):
            profile[i] += local_profile[j, i]

def linear_interp_kick_py(dt, dE, voltage,
                          bin_centers, charge,
                          acc_kick):
    n_slices = len(bin_centers)
    inv_bin_width = (n_slices - 1) / \
        (bin_centers[n_slices - 1] - bin_centers[0])
    helper = np.empty(2 * (n_slices-1), dtype=np.float64)
    for i in prange(n_slices-1):
        helper[2*i] = charge * (voltage[i + 1] - voltage[i]) * inv_bin_width
        helper[2*i+1] = (charge * voltage[i] - bin_centers[i]
                         * helper[2*i]) + acc_kick

    for i in prange(len(dt)):
        fbin = int(np.floor((dt[i]-bin_centers[0])*inv_bin_width))
        if (fbin >= 0) and (fbin < n_slices - 1):
            dE[i] += dt[i] * helper[2*fbin] + helper[2*fbin+1]


def kick_numpy(dt, dE, voltage, omega, phi, acc_kick):
    for j in range(len(voltage)):
        dE += voltage[j] * np.sin(omega[j] * dt + phi[j])
    dE += acc_kick


def drift_numpy(dt, dE, T0, length_ratio, beta, energy, alpha_zero,
                alpha_one, alpha_two):

    inv_beta_sq = 1. / beta**2
    inv_ene_sq = 1. / energy**2
    beam_delta = np.sqrt(1. +
                         inv_beta_sq * (dE*dE * inv_ene_sq + 2 * dE/energy)) - 1.
    dt += T0 * length_ratio * (
        (1 + alpha_zero * beam_delta +
            alpha_one * (beam_delta*beam_delta) +
            alpha_two * beam_delta*beam_delta*beam_delta) *
        (1. + dE/energy) / (1 + beam_delta) - 1)


def histogram_numpy(dt, profile, cut_left, cut_right):
    n_slices = len(profile)
    inv_bin_width = n_slices / (cut_right - cut_left)
    profile[:], _ = np.histogram(
        (dt-cut_left)*inv_bin_width, bins=n_slices, range=(0, n_slices))


def linear_interp_kick_numpy(dt, dE, voltage,
                             bin_centers, charge,
                             acc_kick):
    n_slices = len(bin_centers)
    inv_bin_width = (n_slices - 1) / \
        (bin_centers[n_slices - 1] - bin_centers[0])

    fbin = np.floor((dt-bin_centers[0])*inv_bin_width).astype(np.int32)

    helper1 = charge * (voltage[1:] - voltage[:-1]) * inv_bin_width
    helper2 = (charge * voltage[:-1] - bin_centers[:-1] * helper1) + acc_kick

    for i in range(len(dt)):
        # fbin = int(np.floor((dt[i]-bin_centers[0])*inv_bin_width))
        if (fbin[i] >= 0) and (fbin[i] < n_slices - 1):
            dE[i] += dt[i] * helper1[fbin[i]] + helper2[fbin[i]]
