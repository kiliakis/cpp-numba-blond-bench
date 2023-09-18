import ctypes as ct
import numpy as np
import os

cpplib = ct.cdll.LoadLibrary(os.path.dirname(os.path.abspath(__file__)) +'/libtrack.so')

def __getPointer(x):
    return x.ctypes.data_as(ct.c_void_p)

def __getLen(x):
    return ct.c_int(len(x))


class Precision:
    def __init__(self, precision='double'):
        if precision in ['single', 's', '32', 'float32', 'float', 'f']:
            self.str = 'float32'
            self.real_t = np.float32
            self.c_real_t = ct.c_float
            self.complex_t = np.complex64
            self.num = 1
        elif precision in ['double', 'd', '64', 'float64']:
            self.str = 'float64'
            self.real_t = np.float64
            self.c_real_t = ct.c_double
            self.complex_t = np.complex128
            self.num = 2


precision = Precision('double')

def __c_real(x):
    if precision.num == 1:
        return ct.c_float(x)
    else:
        return ct.c_double(x)


def kick_cpp(dt, dE, voltage, omega, phi, acc_kick):
    cpplib.kick(__getPointer(dt),
                __getPointer(dE),
                __getLen(voltage),
                __getPointer(voltage),
                __getPointer(omega),
                __getPointer(phi),
                __getLen(dt),
                __c_real(acc_kick))


def drift_cpp(dt, dE, T0, length_ratio, beta, energy, alpha_zero,
              alpha_one, alpha_two):
    cpplib.drift(__getPointer(dt),
                 __getPointer(dE),
                 __c_real(T0),
                 __c_real(length_ratio),
                 __c_real(alpha_zero),
                 __c_real(alpha_one),
                 __c_real(alpha_two),
                 __c_real(beta),
                 __c_real(energy),
                 __getLen(dt))

def histogram_cpp(dt, profile, cut_left, cut_right):
    # profile does not need to be initialized at 0
    cpplib.histogram(__getPointer(dt),
                     __getPointer(profile),
                     __c_real(cut_left),
                     __c_real(cut_right),
                     __getLen(profile),
                     __getLen(dt))

def histogram_v2_cpp(dt, profile, cut_left, cut_right):
    # profile does not need to be initialized at 0
    cpplib.histogram_v2(__getPointer(dt),
                     __getPointer(profile),
                     __c_real(cut_left),
                     __c_real(cut_right),
                     __getLen(profile),
                     __getLen(dt))

def histogram_v3_cpp(dt, profile, cut_left, cut_right):
    # profile does not need to be initialized at 0
    cpplib.histogram_v3(__getPointer(dt),
                     __getPointer(profile),
                     __c_real(cut_left),
                     __c_real(cut_right),
                     __getLen(profile),
                     __getLen(dt))

def histogram_v4_cpp(dt, profile, cut_left, cut_right):
    # profile does not need to be initialized at 0
    cpplib.histogram_v4(__getPointer(dt),
                     __getPointer(profile),
                     __c_real(cut_left),
                     __c_real(cut_right),
                     __getLen(profile),
                     __getLen(dt))

def histogram_v5_cpp(dt, profile, cut_left, cut_right):
    # profile does not need to be initialized at 0
    cpplib.histogram_v5(__getPointer(dt),
                     __getPointer(profile),
                     __c_real(cut_left),
                     __c_real(cut_right),
                     __getLen(profile),
                     __getLen(dt))


def linear_interp_kick_cpp(dt, dE, voltage,
                       bin_centers, charge,
                       acceleration_kick):

    cpplib.linear_interp_kick(__getPointer(dt),
                             __getPointer(dE),
                             __getPointer(voltage),
                             __getPointer(bin_centers),
                             __c_real(charge),
                             __getLen(bin_centers),
                             __getLen(dt),
                             __c_real(acceleration_kick))

def linear_interp_kick_v2_cpp(dt, dE, voltage,
                       bin_centers, charge,
                       acceleration_kick):

    cpplib.linear_interp_kick_v2(__getPointer(dt),
                             __getPointer(dE),
                             __getPointer(voltage),
                             __getPointer(bin_centers),
                             __c_real(charge),
                             __getLen(bin_centers),
                             __getLen(dt),
                             __c_real(acceleration_kick))
    
def linear_interp_kick_v3_cpp(dt, dE, voltage,
                       bin_centers, charge,
                       acceleration_kick):

    cpplib.linear_interp_kick_v3(__getPointer(dt),
                             __getPointer(dE),
                             __getPointer(voltage),
                             __getPointer(bin_centers),
                             __c_real(charge),
                             __getLen(bin_centers),
                             __getLen(dt),
                             __c_real(acceleration_kick))