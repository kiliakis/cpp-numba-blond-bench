# imports etc
from numba_src import pytrack
from cpp_src import cpptrack
import time
import os
import numpy as np
# from numba import njit, set_num_threads, threading_layer
import numba
import argparse
import prettytable
from prettytable import PrettyTable
import csv

'''
python driver.py -suite cpp -b histo \
-i 100 --nrf 1 -r 3 -output results-2023-09-13-cpp-3.csv \
-t 1 2 4 8 1 2 4 8 1 2 4 8 1 2 4 8 1 2 4 8 \
-p 1000000 1000000 1000000 1000000 2000000 2000000 2000000 2000000 4000000 4000000 4000000 4000000 8000000 8000000 8000000 8000000 16000000 16000000 16000000 16000000 \
-s 128 128 128 128 256 256 256 256 512 512 512 512 1024 1024 1024 1024 2048 2048 2048 2048

python driver.py -suite cpp -b histo histo_v2 histo_v3 histo_v4 histo_v5 \
-i 100 --nrf 1 -r 5 -output results-2023-09-14-cpp-3.csv \
-t 1 \
-p 1000000 8000000 \
-s 128 1024

python driver.py -suite cpp -b histo histo_v1 histo_v2 histo_v3 histo_v4 histo_v5 \
-i 100 --nrf 1 -r 5 -output results-2023-09-13-cpp.csv \
-t 1 2 4 8 1 2 4 8 1 2 4 8 1 2 4 8 1 2 4 8 \
-p 1000000 \
-s 128

python driver.py -suite numba_par cpp -b histo histo_v1 histo_v2 histo_v3 histo_v4 histo_v5 \
-i 100 --nrf 1 -r 5 -output results-2023-09-13-16Mparts.csv \
-t 1 2 4 8 \
-p 16000000 \
-s 2048

python driver.py -suite numpy -b histo histo_v1 histo_v2 histo_v3 histo_v4 histo_v5 \
-i 100 --nrf 1 -r 5 -output results-2023-09-13-numpy.csv \
-t 1 \
-p 1000000 2000000 4000000 8000000 \
-s 128 256 512 1024
'''
# 

parser = argparse.ArgumentParser(description='Benchmark numba and pure (ctypes) C++')

parser.add_argument('-r', '--repeats', type=int, default=1,
                    help='Repeat all experiments N times')

parser.add_argument('-i', '--iterations', type=int, default=100,
                    help='Number of Iterations to run')

parser.add_argument('-b', '--benchmarks', nargs='+', default=['kick', 'drift', 'histo', 'lik'],
                    help='Benchmarks to profile.')

parser.add_argument('-suite', '--suite', nargs='+', default=['numpy', 'cpp', 'numba', 'numba_fastm', 'numba_par'],
                    help='Suites to run.')

parser.add_argument('-p', '--particles', nargs='+', type=int, default=[int(1e6)],
                    help='Number of particles.')

parser.add_argument('-s', '--slices', nargs='+', type=int, default=[512],
                    help='Number of slices.')

parser.add_argument('-nrf', '--nrf', nargs='+', type=int, default=[1],
                    help='Number of rf stations.')

parser.add_argument('-t', '--threads', nargs='+', type=int, default=[1],
                    help='Number of threads to use.')

parser.add_argument('-output', '--output', type=str, default=None,
                    help='Output file name.')

cpp_funcs = {
    'kick': cpptrack.kick_cpp,
    'drift': cpptrack.drift_cpp,
    'histo': cpptrack.histogram_cpp,
    'histo_v2': cpptrack.histogram_v2_cpp,
    'histo_v3': cpptrack.histogram_v3_cpp,
    'histo_v4': cpptrack.histogram_v4_cpp,
    'histo_v5': cpptrack.histogram_v4_cpp,
    'lik': cpptrack.linear_interp_kick_cpp,
}

py_funcs = {
    'kick': pytrack.kick_py,
    'drift': pytrack.drift_py,
    'histo': pytrack.histogram_py,
    'histo_v1': pytrack.histogram_v1_py,
    'histo_v2': pytrack.histogram_v2_py,
    'histo_v3': pytrack.histogram_v3_py,
    'histo_v4': pytrack.histogram_v4_py,
    'histo_v5': pytrack.histogram_v5_py,
    'lik': pytrack.linear_interp_kick_py
}

numpy_funcs = {
    'kick': pytrack.kick_numpy,
    'drift': pytrack.drift_numpy,
    'histo': pytrack.histogram_numpy,
    'lik': pytrack.linear_interp_kick_numpy
}

# func_signature = {
#     'kick': 'void(float64, float64, float64, float64, float64, float64)',
#     'drift': 'void(float64, float64, float64, float64, float64, float64, float64, float64, float64)',
#     'histo': 'void(float64, float64, float64, float64)',
#     'lik': 'void(float64, float64, float64, float64, float64, float64)'
# }

if __name__ == "__main__":
    args = parser.parse_args()

    # input arguments
    print('Input arguments: ')
    for k, v in vars(args).items():
        print(k, v)

    n_particles_lst = args.particles
    n_slices_lst = args.slices
    n_rf_lst = args.nrf
    n_iter = args.iterations
    n_repeats = args.repeats
    benchmarks = args.benchmarks
    n_threads_lst = args.threads

    max_len = max(len(n_particles_lst), len(n_slices_lst), len(n_rf_lst), len(n_threads_lst))
    
    n_particles_lst = n_particles_lst * max_len if len(n_particles_lst) == 1 else n_particles_lst
    n_slices_lst = n_slices_lst * max_len if len(n_slices_lst) == 1 else n_slices_lst
    n_rf_lst = n_rf_lst * max_len if len(n_rf_lst) == 1 else n_rf_lst
    n_threads_lst = n_threads_lst * max_len if len(n_threads_lst) == 1 else n_threads_lst

    # print all lists
    # print('n_particles_lst: ', n_particles_lst, len(n_particles_lst))
    # print('n_slices_lst: ', n_slices_lst, len(n_slices_lst))
    # print('n_rf_lst: ', n_rf_lst, len(n_rf_lst))
    # print('n_threads_lst: ', n_threads_lst, len(n_threads_lst))

    assert len(n_particles_lst) == len(n_slices_lst) == len(n_rf_lst) == len(n_threads_lst) == max_len, \
        'Number of arguments must be 1 or equal to the maximum length of all arguments'


    # load functions
    numba_funcs = {}
    for k in benchmarks:
        if k in py_funcs:
            numba_funcs[k] = numba.njit()(py_funcs[k]) 

    numba_fastmath_funcs = {}
    for k in benchmarks:
        if k in py_funcs:
            numba_fastmath_funcs[k] = numba.njit(fastmath=True, )(py_funcs[k])

    numba_parallel_funcs = {}
    for k  in benchmarks:
        if k in py_funcs:
            numba_parallel_funcs[k] = numba.njit(parallel=True, fastmath=True, nogil=True, )(py_funcs[k])

    impl_dict = {
        'numpy': numpy_funcs,
        'cpp': cpp_funcs,
        'numba': numba_funcs,
        'numba_fastm': numba_fastmath_funcs,
        'numba_par': numba_parallel_funcs,
    }

    for rep in range(n_repeats):
        print(f'---- Run {rep+1}/{n_repeats} ----')
        for n_threads, n_rf, n_slices, n_particles in zip(n_threads_lst, n_rf_lst, n_slices_lst, n_particles_lst):
            # Set the max number of threads
            os.environ['OMP_NUM_THREADS'] = str(n_threads)
            numba.set_num_threads(n_threads)

            # Initialize arrays and other variables
            np.random.seed(0)
            dt = np.random.normal(loc=1e-9, scale=1e-10, size=n_particles)
            dE = np.random.normal(loc=0, scale=1e7, size=n_particles)
            profile = np.zeros(n_slices, dtype=float)
            dt_orig = np.copy(dt)
            dE_orig = np.copy(dE)

            cut_left = 0.9 * dt.min()
            cut_right = 1.1 * dt.max()
            bin_edges = np.linspace(cut_left, cut_right, n_slices+1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
            induced_voltage = 6e6 * np.random.uniform(size=n_slices)

            voltage = 6e6 * np.random.uniform(size=n_rf)
            omega_rf = 2.5e9 * np.random.uniform(size=n_rf)
            phi_rf = np.random.uniform(size=n_rf)
            acc_kick = 1e4
            beta = 0.99
            charge = 1.0
            alpha_zero = 1e-2
            alpha_one = 1e-3
            alpha_two = 1e-4
            energy = 450e9
            T0 = 9e-5
            length_ratio = 1.0


            func_args = {
                'kick': (dt, dE, voltage, omega_rf, phi_rf, acc_kick),
                'drift': (dt, dE, T0, length_ratio, beta, energy, alpha_zero, alpha_one, alpha_two),
                'histo': (dt, profile, cut_left, cut_right),
                'histo_v1': (dt, profile, cut_left, cut_right),
                'histo_v2': (dt, profile, cut_left, cut_right),
                'histo_v3': (dt, profile, cut_left, cut_right),
                'histo_v4': (dt, profile, cut_left, cut_right),
                'histo_v5': (dt, profile, cut_left, cut_right),
                'lik': (dt, dE, induced_voltage, bin_centers, charge, acc_kick)
            }

            # call functions, in loops
            header = ['suite', 'benchmark', 'avg(ms)', 'std', 'min', 'max', 'particles', 'slices', 'n_rf', 'n_thr']
            table = PrettyTable(header)
            table.align = 'l'
            table.hrules = prettytable.NONE
            rows = []
            # for impl, d in impl_dict.items():
            for impl in args.suite:
                d = impl_dict[impl]
                for b in benchmarks:
                    if b not in d:
                        continue
                    # print(impl, b)
                    # warmup
                    d[b](*func_args[b])
                    # print(dE)
                    timings = np.zeros(n_iter, float)
                    # then start timing
                    for i in range(n_iter):
                        dt[:] = dt_orig[:]
                        dE[:] = dE_orig[:]
                        start_t = time.time()
                        d[b](*func_args[b])
                        total_t = time.time() - start_t
                        timings[i] = 1e3 * total_t
                    # print('Profile sum: ', np.sum(profile))
                    avg = np.round(np.mean(timings), 3)
                    std = np.round(np.std(timings), 3)
                    min_t = np.round(np.min(timings), 3)
                    max_t = np.round(np.max(timings), 3)
                    rows.append([impl, b, avg, std, min_t, max_t, n_particles, n_slices, n_rf, n_threads])

            rows = sorted(rows, key=lambda x: x[1])
            for r in rows:
                table.add_row(r)
            # report
            print(table)

            # if args.output is not None write to csv file
            if args.output is not None:
                # write rows to csv file
                with open(args.output, 'a') as f:
                    writer = csv.writer(f, delimiter='\t')
                    if rep == 0 and n_threads == n_threads_lst[0]:
                        writer.writerow(header)
                    writer.writerows(rows)
            
            del dE, dt, profile, dt_orig, dE_orig
            del bin_edges, bin_centers, induced_voltage, voltage, omega_rf, phi_rf
