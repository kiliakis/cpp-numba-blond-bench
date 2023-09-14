import subprocess
import os
import numpy as np
from datetime import datetime

now = datetime.now().strftime("%Y-%m-%d_%H_%M")
project_dir = '/afs/cern.ch/work/k/kiliakis/git/cpp-numba-blond-bench/'
exe = os.path.join(project_dir, 'driver.py')
outfiles = project_dir + f'results/{now}/'

benches = ['kick', 'drift', 'histo', 'lik']
# benches = ['histo']

suites = ['cpp', 'numba', 'numba_par']
n_iterations_list = ['20']
# n_points_list = np.array([1e6, 2e6, 4e6, 8e6], dtype=int)
# n_points_list = np.array([1e6, 8e6], dtype=int)
n_points_list = np.array([8e6], dtype=int)

# n_threads_list = ['1', '2', '4', '8']
# n_threads_list = ['1', '8']
n_threads_list = ['8']

repeats = 10
# os.chdir(exe_dir)
total_sims = len(n_iterations_list) * len(n_points_list) * \
    len(n_threads_list) * repeats

# First compile
subprocess.run(['python', 'compile.py', '-p'])

current_sim = 0
for n_iterations in n_iterations_list:
    for n_points in n_points_list:
        for n_threads in n_threads_list:
            os.environ['OMP_NUM_THREADS'] = str(n_threads)
            os.environ['NUMBA_NUM_THREADS'] = str(n_threads)
            name = f'_p{n_points}_iter{n_iterations}_thr{n_threads}'
            if not os.path.exists(outfiles):
                os.makedirs(outfiles)
            #res = open(outfiles + name+'.res', 'w')
            stdout = open(outfiles + name+'.stdout', 'w')
            stderr = open(outfiles + name+'.stderr', 'w')
            n_slices = int(n_points//1000)
            for i in range(0, repeats):
                exe_args = ['python', exe,
                            f'-p{n_points}',
                            f'-t{n_threads}',
                            f'-i{n_iterations}',
                            f'-s{n_slices}', 
                            f'--benchmarks {" ".join(benches)}',
                            f'--suite {" ".join(suites)}']
                exe_args = ' '.join(exe_args) 
                print(n_iterations, n_points, n_threads, i)
                #start = time.time()
                subprocess.run(exe_args,
                                stdout=stdout,
                                stderr=stderr,
                                env=os.environ.copy(),
                                shell=True
                                )
                #end = time.time()
                current_sim += 1
                # res.write(str(end-start)+'\n')
                print("%.2f %% is completed" %
                      (100.0 * current_sim / total_sims))
