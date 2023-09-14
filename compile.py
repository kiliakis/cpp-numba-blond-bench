# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''

@author: Danilo Quartullo, Konstantinos Iliakis
'''

# MAKE SURE YOU HAVE GCC 4.8.1 OR LATER VERSIONS ON YOUR SYSTEM LINKED TO YOUR
# SYSTEM PATH.IF YOU ARE ON CERN-LXPLUS YOU CAN TYPE FROM CONSOLE
# source /afs/cern.ch/sw/lcg/contrib/gcc/4.8.1/x86_64-slc6/setup.sh
# TO GET GCC 4.8.1 64 BIT. IN GENERAL IT IS ADVISED TO USE PYTHON 64 BIT PLUS
# GCC 64 BIT.

from __future__ import print_function
import os
import sys
import subprocess
import ctypes
import argparse

path = os.path.realpath(__file__)
basepath = os.sep.join(path.split(os.sep)[:-1])

parser = argparse.ArgumentParser(description='Compile C++ library')

parser.add_argument('-p', '--parallel',
                    default=False, action='store_true',
                    help='Produce Multi-threaded code. Use the environment'
                         ' variable OMP_NUM_THREADS=xx to control the number of'
                         ' threads that will be used.'
                         ' Default: Serial code')

parser.add_argument('-c', '--compiler', type=str, default='g++',
                    help='C++ compiler that will be used to compile the'
                         ' source files. Default: g++')

parser.add_argument('--flags', type=str, default='',
                    help='Additional compile flags.')

parser.add_argument('--libs', type=str, default='',
                    help='Any extra libraries needed to compile')

parser.add_argument('-libname', '--libname', type=str, default=os.path.join(basepath, 'cpp_src/libtrack.so'),
                    help='The library name, without the file extension.')

parser.add_argument('-optimize', '--optimize', action='store_true',
                    help='Auto optimize the compiled library.')

# Additional libs needed to compile the blond library
libs = []

# EXAMPLE FLAGS: -Ofast -std=c++11 -fopt-info-vec -march=native
#                -mfma4 -fopenmp -ftree-vectorizer-verbose=1
cflags = ['-O3', '-std=c++11', '-shared']

cpp_files = [
    os.path.join(basepath, 'cpp_src/tracking.cpp'),
    os.path.join(basepath, 'cpp_src/openmp.cpp'),
]

if __name__ == "__main__":
    args = parser.parse_args()
    compiler = args.compiler

    if args.libs:
        libs = args.libs.split()

    if args.parallel:
        cflags += ['-fopenmp', '-DPARALLEL', '-D_GLIBCXX_PARALLEL']

    if args.flags:
        cflags += args.flags.split()

    if 'posix' in os.name:
        cflags += ['-fPIC']
        if args.optimize:
            # Check compiler defined directives
            # This is compatible with python3.6 - python 3.9
            # The universal_newlines argument transforms output to text (from binary)
            ret = subprocess.run([compiler + ' -march=native -dM -E - < /dev/null | egrep "SSE|AVX|FMA"'],
                                 shell=True, stdout=subprocess.PIPE, universal_newlines=True)

            # If we have an error
            if ret.returncode != 0:
                print('Compiler auto-optimization did not work. Error: ', ret.stdout)
            else:
                # Format the output list
                stdout = ret.stdout.replace('#define ', '').replace(
                    '__ 1', '').replace('__', '').split('\n')
                # Add the appropriate vectorization flag (not use avx512)
                if 'AVX2' in stdout:
                    cflags += ['-mavx2']
                elif 'AVX' in stdout:
                    cflags += ['-mavx']
                elif 'SSE4_2' in stdout or 'SSE4_1' in stdout:
                    cflags += ['-msse4']
                elif 'SSE3' in stdout:
                    cflags += ['-msse3']
                else:
                    cflags += ['-msse']

                # Add FMA if supported
                if 'FMA' in stdout:
                    cflags += ['-mfma']

        root, ext = os.path.splitext(args.libname)
        if not ext:
            ext = '.so'
        libname = os.path.abspath(root + ext)

    elif 'win' in sys.platform:
        root, ext = os.path.splitext(args.libname)
        if not ext:
            ext = '.dll'
        libname = os.path.abspath(root + ext)

        if hasattr(os, 'add_dll_directory'):
            directory, filename = os.path.split(libname)
            os.add_dll_directory(directory)

    else:
        print(
            'YOU ARE NOT USING A WINDOWS OR LINUX OPERATING SYSTEM. ABORTING...')
        sys.exit(-1)
    command = [compiler] + cflags + ['-o', libname] + cpp_files + libs

    print('Enable Multi-threaded code: ', args.parallel)
    print('C++ Compiler: ', compiler)
    print('Compiler version: ')
    subprocess.run([compiler, '--version'])
    print('Compiler flags: ', ' '.join(cflags))
    print('Extra libraries: ', ' '.join(libs))
    print('Library name: ', libname)

    # If it exists already, remove the library before re-compiling
    if os.path.isfile(libname):
        try:
            os.remove(libname)
        except OSError:
            pass

    print('Compiling:\n', ' '.join(command))
    ret = subprocess.run(command)
    if ret.returncode != 0:
        print('\nThere was a compilation error.')
    else:
        try:
            if ('win' in sys.platform) and hasattr(os, 'add_dll_directory'):
                lib = ctypes.CDLL(libname, winmode=0)
            else:
                lib = ctypes.CDLL(libname)
            print('\nThe library has been successfully compiled.')
        except Exception as e:
            print('\nCompilation failed.')
            print(e)

