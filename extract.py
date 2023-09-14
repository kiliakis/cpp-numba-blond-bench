#!/usr/bin/python
import os
import csv
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Extract results')

parser.add_argument('-i', '--indir', type=str, help='Input directory')

parser.add_argument('-o', '--outfile', type=str, 
                    help='File to write the results.')

       
def string_between(string, before, after):
    temp = string
    if before:
        temp = temp.split(before)[1]
    if after:
        temp = temp.split(after)[0]
    return temp


def extract_results(input, out):
    header = ['suite', 'bench', 'threads', 'parts', 'slices', 'nrf', 'iter', 'avg(ms)',
              'std', 'min', 'max']
    records = []
    for dirs, subdirs, files in os.walk(input):
        for file in files:
            if ('.stdout' not in file):
                continue
            n_iter = string_between(file, '_iter', '_')
            n_parts = string_between(file, '_p', '_')
            threads = string_between(file, '_thr', '.stdout')
            d = {}
            for line in open(os.path.join(dirs, file), 'r'):
                line = line.replace(' ', '')
                suite, bench, time, std, min_t, max_t, n_parts, n_slices, nrf = line.split('|')[1:-1]
                try:
                    time = float(time)
                    std = float(std)
                    min_t = float(min_t)
                    max_t = float(max_t)
                except ValueError:
                    continue
                if suite not in d:
                    d[suite] = {}
                if bench not in d[suite]:
                    d[suite][bench] = {
                        'avg(ms)': [],
                        'std': [],
                        'min': [],
                        'max': [],
                    }
                d[suite][bench]['avg(ms)'].append(time)
                d[suite][bench]['std'].append(std)
                d[suite][bench]['min'].append(min_t)
                d[suite][bench]['max'].append(max_t)
            for suite, d_suite in d.items():
                for bench, data in d_suite.items():
                    if len(data['avg(ms)']) > 4:
                        keepidx = np.arange(2, len(data['avg(ms)'])-2)
                    elif len(data['avg(ms)']) > 2:
                        keepidx = np.arange(1, len(data['avg(ms)'])-1)
                    else:
                        keepidx = np.arange(0, len(data['avg(ms)']))
                    avg = f"{np.mean(np.array(data['avg(ms)'])[keepidx]):.3f}"
                    std = f"{np.mean(np.array(data['std'])[keepidx]):.3f}"
                    min_t = f"{np.mean(np.array(data['min'])[keepidx]):.3f}"
                    max_t = f"{np.mean(np.array(data['max'])[keepidx]):.3f}"
                    records.append([suite, bench, threads, n_parts, n_slices, nrf, n_iter, avg, std, min_t, max_t])
    records.sort(key=lambda a: (a[1], a[0], int(a[3]), int(a[4])))
    writer = csv.writer(open(out, 'w'), lineterminator='\n', delimiter='\t')
    writer.writerow(header)
    writer.writerows(records)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.outfile is None:
        args.outfile = 'results-' + os.path.basename(os.path.normpath(args.indir)) + '.csv'
    extract_results(args.indir, args.outfile)
    
    
