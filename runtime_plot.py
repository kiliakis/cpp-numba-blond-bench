import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from plot.plotting_utilities import *
import argparse

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'

parser = argparse.ArgumentParser(description='Generate the figure with of the strong scaling experiment.',
                                 usage='python {} -i results/'.format(this_filename))

parser.add_argument('-i', '--infile', type=str,
                    help='The input file with the results.')

parser.add_argument('-o', '--outfile', type=str,
                    help='The plot file.')


parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plots.')


args = parser.parse_args()

gconfig = {
    'hatches': {'thr1': '', 'thr8': 'xx'},
    'colors': {'cpp': 'xkcd:light blue', 'numba': 'xkcd:light orange', 'numba_fastm': 'xkcd:light green', 'numba_par':'xkcd:light red', 'numpy':'xkcd:light red'},
    'annotate': {
        'fontsize': 9,
        'textcoords': 'data',
        'va': 'bottom',
        'ha': 'center'
    },
    'legend': {
        'loc': 'upper left', 'ncol': 2, 'handlelength': 1.5, 'fancybox': True,
        'framealpha': .5, 'fontsize': 11, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0.1, 'columnspacing': 0.8,
        'bbox_to_anchor': (.46, 0.99)
    },
    'subplots_adjust': {
        'wspace': 0.05, 'hspace': 0.1, 'top': 0.93
    },
    'ylim': [0, 15],
    'fontname': 'DejaVu Sans Mono',
    'outfiles': ['{}/{}-{}.png', '{}/{}-{}.pdf'],
    'lines': {
        # 'bench': ['kick', 'drift', 'histo'],
        # 'suite': ['cpp', 'numba', 'numba_fastm', 'numba_par'],
        'threads': ['1', '8'],
        'parts': ['1000000', '8000000']
    },
}
# plt.rcParams['ps.useafm'] = True
# plt.rcParams['pdf.use14corefonts'] = True
# plt.rcParams['text.usetex'] = True #Let TeX do the typsetting
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath'] #Force sans-serif math mode (for axes labels)
# plt.rcParams['font.family'] = 'sans-serif' # ... for regular text
# plt.rcParams['font.sans-serif'] = 'Helvetica'


# plt.rcParams['font.family'] = gconfig['fontname']
# plt.rcParams['text.usetex'] = True

if __name__ == '__main__':
    if not args.outfile:
        args.outfile = args.infile.split('.csv')[0]
    # We want different plots
    # For 1 and 8 threads
    # One bar per bench-suite, one subplot per parts (1M, 8M)
    data = np.genfromtxt(args.infile, delimiter='\t', dtype=str)
    header, data = list(data[0]), data[1:]
    plot_dir = get_plots(header, data, gconfig['lines'],
                     exclude=gconfig.get('exclude', []),
                     prefix=True)
    # print(plot_dir)
    # exit()
    # particle_list = ['1000000', '8000000']
    particle_list = ['1000000']

    fig, ax_arr = plt.subplots(ncols=1, nrows=len(particle_list),
                               sharex=False, sharey='row', figsize=(8,4))
    ax_arr = np.ravel(ax_arr)
    labels = set()
    i = 0
    for parts in particle_list:
        ax = ax_arr[i]
        plt.sca(ax)
        plt.title(f'Particles: {parts}')
        yvals = {}
        xvals = {}
        evals = {}
        # print(plot_dir.keys())
        # print(sorted(list(plot_dir.keys())))
        for key, vals in plot_dir.items():
            if parts not in key:
                continue
            threads = key.split('threads')[1].split('_')[0]
            for bench in ['kick', 'drift', 'histo', 'lik']:
                if bench not in yvals:
                    yvals[bench] = []
                    xvals[bench] = []
                    evals[bench] = []
                print(vals)
                for row in vals:
                    if bench not in row:
                        continue
                    suite = row[header.index('suite')]
                    if (threads == '8') and (suite not in ['cpp', 'numba_par']):
                        continue
                    if (threads == '1') and (suite in ['numba_par']):
                        continue
                    yvals[bench].append(float(row[header.index('avg(ms)')]))
                    xval = row[header.index('suite')]
                    if int(threads) > 1:
                        xval += f'-thr{threads}'
                        xval = xval.replace('_par', '')
                    xvals[bench].append(xval)
                    evals[bench].append(float(row[header.index('std')]))

        # print(yvals, xvals)
        # exit()
        pos = 0
        labels = set()
        for bench in yvals.keys():
            for xi, yi, ei in zip(xvals[bench], yvals[bench], evals[bench]):
                if xi in labels:
                    label = None
                else:
                    label = xi
                    labels.add(label)
                plt.bar(pos, yi, width=0.8, color=gconfig['colors'][xi.split('-')[0]], label=label,
                    edgecolor='black', hatch=gconfig['hatches'].get(xi.split('-')[-1], ''))
                    # , yerr=ei, capsize=2)
                if yi > gconfig['ylim'][1]:
                    plt.gca().annotate(f'{yi:.1f}', xy=(pos+0.5, gconfig['ylim'][1]*.95), ha='left', va='bottom',
                        rotation=0,
                        textcoords='data')

                else:
                    plt.gca().annotate(f'{yi:.1f}', xy=(pos, yi), ha='center', va='bottom',
                        textcoords='data')
                pos+= 1
            pos+=1
        i+=1
        plt.grid(True, which='both', axis='y', alpha=0.5)
        plt.gca().set_facecolor('0.85')
        plt.xticks([2, 8, 14, 20], ['kick', 'drift', 'histo', 'interp_kick'])
        plt.ylabel('Runtime (ms)')
        plt.legend(**gconfig['legend'])
        plt.ylim(gconfig['ylim'])
        plt.tight_layout()
    plt.savefig(args.outfile, dpi=400, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()
