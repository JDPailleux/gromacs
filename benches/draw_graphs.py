import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# -----------------------------------------------------------------------------

list_of_measures = [
  "Neighbor search",
  "Force",
  "NB X/F buffer ops.",
  "Write traj.",
  "Update",
  "Constraints",
  "Rest",
  "PME spread",
  "PME gather",
  "PME 3D-FFT",
  "PME solve Elec"
]

def measure2filename(m):
    return m.replace(' ', '-').replace('/', '-').replace('.', '-') \
           + '-gromacs.svg'

# -----------------------------------------------------------------------------

DEBUG = False

def debug(msg):
    if DEBUG:
        print('DEBUG: {}'.format(msg))

# -----------------------------------------------------------------------------

def read_perf_file(filename):
    ret = dict()
    try_and_parse = False
    with open(filename) as perf_file:
        for line in perf_file:
            if line.find('T I M E   A C C O U N T I N G') != -1:
                try_and_parse = True
            if not try_and_parse:
                continue
            for m in list_of_measures:
                if line.find(' ' + m + ' ') != -1:
                    tmp = line.split(' ')
                    j = 0
                    for i in range(len(tmp) - 1, 0, -1):
                        if tmp[i] != '':
                            j = j + 1
                        if j == 3:
                            ret[m] = float(tmp[i])
                            debug('{} = {}'.format(m, tmp[i]))
                            break
            if line.find('Performance:') != -1:
                tmp = line.split(' ')
                j = 0
                for i in range(len(tmp)):
                    if tmp[i] != '':
                        j = j + 1
                    if j == 2:
                        ret['ns/day'] = float(tmp[i])
                        debug('ns/day = {}'.format(m, tmp[i]))
                        break
    return ret

# -----------------------------------------------------------------------------

def transpose_dict_dict(d):
    ret = dict()
    secondary_keys = list(d.keys())
    primary_keys = set()
    for p in d.items():
        primary_keys = primary_keys | set(p[1])
    for p in primary_keys:
        ret[p] = dict()
    for p in primary_keys:
        for s in secondary_keys:
            try:
                ret[p][s] = d[s][p]
            except:
                print('-- Warning: no key {}/{}'.format(s, p))
    return ret

# -----------------------------------------------------------------------------

def draw_graph(output_filename, title, data):
    simd_exts_in_order = ["CPU", "SSE2", "NSIMD SSE2", "SSE42", "NSIMD SSE42",
                          "AVX", "NSIMD AVX", "AVX2", "NSIMD AVX2",
                          "AVX512_KNL", "NSIMD AVX512_KNL", "AVX512_SKYLAKE",
                          "NSIMD AVX512_SKYLAKE", "NEON128", "NSIMD NEON128",
                          "AARCH64", "NSIMD AARCH64"]

    #def gen_svg(title, xlabel, mapping, svg):
    plt.rcParams.update({'figure.autolayout': True})
    plt.style.use('ggplot')
    fig, ax = plt.subplots()

    labels = []
    values = []
    for simd_ext in simd_exts_in_order:
        if not simd_ext in data:
            continue
        labels.append(simd_ext)
        values.append(data[simd_ext])
    values = tuple(values)
    labels = tuple(labels)
    n = len(labels)

    debug(title)
    debug('data   = {}'.format(data))
    debug('values = {}'.format(values))
    debug('labels = {}'.format(labels))
    debug('- - - - -')

    ind = np.arange(n)
    bars = ax.barh(ind, values)
    for i in range(n):
        if labels[i].find('NSIMD') == -1:
            bars[i].set_color('sandybrown')
        else:
            bars[i].set_color('cadetblue')
        ax.text(0, i, ' {:0.2f}'.format(values[i]), color='black', va='center')
    if title != 'ns/day':
        ax.set(title=title, xlabel='seconds')
    else:
        ax.set(title=title)
    plt.gca().invert_yaxis()
    plt.yticks(ind, labels)
    plt.savefig(output_filename)

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    res = dict()
    for filename in sys.argv[1:]:
        debug('input = {}'.format(filename))
        if not os.path.isfile(filename):
            print('-- Warning: no performance file named ' + filename)
            continue
        ext = ' '.join((filename.split('/')[-1]).split('-')[0:-1]).upper()
        res[ext] = read_perf_file(filename)

    latex = ''
    res = transpose_dict_dict(res)
    for p in res.items():
        svg = measure2filename(p[0])
        pdf = svg.replace('.svg', '.pdf')
        draw_graph(svg, p[0], p[1])
        os.system('rsvg-convert -f pdf -o {} {}'.format(pdf, svg))
        latex += \
          '\\subsection{{Comparison graph for performance counter: {}}}\n'. \
          format(p[0])
        latex += '\\begin{center}\n'
        latex += '  \includegraphics[width=0.75\\textwidth]{{{}}}'.format(pdf)
        latex += '\\end{center}\n'
    with open('comparison-graphs.tex', 'w') as fout:
        fout.write(latex)
