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
    simd_exts_in_order = ["NONE", "SSE2", "NSIMD SSE2", "SSE42", "NSIMD SSE42",
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

def get_cpp_loc(output_filename, cpp_files):
    cmd = ' '.join(['sloccount'] + cpp_files) + ' >{}'.format(output_filename)
    if (os.system(cmd) != 0):
        debug('get_cpp_loc: error: {}'.format(cmd))
        return 0
    with open(output_filename, 'rb') as fin:
        for line in perf_file:
            if line.startswith('cpp:'):
                tab = line.split(' ')
                return int([i for i in tab[1:] if i != ''][0])
    return 0

def draw_graph_loc_simd(output_filename, simd_dir):
    native_simd_backends = [
        'impl_arm_neon_asimd',
        'impl_x86_avx_256',
        'impl_x86_avx_512_knl',
        'impl_x86_sse2',
        'scalar',
        'impl_arm_neon',
        'impl_x86_avx_128_fma',
        'impl_x86_avx2_256',
        'impl_x86_avx_512',
        'impl_x86_sse4_1'
    ]
    native_simd_loc = get_cpp_loc('native_simd_loc.txt',
                                  [os.path.join(simd_dir, i) \
                                   for i in native_simd_backends])
    nsimd_loc = get_cpp_loc('nsimd_loc.txt',
                            os.path.join(simd_dir, 'impl_nsimd'))

    plt.rcParams.update({'figure.autolayout': True})
    plt.style.use('ggplot')
    fig, ax = plt.subplots()

    labels = tuple(['Without NSIMD', 'With NSIMD'])
    values = tuple([native_simd_loc, nsimd_loc])
    n = len(labels)

    debug(title)
    debug('values = {}'.format(values))
    debug('labels = {}'.format(labels))
    debug('- - - - -')

    ind = np.arange(n)
    bars = ax.barh(ind, values)
    for i in range(n):
        if labels[i] = 'With NSIMD' == -1:
            bars[i].set_color('sandybrown')
        else:
            bars[i].set_color('cadetblue')
        ax.text(0, i, ' {:0.2f}'.format(values[i]), color='black', va='center')
    ax.set(title=title)
    plt.gca().invert_yaxis()
    plt.yticks(ind, labels)
    plt.savefig(output_filename)

# -----------------------------------------------------------------------------
# argv[1]  == GROMACS SIMD backends directory
# argv[2:] == List of log files

if __name__ == "__main__":
    res = dict()
    draw_graph_loc_simd('simd-loc-gromacs.pdf', sys.argv[1])
    for filename in sys.argv[2:]:
        debug('input = {}'.format(filename))
        if not os.path.isfile(filename):
            print('-- Warning: no performance file named ' + filename)
            continue
        ext = ' '.join((filename.split('/')[-1]).split('-')[0:-1]).upper()
        res[ext] = read_perf_file(filename)

    latex = ''
    res = transpose_dict_dict(res)
    for p in res.items():
        if len(p[1]) <= 1:
            continue
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
