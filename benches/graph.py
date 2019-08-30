import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def read_perf_file(perf_path, items):
    with open(perf_path) as perf_file:
        for line in perf_file:
            if line.startswith("Line tpr PME"):
                for i in range(2):
                    res = next(perf_file)
                    res = res.replace("-1( -1)", "-1").split()
                    rank = res[2]
                    gcycles = float(res[3])
                    stddev = float(res[4])
                    nsday = float(res[5])
                    items[rank] = {"gcycles": gcycles, "nsday": nsday, "stddev": stddev}

if __name__ == "__main__":
    for ext in ["sse2", "sse42", "avx", "avx2", "avx512-skylake","aarch64"]:
        ext_perf_path =  ext + "-perf.out"
        if not os.path.isfile(ext_perf_path):
            print("I don't have the performance file for " + ext, file=sys.stderr)
            continue

        nsimd_ext_perf_path = "nsimd-" + ext + "-perf.out"
        if not os.path.isfile(nsimd_ext_perf_path):
            print("I don't have the performance file for nsimd-"+ext, file=sys.stderr)
            continue

        items = dict()
        nsimd_items = dict()

        read_perf_file(ext_perf_path, items)
        read_perf_file(nsimd_ext_perf_path, nsimd_items)

        fig, axes = plt.subplots(1, 2, figsize=(10,5))

        plt.suptitle(ext.upper())

        width=.35
        ind = np.arange(2);

        def setup_subplot(ax, key):
            key_ext = [items["-1"][key], items["0"][key]]
            key_nsimd_ext = [nsimd_items["-1"][key], nsimd_items["0"][key]]
            p1 = ax.bar(ind, key_ext, width)
            p2 = ax.bar(ind+width, key_nsimd_ext, width)
            ax.set_xticks(ind + width/2)
            ax.set_xticklabels(("-1", "0"))
            ax.legend((p1[0], p2[0]), (ext, "nsimd-"+ext), loc='lower right')
            ax.set_xlabel("PMR")

        setup_subplot(axes[0], "nsday")
        axes[0].set_ylabel("ns/day")

        setup_subplot(axes[1], "gcycles")
        axes[1].set_ylabel("gcycles")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.savefig(ext + ".pdf")

        for exti, data in ((ext, items), ("nsimd-" + ext, nsimd_items)):
            with open(exti + "-data.csv", 'w') as data_file:
                data_file.write("PME Ranks,Gcycles Average,Standard Deviation, ns/day\n")
                for i in ["-1","0"]:
                    data_file.write("{},{},{},{}\n".format(i, data[i]["gcycles"], data[i]["stddev"], data[i]["nsday"]))


