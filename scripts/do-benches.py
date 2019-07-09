import sys 
import os
from optparse import OptionParser

nsimd_build_path = "/home/jpailleux/Bureau/nsdev/nsimd/build"


### Gneration du fichier latex ###
# Defintion du préambule
def preambule(*packages):
  p = ""
  for i in packages:
    p = p+"\\usepackage{"+i+"}\n"
  return p

def write_latex_file(simd, new_body):
    start = "\\documentclass[12pt,a4paper,french]{article}\n\\usepackage[utf8]{inputenc}\n"
    start = start+preambule('amsmath','lmodern','babel', 'color', 'listings', 'fullpage')

    start = start + "\\lstset{basicstyle=\\small\\sffamily}" + "\\title{\\textbf{\\huge Benchmarks Gromacs}}\n" + "\\begin{document}\n\\maketitle\n"
    end = "\\end{document}\n"

    body = "\section{" + simd +  "}\n"
    body = body + new_body + "\\\\\n"

    container = start + body + end
    file = "gromacs_benches.tex"
    if os.path.exists(file):
      os.remove(file)
    fwriter = open("gromacs_benches.tex","x") # "x" pour la création et l'écriture
    fwriter.write(container)
    fwriter.close()


# Compile gromacs
def compil_gromacs(simd):
  os.system("cd ../build/")

  if simd == "nsimd" :
    print("We do nothing for this SIMD instruction set")
    # os.system("cmake .. -DGMX_SIMD=NSIMD -DGMX_MPI=on -DCMAKE_PREFIX_PATH=" + nsimd_build_path)
  elif simd == "sse2" :
    os.system("cmake .. -DGMX_SIMD=SSE2 -DGMX_MPI=on")
  elif simd == "sse4.1" :
    os.system("cmake .. -DGMX_SIMD=SSE4.1 -DGMX_MPI=on")
  elif simd == "avx" :
    os.system("cmake .. -DGMX_SIMD=AVX_256 -DGMX_MPI=on")  
  elif simd == "avx2" :
    os.system("cmake .. -DGMX_SIMD=AVX2_256 -DGMX_MPI=on")
  elif simd == "arm_neon" :
    # os.system("cmake .. -DGMX_SIMD=ARM_NEON)
    print("We do nothing for this SIMD instruction set")
  else :
    os.system("cmake .. -DGMX_MPI=on")

  # os.system("make -j 20")
  os.system("cd ../scripts/")


# Run benchmarks and save the result into a latex file
def benchmark():
  os.system("cd ../build/bin/")
  # nb_process = [1, 2, 16, 32, 64]
  nb_process = [1,2]
  perf = ""
  for i in nb_process :
    perf = perf + "\n\n\subsection{Number of MPI rank : " + str(i) +  "}\n"
    cmd_bench = "gmx_mpi tune_pme -np " + str(i) + " -s " + "../topol.tpr" + " -mdrun 'gmx_mpi mdrun'"
    os.system(cmd_bench)

    # The result of this command is in 
    with open('perf.out', 'r') as perf_file:
      perf = perf + "\\begin{lstlisting}\n" + perf_file.read() + "\n\\end{lstlisting}\n"

  os.system("cd ../scripts/")
  os.system("rm *bench.log* *perf.out*")
  return perf


#### MAIN ####
parser = OptionParser()
parser.add_option("-s", "--simd", dest="simd",
                  help="SIMD instruction set supported by gromacs")
parser.add_option("-c", "--clean", dest="clean",
                  help="Clean the content of gromacs_benches.tex", default=False)

(options, args) = parser.parse_args()

# We remove clean the report
if options.clean :
  os.remove("gromacs_benches.tex")
  f = open("gromacs_benches.tex","w+") 
  f.close()

# Compiling gromacs based on simd value
compil_gromacs(options.simd)

# Run benchmarks
benches = benchmark()
write_latex_file(options.simd, benches)

