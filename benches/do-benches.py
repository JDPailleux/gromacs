import sys 
import os
from optparse import OptionParser


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

    start = start + "\\lstset{basicstyle=\\small\\sffamily}" + "\\title{\\textbf{\\huge Benchmarks results of Gromacs}}\n" + "\\begin{document}\n\\maketitle\n"
    end = "\\end{document}\n"

    if simd != "nsimd":
      body = "\section{Performance tests beetween " + simd + " and NSIMD for " + simd + "}\n"
    else :
      body = "\section{Performance tests beetween of NSIMD}\n"
    body = body + new_body + "\n"

    container = start + body + end
    file = "gromacs_benches_"+ simd+".tex"
    if os.path.exists(file):
      os.remove(file)
    fwriter = open("gromacs_benches_"+ simd+".tex","x") # "x" pour la création et l'écriture
    fwriter.write(container)
    fwriter.close()


# Compile gromacs with MPI and without MPI
def compil_gromacs(simd, nsimd_build_path):

  if simd == "nsimd" or simd == "NSIMD":
    print("We do nothing for this SIMD instruction set")
    os.system("cd ../build/; cmake .. -DGMX_SIMD=NSIMD -DGMX_MPI=on -DGMX_NSIMD_BUILD_PATH=" + nsimd_build_path +" -DCMAKE_PREFIX_PATH=" + nsimd_build_path +"/build")
    os.system("cd ../build/; make -j30")
    os.system("cd ../build/; cmake .. -DGMX_SIMD=NSIMD -DGMX_MPI=off -DGMX_NSIMD_BUILD_PATH=" + nsimd_build_path +" -DCMAKE_PREFIX_PATH=" + nsimd_build_path +"/build")
  elif simd == "sse2" or simd == "SSE2" :
    os.system("cd ../build/; cmake .. -DGMX_SIMD=SSE2 -DGMX_MPI=on")
    os.system("cd ../build/; make -j30")
    os.system("cd ../build/; cmake .. -DGMX_SIMD=SSE2 -DGMX_MPI=off")
  elif simd == "sse4.1" or simd == "SSE4.1" :
    os.system("cd ../build/; cmake .. -DGMX_SIMD=SSE4.1 -DGMX_MPI=on")
    os.system("cd ../build/; make -j30")
    os.system("cd ../build/; cmake .. -DGMX_SIMD=SSE4.1 -DGMX_MPI=off")
  elif simd == "avx" or simd == "AVX_256" :
    os.system("cd ../build/; cmake .. -DGMX_SIMD=AVX_256 -DGMX_MPI=on") 
    os.system("cd ../build/; make -j30") 
    os.system("cd ../build/; cmake .. -DGMX_SIMD=AVX_256 -DGMX_MPI=off")  
  elif simd == "avx2" or simd == "AVX2_256" :
    os.system("cd ../build/; cmake .. -DGMX_SIMD=AVX2_256 -DGMX_MPI=on")
    os.system("cd ../build/; make -j30")
    os.system("cd ../build/; cmake .. -DGMX_SIMD=AVX2_256 -DGMX_MPI=off")
  elif simd == "avx512" or simd == "AVX_512" :
    os.system("cd ../build/; cmake .. -DGMX_SIMD=AVX_512 -DGMX_MPI=on")
    os.system("cd ../build/; make -j30")
    os.system("cd ../build/; cmake .. -DGMX_SIMD=AVX_512 -DGMX_MPI=off")
  elif simd == "avx512_knl" or simd == "AVX_512_KNL" :
    os.system("cd ../build/; cmake .. -DGMX_SIMD=AVX_512_KNL -DGMX_MPI=on")
    os.system("cd ../build/; make -j30")
    os.system("cd ../build/; cmake .. -DGMX_SIMD=AVX_512_KNL -DGMX_MPI=off")
  elif simd == "arm_neon" or simd == "ARM_NEON" :
    os.system("cd ../build/; cmake .. -DGMX_SIMD=ARM_NEON -DGMX_MPI=on")
    os.system("cd ../build/; make -j30")
    os.system("cd ../build/; cmake .. -DGMX_SIMD=ARM_NEON -DGMX_MPI=off")
  else : # Auto
    os.system("cd ../build/; cmake .. -DGMX_MPI=on")
    os.system("cd ../build/; make -j30")
    os.system("cd ../build/; cmake .. -DGMX_MPI=off")

  os.system("cd ../build/; make -j30")

# Run benchmarks and save the result into a latex file
def benchmark(simd, nsimd_path):
  perf = ""
  os.system("../build/bin/gmx tune_pme -r 10 -s topol.tpr -mdrun 'gmx_mpi mdrun'")

  if simd != "nsimd":
    # The result of this command is in 
    with open('perf.out', 'r') as perf_file:
      perf = perf +"\\subsection{"+ simd +"}\n" + "\\begin{lstlisting}[frame=single]\n" + perf_file.read() + "\n\\end{lstlisting}\n"
    
    # Compilation of NSIMD
    os.system("cd ../build/; cmake .. -DGMX_SIMD=NSIMD -DGMX_MPI=on -DGMX_NSIMD_BUILD_PATH=" + nsimd_path +" -DNSIMD_FOR="+ simd +" -DCMAKE_PREFIX_PATH=" + nsimd_path +"/build")
    os.system("cd ../build/; make -j30")
    os.system("cd ../build/; cmake .. -DGMX_SIMD=NSIMD -DGMX_MPI=off -DGMX_NSIMD_BUILD_PATH=" + nsimd_path +" -DNSIMD_FOR="+ simd +" -DCMAKE_PREFIX_PATH=" + nsimd_path +"/build")
    os.system("cd ../build/; make -j30")
    
    compil_gromacs("nsimd", nsimd_path)
    with open('perf.out', 'r') as perf_file:
      perf = perf +"\\subsection{NSIMD for " + simd + "}\n" +"\\begin{lstlisting}[frame=single]\n" + perf_file.read() + "\n\\end{lstlisting}\n"
  else :
    with open('perf.out', 'r') as perf_file:
      perf = perf +"\\subsection{NSIMD}\n" + "\\begin{lstlisting}[frame=single]\n" + perf_file.read() + "\n\\end{lstlisting}\n"

  os.system("rm *bench.log* *perf.out*")
  return perf


#### MAIN ####
parser = OptionParser()
parser.add_option("-s", "--simd", dest="simd",
                  help="SIMD instruction set supported by gromacs")
parser.add_option("-p", "--nsimd_path", dest="nsimd_path",
                  help="NSIMD build path")
parser.add_option("-c", "--clean", dest="clean",
                  help="Clean the content of gromacs_benches.tex", default=False)

(options, args) = parser.parse_args()

# We remove clean the report
if options.clean :
  os.remove("gromacs_benches_"+ options.simd+".tex")
  f = open("gromacs_benches_"+ options.simd+".tex","w+") 
  f.close()

# Compiling gromacs based on simd value
compil_gromacs(options.simd, options.nsimd_path)

# Run benchmarks
benches = benchmark(options.simd, options.nsimd_path)
write_latex_file(options.simd, benches)


