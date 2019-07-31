import sys 
import os
from optparse import OptionParser

def packages_and_title():
  return r"""
\\input{tex/preambule}
\\usepackage{listings}
\\usepackage{algorithm}
\\usepackage{amsmath} 
\\usepackage{algorithmic}
\\usepackage{array,multirow,makecell}
\\usepackage[table]{xcolor}

\\usepackage[T1]{fontenc}
\\usepackage[utf8]{inputenc}
\\usepackage{multicol}


\\usepackage{fancyhdr}
\\pagestyle{fancy}
\\renewcommand{\headrulewidth}{0pt}
\\renewcommand{\footrulewidth}{0pt}
\\rhead{
\\includegraphics[width=4.5cm]{tex/scale.png}
}

\\title{\\includegraphics[width=27em]{tex/scale.png}\\\\ \\vspace{9em}{\\fontsize{30}{40} \\textbf{Benchmarks of NSIMD on Intel Skylake/AVX-512 capable chip}}}
\\begin{document}
\\maketitle
\\vspace{10em}
\\textbf{AGENIUM SCALE}\\\\
Rue Noetzlin\\\\
Batiment 660, Digiteo Labs\\\\
91 190 Gif-sur-Yvette\\\\
01 69 15 32 32 \\\\
contact@numsclae.com\\\\
https://www.numscale.com\\\\

\\clearpage
\\pagenumbering{gobble}

\\begin{flushright}\\textit{}\\end{flushright}


\\tableofcontents
\\newpage
"""

def introduction():
  res = os.popen("gcc --version").readlines()
  compiler_version = ''.join(res)
  res = os.popen("uname -a").readlines()
  info_sys = ''.join(res)
  res = os.popen("lscpu").readlines()
  info_cpu = ''.join(res)
  res = os.popen("cat /proc/meminfo").readlines()
  info_ram = ''.join(res)
  res = os.popen("ldd --version").readlines()
  ldd = ''.join(res)

  intro = r"""
  \\section{Introduction}
	\\subsection{About this document}
 This document presents the results of benchmarks performed by AGNENIUM SCALE on Intel Skylake/AVX-512 capable chip.\\\\
 
	This document is meant to be read by software developers.  The explanations provided in thepresent  sections  are  not  intended  to  be  detailed.   We  assume  that  the  reader  has  sufficientknowledge to understand the present document.  If you have any relevant question feel free to contact us at \\href{contact@numscale.com}{contact@numscale.com}. \\\\
 
 This first part of this document is the introduction which provides detailed information about the benchmarking setup, such as hardware, software, and metrics used.  The second part gives benchmark results that allow you to have a quick idea of how NSIMD performs against other GROMACS versions. The third part provides the comparaison of the performances between NSIMD and the others SIMD extension.\\\\
 
 For the benchmarks performed, the version of gromacs is 2019.3 (5.0) published on june 2019 from commit \href{https://github.com/gromacs/gromacs/commit/e5a8e1537f19b8c2a3781ceb38b926dac93e5c26}{https://github.com/gromacs/gromacs/commit/e5a8e1537f19b8c2a3781ceb38b926dac93e5c26}. However, the version used has several modifications to integrate NSIMD into the source code. You will find it at  \\href{https://github.com/agenium-scale/gromacs}{https://github.com/agenium-scale/gromacs}. You can see the corrections and the differents additions to the NSIMD version on the nsimd-translate branch in this fork of the GROMACS's repository.
 
	\\subsection{Benchmark Setup}
	
	All benchmarks are performed using tools given by GROMACS for testings the performace. For each benchmarks we have use \\textit{gmx tune\\_pme} with only one MPI rank. For more information on \textit{gmx tune\\_pme} please see: \\\href{http://manual.gromacs.org/documentation/2018/onlinehelp/gmx-tune\\_pme.html}{http://manual.gromacs.org/documentation/2018/onlinehelp/gmx-tune\\_pme.html}.\\
	
	For the benchmark runs, the default of 1000 time steps should suffice for most systems. The dynamic load balancing needs about 100 time steps to adapt to local load imbalances, therefore the time step counters are by default reset after 100 steps. \\\\
	
	After calling \\textit{gmx\\_mpi mdrun} several times (option \\textit{-r} repeat each test \\textit{r} times), detailed performance information is available in the output file \textit{perf.out}. Note that during the benchmarks, a couple of temporary files are created and are deleted after each test. 
	
	\\subsection{Contents of this document}
	
	In the following section we dump the outputs of some well-known shell commands that pro-vide useful information on the machine, operating system, compiler version, that were used forbenchmarks. Dumps of shell commands have two main advantages:
	
	\\begin{itemize}
	\\item Very accurate information about the benchmarking environment is provided in raw formideal for our intended audience.
	\\item We do not have time to write some cumbersome code to generate beautiful English sentencesto describe what is best described by shell commands whose output format is well knownby our intended audience.
	\\end{itemize}
	
		
	\\subsection{Organization of benchmarks} 
	
For each SIMD extensions used, the following benchmarks are performed:
	
	\\begin{itemize}
	\\item The report made by GROMACS are given below for each SIMD extension tested (SSE2, SSE4.1, AVX, AVX2 and AVX 512 Skylake) and NSIMD. 
	
	\\item This report provides information on the number of MPI rank used, on the input file and the command used to launch the benchmark. There is also information on the simulation throughput in nanosecond per day (higher is better) as well as the number of cycles for each PME rank. In addition, it is indicated which PME rank is the most recommended for better performance during the simulation.

	\\item Essentials results information are extracted from this previous report (Average of simulation throughput and number of cycles). 
	
	\\item A performance comparison between NSIMD and every SIMD extensions used during the benchmarks.
	\\end{itemize}
	
	\\subsection{Compiler version}
	\\begin{lstlisting}[frame=single]
""" + compiler_version + r""" 
\\end{lstlisting}
	
	\\subsection{Operating system description}
\\begin{lstlisting}[frame=single]
""" + info_sys + r"""
\\end{lstlisting}

\subsection{Information about the CPU architecture}
\begin{lstlisting}[frame=single]
""" + info_cpu + r"""
\\end{lstlisting}
	
\\subsection{Information about the RAM}
\\begin{lstlisting}[frame=single]
 """ + info_ram + r"""
\\end{lstlisting}

\\subsection{Information about the standard library}
\\begin{lstlisting}[frame=single] 
""" + ldd + r"""
\\end{lstlisting}\n"""
  return intro



### Gneration du fichier latex ###
# Defintion du préambule
def preambule(*packages):
  p = ""
  for i in packages:
    p = p+"\\usepackage{"+i+"}\n"
  return p

def write_latex_file(simd, new_body):
    start = packages_and_title()
    start = start + "\\documentclass[12pt,a4paper,french]{article}\n\\usepackage[utf8]{inputenc}\n"
    start = start + preambule('amsmath','lmodern','babel', 'color', 'listings', 'fullpage')
    start = start + introduction()

    # start = start + "\\lstset{basicstyle=\\small\\sffamily}" + "\\title{\\textbf{\\huge Benchmarks results of Gromacs}}\n" + "\\begin{document}\n\\maketitle\n"
    end = "\\end{document}\n"

    if simd != "nsimd":
      body = "\section{Performance tests beetween " + str(simd) + " and NSIMD for " + str(simd) + "}\n"
    else :
      body = "\section{Performance tests beetween of NSIMD}\n"
    body = body + new_body + "\n"

    container = start + body + end
    file = "gromacs_benches_" + simd + ".tex"
    if os.path.exists(file):
      os.remove(file)
    fwriter = open("gromacs_benches_" + simd + ".tex","x") # "x" pour la création et l'écriture
    fwriter.write(container)
    fwriter.close()


# Compile gromacs with MPI and without MPI
def compil_gromacs(simd, nsimd_path):
  cmd_make = "cd ../build/; make -j20; export LD_LIBRARY_PATH=" + str(nsimd_path) + "/build"

  if simd == "nsimd" or simd == "NSIMD":
    print("We do nothing for this SIMD instruction set")
    os.system("cd ../build/; cmake .. -DGMX_SIMD=NSIMD -DGMX_MPI=on -DGMX_NSIMD_PATH=" + str(nsimd_path) +" -DCMAKE_PREFIX_PATH=" + str(nsimd_path) +"/build")
    os.system(cmd_make)
    os.system("cd ../build/; cmake .. -DGMX_SIMD=NSIMD -DGMX_MPI=off -DGMX_NSIMD_PATH=" + str(nsimd_path) +" -DCMAKE_PREFIX_PATH=" + str(nsimd_path) +"/build")
  elif simd == "sse2" or simd == "SSE2" :
    os.system("cd ../build/; cmake .. -DGMX_SIMD=SSE2 -DGMX_MPI=on")
    os.system(cmd_make)
    os.system("cd ../build/; cmake .. -DGMX_SIMD=SSE2 -DGMX_MPI=off")
  elif simd == "sse4.1" or simd == "SSE4.1" :
    os.system("cd ../build/; cmake .. -DGMX_SIMD=SSE4.1 -DGMX_MPI=on")
    os.system(cmd_make)
    os.system("cd ../build/; cmake .. -DGMX_SIMD=SSE4.1 -DGMX_MPI=off")
  elif simd == "avx" or simd == "AVX_256" or simd == "AVX":
    os.system("cd ../build/; cmake .. -DGMX_SIMD=AVX_256 -DGMX_MPI=on") 
    os.system(cmd_make)
    os.system("cd ../build/; cmake .. -DGMX_SIMD=AVX_256 -DGMX_MPI=off")  
  elif simd == "avx2" or simd == "AVX2_256" or simd == "AVX2" :
    os.system("cd ../build/; cmake .. -DGMX_SIMD=AVX2_256 -DGMX_MPI=on")
    os.system(cmd_make)
    os.system("cd ../build/; cmake .. -DGMX_SIMD=AVX2_256 -DGMX_MPI=off")
  elif simd == "avx512" or simd == "AVX_512" :
    os.system("cd ../build/; cmake .. -DGMX_SIMD=AVX_512 -DGMX_MPI=on")
    os.system(cmd_make)
    os.system("cd ../build/; cmake .. -DGMX_SIMD=AVX_512 -DGMX_MPI=off")
  elif simd == "avx512_knl" or simd == "AVX_512_KNL" :
    os.system("cd ../build/; cmake .. -DGMX_SIMD=AVX_512_KNL -DGMX_MPI=on")
    os.system(cmd_make)
    os.system("cd ../build/; cmake .. -DGMX_SIMD=AVX_512_KNL -DGMX_MPI=off")
  elif simd == "arm_neon" or simd == "ARM_NEON" :
    os.system("cd ../build/; cmake .. -DGMX_SIMD=ARM_NEON -DGMX_MPI=on")
    os.system(cmd_make)
    os.system("cd ../build/; cmake .. -DGMX_SIMD=ARM_NEON -DGMX_MPI=off")
  else : # Auto
    os.system("cd ../build/; cmake .. -DGMX_MPI=on")
    os.system(cmd_make)
    os.system("cd ../build/; cmake .. -DGMX_MPI=off")

  os.system(cmd_make)

# Run benchmarks and save the result into a latex file
def benchmark(SIMD, nsimd_path):
  perf = ""
  cmd_bench = "../build/bin/gmx tune_pme -r 10 -s topol.tpr -mdrun '../build/bin/gmx_mpi mdrun'"
  if SIMD == "sse2" or SIMD == "SSE2":
    list_simd = ["SSE2"]
  elif SIMD == "sse4.1" or SIMD == "SSE4.1" :
    list_simd = ["SSE2", "SSE4.1"]
  elif SIMD == "avx" or SIMD == "AVX" :
    list_simd = ["SSE2", "SSE4.1", "AVX"]
  elif SIMD == "avx2" or SIMD == "AVX2" :
    list_simd = ["SSE2", "SSE4.1", "AVX", "AVX2"]
  elif SIMD == "avx512" or SIMD == "AVX_512":
    list_simd = ["SSE2", "SSE4.1", "AVX", "AVX2", "AVX_512"]
  else :
    list_simd = [SIMD]

  for simd in list_simd :
    if str(simd) != "nsimd":
      compil_gromacs(simd, nsimd_path)
      os.system(cmd_bench)
      # The result of this command is in 
      with open('perf.out', 'r') as perf_file:
        perf = perf +"\\subsection{"+ str(simd) +"}\n" + "\\begin{lstlisting}[frame=single]\n" + perf_file.read() + "\n\\end{lstlisting}\n"
      
      # Compilation of NSIMD
      os.system("cd ../build/; cmake .. -DGMX_SIMD=NSIMD -DGMX_MPI=on -DGMX_NSIMD_PATH=" + str(nsimd_path) +" -DGMX_NSIMD_FOR="+ str(simd) +" -DCMAKE_PREFIX_PATH=" + str(nsimd_path) +"/build")
      os.system("cd ../build/; make -j30")
      os.system("cd ../build/; cmake .. -DGMX_SIMD=NSIMD -DGMX_MPI=off -DGMX_NSIMD_PATH=" + str(nsimd_path) +" -DGMX_NSIMD_FOR="+ str(simd) +" -DCMAKE_PREFIX_PATH=" + str(nsimd_path) +"/build")
      os.system("cd ../build/; make -j30; export LD_LIBRARY_PATH=" + str(nsimd_path) + "/build")
      os.system(cmd_bench)
      
      with open('perf.out', 'r') as perf_file:
        perf = perf +"\\subsection{NSIMD for " + str(simd) + "}\n" +"\\begin{lstlisting}[frame=single]\n" + perf_file.read() + "\n\\end{lstlisting}\n"
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
# compil_gromacs(options.simd, options.nsimd_path)

# Run benchmarks
benches = benchmark(options.simd, options.nsimd_path)
write_latex_file(options.simd, benches)


