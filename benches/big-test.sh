#!/bin/sh

cd `dirname $0`
TIMESTAMP=`date +%Y%m%d%H%M%S`
DATE=`date +%d/%m/%Y-%H:%M:%S`
DEST_DIR="_benches"

# Get list of machines 
# SSE2 SSE42 AVX AVX2 AVX512
# MACHINES="camelot benoic logres gaunes glastonbury"
MACHINES="glastonbury"


# If the machines.txt doesn't exist
if [ -e "machines.txt" ]; then
  MACHINES=`cat machines.txt`
fi

# Clean repository
if [ "$1" = "clean" ]; then
  # Clean all files on all machines
  for m in ${MACHINES}; do
    echo "==> clean ${m}"
    ssh jdpailleux@${m} "rm -rf ${DEST_DIR}/*"
    ssh jdpailleux@${m} "rm -rf ${DEST_DIR}/gromacs/benches/topol/topol.tpr"
  done
  exit 0
fi

if [ "$1" = "ps" ]; then
  # List of process
  ssh -t jdpailleux@$2 "watch -n 1 pstree -A"
  exit 0
fi


# Execute benchmarks
for m in ${MACHINES}; do
  echo "==> to ${m}"
  ssh -t jdpailleux@${m} "mkdir -p ${DEST_DIR}; cd ${DEST_DIR}"
    
  # Prepare environment
  # if [ -d "gromacs/" ]; then
  #   rm -rf gromacs/
  # fi

  # if [ -d "nsimd/" ]; then
  #   rm -rf nsimd/
  # fi

  # (git clone git@github.com:agenium-scale/gromacs.git)
  # (git clone ssh://git@phabricator2.numscale.com/diffusion/67/nsimd.git)
  
  # Create gromacs and nsimd archives
  # tar -cvzf gromacs.tar.gz gromacs/
  # tar -cvzf nsimd.tar.gz nsimd/

  # scp gromacs.tar.gz jdpailleux@${m}:${DEST_DIR}
  # scp nsimd.tar.gz jdpailleux@${m}:${DEST_DIR}
  # ssh jdpailleux@${m} "cd ${DEST_DIR}; tar -xzvf gromacs.tar.gz; tar -xzvf nsimd.tar.gz; rm *.tar.gz" #
  
  NSIMD_PATH=$(ssh jdpailleux@${m} "cd ${DEST_DIR}/nsimd/; pwd")

    # Depending on which machine we are, set GMX_SIMD flags properly
    if [ "${m}" = "camelot" ]; then
      GMX_SIMD="SSE2 SSE4.1 AVX_256"
    elif [ "${m}" = "benoic" ]; then
      GMX_SIMD="SSE2 SSE4.1 AVX_256"
    elif [ "${m}" = "glastonbury" ]; then
      GMX_SIMD="SSE2 SSE4.1 AVX_256 AVX2_256 AVX_512"
    elif [ "${m}" = "gaunes" ]; then
      GMX_SIMD="SSE2 SSE4.1 AVX_256 AVX2_256"
    else
      GMX_SIMD="SSE2 SSE4.1 AVX_256"
    fi

    ssh jdpailleux@${m} "mkdir -p ${DEST_DIR}/gromacs/build;"
   
    # Compilation and benches
    for SIMD in ${GMX_SIMD}; do
      echo "==> Benches ${SIMD}"
      ssh jdpailleux@${m} """cd ${DEST_DIR}/gromacs/build; 
      cd ../benches;
      python3 do-benches.py --simd=${SIMD};"""
          
    done

    # Compilation and benches for NSIMD
    GMX_NSIMD_PATH="${NSIMD_PATH}"
    echo "==> Benches NSIMD"
    ssh jdpailleux@${m}  """ cd ${DEST_DIR}/gromacs/build; 
    cd ../scripts;
    python3 do-benches.py --simd=nsimd --nsimd_path=${GMX_NSIMD_PATH}; """
    
  exit

done
