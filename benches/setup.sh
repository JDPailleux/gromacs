#!/bin/bash

print_error () {
    echo "$@" 1>&2
}

noerror () {
    "$@"
    if ! [ $? -eq 0 ]; then
        print_error "Command '$@' failed"
        exit 1
    fi
}

guess_vector_extension () {
    for EXT in avx512f avx2 avx sse4_2 sse2 asimd neon; do
        RET=$(cat /proc/cpuinfo | grep -oE "\<${EXT}\>" | head -n 1)
        if [ "${RET}" != "" ]; then
            case "${RET}" in
                asimd) echo aarch64;;
                avx512f) echo avx512-skylake;;
                sse4_2) echo sse42;;
                *) echo "${RET}";;
                esac
            break
        fi
    done
}

get_vector_extension () {
    if [ $# == 1 ] && [ -n "${1}" ] ; then
        if vector_extension_supported "${1}"; then
            if [ "${1##nsimd-}" == "auto" ]; then
                echo $(echo $1 | grep -o "nsimd-")$(guess_vector_extension)
            else
                echo "${1}"
            fi
        else
            print_error -e "Unsupported SIMD extension: \033[1;31m${1}\033[0m, try one of sse2, sse42, avx, avx2, avx512_skylake, neon, aarch64, auto."
            return 1
        fi
    else
        print_error "Expects exactly one of (nsimd-)sse2, sse42, avx, avx2, avx512_skylake, neon, aarch64, auto."
        return 1
    fi
}

vector_extension_supported () {
    echo "${1}" | grep -Ei "^(nsimd-)?(sse2|sse42|avx|avx2|avx512-skylake|aarch64|auto)$" 1> /dev/null
    return $?
}

return_yes () {
    echo "yes"
}

return_no () {
    echo "no"
}

print_result () {
    "$@"
    if [ $? -eq 0 ] ; then
        return_yes
    else
        return_no
    fi
}

silent () {
    "$@" &> /dev/null
}

check () {
    case "${1}" in
        nsconfig)
            silent which nsconfig;;
        nsimd)
            nsimd_is_installed "${2}";;
        gromacs)
            gromacs_is_installed "${2}";;
        *)
            print_error "Unknown installation ${@}"
            false;;
    esac
    status=$?
    if ! [ "${status}" == "0" ] ; then
        print_error "Could not find ${1}"
    else
        echo "${1} is already installed"
    fi
    return "$status"
}


clean () {
    rm -rf nsimd gromacs
}

download_project () {
    if ! [ -d "${1%%-*}" ]; then
        case "${1}" in
            nsimd-private) ADDRESS="ssh://git@phabricator2.numscale.com/diffusion/67/nsimd.git";;
            nsimd-public) ADDRESS="https://github.com/agenium-scale/nsimd.git";;
            gromacs) ADDRESS="https://github.com/agenium-scale/gromacs.git";;
            ns2) ADDRESS="ssh://git@phabricator2.numscale.com/source/ns2.git";;
            nsconfig) ADDRESS="ssh://git@phabricator2.numscale.com/source/nsconfig.git"
        esac
        git clone "${ADDRESS}"
        STATUS="$?"
        if ! [ ${STATUS} -eq  0 ]; then
            print_error "Could not clone ${1} error: ${GITERR}"
            return -1
        fi
    else
        silent pushd "${1%%-*}"
        git fetch --all
        SUM_HEAD=$(git rev-parse HEAD)
        SUM_BRANCH=$(git rev-parse origin/${2:-master})
        if ! [ "${SUM_HEAD}" == "${SUM_BRANCH}" ]; then
            git stash
            git pull -q
            git stash pop
        fi
        silent popd
    fi

}

gromacs_is_cloned () {
    [ -d "gromacs" ]
}

# param: one of nsimd, sse2, sse42, avx, avx2, avx512
gromacs_is_installed () {
    VECTOR_EXTENSION=$(get_vector_extension "${1}")
    [ -z "${VECTOR_EXTENSION}" ] && return 1
    BUILDDIR="gromacs/build-${VECTOR_EXTENSION}"

    if [ -e "${BUILDDIR}/bin/gmx" ] && [ -e "${BUILDDIR}/bin/gmx_mpi" ] \
     && [ -e "${BUILDDIR}/bin/template" ]; then
        return 0
    fi
    return 1
}

nsimd_is_cloned () {
    [ -d "nsimd" ]
}

# param one of sse2, sse42, avx, avx2, avx512
nsimd_is_installed () {
    silent ls nsimd/build/libnsimd_*.so
    FOUND="$?"
    if [ ${FOUND} -ne 0 ]; then
        return 1
    fi


    LIB=$(ls nsimd/build/libnsimd_*.so | head -n 1)
    if [ $# -eq 1 ]; then
        VECTOR_EXTENSION=$(get_vector_extension "${1}")
        if [ $? -eq 0 ]; then
            nm ${LIB} | grep "nsimd_put_${VECTOR_EXTENSION}" > /dev/null
            return
        else
            return 1
        fi
    fi

    return 0
}


nsimd_simd_flag () {
    case "${1#nsimd-}" in
        sse2 | sse42 | avx | avx2) echo "${1#nsimd-}";;
        aarch64) echo "${1#nsimd-}" -Dgen_latest_simd=off;;
        avx512-skylake) echo avx512_skylake;;
        *) print_error "Unknown SIMD flag to build nsimd for ${1}."; return 1;;
    esac
}

# param one of sse2, sse42, avx, avx2, avx512-skylake, aarch64
nsimd_install () {
    check nsimd "${1##nsimd-}" && return 0
    check nsconfig || return 1


    noerror download_project nsimd-private
    silent pushd nsimd
    noerror download_project nsimd-public
    if ! grep 'optional mpfr' build.nsconfig &> /dev/null; then
        git apply ../../nsimd.patch
        if ! [ $? -eq 0 ]; then
            print_error "Could not patch nsimd"
            return 1
        fi
    fi
    silent popd

    VECTOR_EXTENSION=$(get_vector_extension "${1##nsimd-}")
    [ -z "${VECTOR_EXTENSION}" ] && return 1

    if nsimd_is_installed ${VECTOR_EXTENSION}; then
        echo -e "\033[1;33mNSIMD for ${VECTOR_EXTENSION} is already installed\033[0m"
        return 0
    fi

    silent pushd nsimd
    echo "Installing nsimd for ${VECTOR_EXTENSION}"
    rm -rf build
    mkdir build
    python3 egg/hatch.py -acCos
    cd build
    # cmake .. -DSIMD=${VECTOR_EXTENSION^^}
    # make
    noerror nsconfig .. -Dsimd=$(nsimd_simd_flag ${VECTOR_EXTENSION})
    ninja
    silent popd
}

gromacs_nsimd_flag () {
    case "${1}" in
        nsimd-sse2)  echo SSE2;;
        nsimd-sse42) echo SSE4.1;;
        nsimd-avx)   echo AVX;;
        nsimd-avx2)  echo AVX2;;
        nsimd-avx512-skylake) echo AVX_512;;
        aarch64) echo AARCH64;;
        *) print_error "Cannot translate SIMD extension '${1}' for NSIMD."; return 1;;
    esac
}

gromacs_gmx_simd_flag () {
    case "${1}" in
        sse2)  echo SSE2;;
        sse42) echo SSE4.1;;
        avx)   echo AVX_256;;
        avx2)  echo AVX2_256;;
        avx512-skylake) echo AVX_512;;
        avx512-knl) echo AVX_512_KNL;;
        aarch64) echo ARM_NEON_ASIMD;;
        nsimd-*) echo NSIMD;;
        *) print_error "Cannot translate SIMD extension '${1}' for GROMACS."; return 1;;
    esac
}

gromacs_install () {
    VECTOR_EXTENSION=$(get_vector_extension "${1}")
    [ -z "${VECTOR_EXTENSION}" ] && return 1

    silent check gromacs ${VECTOR_EXTENSION} && return 0

    download_project gromacs nsimd-translate

    silent pushd gromacs
    git checkout nsimd-translate
    git pull -q
    silent popd

    BUILDDIR=build-${VECTOR_EXTENSION}

    ROOT=$(pwd)

    if ! [ -d "gromacs/${BUILDDIR}" ]; then
        mkdir "gromacs/${BUILDDIR}"
    fi

    pushd "gromacs/${BUILDDIR}"

    echo ${VECTOR_EXTENSION} | grep "nsimd-" &>/dev/null
    if [ $? -eq 0 ]; then
        NSIMD_CMAKE_OPTS="-DGMX_NSIMD_PATH=${ROOT}/nsimd -DGMX_NSIMD_FOR=$(gromacs_nsimd_flag ${VECTOR_EXTENSION}) -DCMAKE_PREFIX_PATH=${ROOT}/nsimd/build"
    fi

    for GMX_MPI in on off; do
        cmake .. -DGMX_SIMD=$(gromacs_gmx_simd_flag ${VECTOR_EXTENSION}) -DGMX_MPI=${GMX_MPI} ${NSIMD_CMAKE_OPTS} || return 0
        make -j $(nproc) || return 0
    done
    popd
}

nsconfig_install () {
    silent check nsconfig && return 0

    noerror download_project ns2
    noerror download_project nsconfig
    pushd nsconfig
    make -f Makefile.nix -j 4
    popd
}

bench () {
    VECTOR_EXTENSION=$(get_vector_extension "${1}")
    [ -z "${VECTOR_EXTENSION}" ] && return 1

    ${CXX-g++} --version &> "${VECTOR_EXTENSION}-compiler.info"
    uname -a &> "${VECTOR_EXTENSION}-system.info"
    lscpu &> "${VECTOR_EXTENSION}-cpu.info"
    cat /proc/meminfo &> "${VECTOR_EXTENSION}-mem.info"
    ldd --version &> "${VECTOR_EXTENSION}-ldd.info"

    BUILDDIR="build-${VECTOR_EXTENSION}"
    export LD_LIBRARY_PATH="${PWD}/nsimd/build:${LD_LIBRARY_PATH}"
    export PATH="${PWD}/gromacs/${BUILDDIR}/bin:${PATH}"
    OMP_NUM_THREADS=24 gmx tune_pme -r 10 -s gromacs/topol.tpr -mdrun "gromacs/${BUILDDIR}/bin/gmx_mpi mdrun"
    mv perf.out "${VECTOR_EXTENSION}-perf.out"
    rm -f bench*
}

autorun () {
    SIMD=$1
    noerror nsconfig_install
    noerror nsimd_install ${SIMD}
    noerror gromacs_install ${SIMD}
    noerror bench ${SIMD}
}

run() {
    CONF_FILE="${HOME}/.config/gromacs-bench/$(hostname).sh"

    if ! [ -e "${CONF_FILE}" ]; then
        print_error "Configuration file '${CONF_FILE}' not found"
    elif ! source ${CONF_FILE}; then
        print_error "There was an error sourcing the configuration file '${CONFIG_FILE}'"
        return 1
    fi
    echo "$@"
    noerror cd ${GROMACS_ROOT_DIR}/gromacs/bench

    export PATH="${PWD}/nsconfig":${PATH}

    echo "${1}" | grep -E "((guess|get)_vector_extension|(nsimd|gromacs|nsconfig)_install|nsimd_is_installed|gromacs_n?simd|clean|bench|autorun)" > /dev/null
    if [ "$?" == 0 ]; then
        print_result "${@}"
    else
        print_error "Unknown command ${1}"
    fi
}

run "$@"
