#!/bin/bash


## Output to stderr
stderr() {
    echo "$@" > /dev/stderr
}

## Usage
if [ ! -e "$1" ]; then
    stderr "usage: $0 <translator>"
    exit 1
fi

## Get project root path
GROMACS_DIR="$( git rev-parse --show-toplevel )"

## Translator setup
TRANSLATOR="$1"
TRANSLATOR_OPTIONS="\
  -extra-arg=-std=c++11 \
  -extra-arg=-I${GROMACS_DIR}/build/src/ \
  -extra-arg=-I${GROMACS_DIR}/src \
"
TRANSLATOR_REPORT="${GROMACS_DIR}/NSIMD-TRANSLATED.md"


## Apply translator to given SIMD impl and create translated files
translate() {
    SIMD=$1
    IMPL=$2
    SIMD_IMPL_PREFIX="${GROMACS_DIR}/src/gromacs/simd/${IMPL}/${IMPL}"
    SIMD_SUFFIX="
    _util_float
    _util_double
    _simd_float
    _simd_double
    _simd4_float
    _simd4_double

    "
    for suffix in ${SIMD_SUFFIX}; do
        ## Craft header paths
        H="${SIMD_IMPL_PREFIX}${suffix}.h"
        H_TRANSLATED="${SIMD_IMPL_PREFIX}${suffix}.nsimd.h"
        ## Check if file exists
        if [ -e "${H}" ]; then
            echo "** translating: ${H}" > /dev/stderr
            ## Translate and output to a .nsimd.h (to keep track of what has been translated) 
            echo ${TRANSLATOR} -simd=${SIMD} ${H} -o=${H_TRANSLATED}
            ${TRANSLATOR} -simd=${SIMD} ${TRANSLATOR_OPTIONS} ${H} -o=${H_TRANSLATED}
        else
            stderr "** error: no such file: ${H}"
        fi
    done
}

report() {
    SIMD_MARKERS="_mm"
    echo "Translation report:"
    echo "==================="
    echo ""
    for f in $( find ${GROMACS_DIR} -name "*.nsimd.h" ); do
        H="$( echo ${f} | sed 's/\.nsimd.h/.h/g' )"
        H_COUNT="$( grep -E ${SIMD_MARKERS} ${H} | wc -l )"
        H_TRANSLATED_COUNT="$( grep -E ${SIMD_MARKERS} ${f} | wc -l )"
        ## Use scale so can use decimal operations
        R="$( echo "scale=2; ((${H_COUNT} - ${H_TRANSLATED_COUNT}) / ${H_COUNT}) * 100" | bc )"
        H_PRETTY="$( echo ${H} | sed "s#${GROMACS_DIR}/##" )"
        echo "${H_PRETTY}: ${R}% (original=${H_COUNT}, translated=${H_TRANSLATED_COUNT})"
    done
}

## Translate everything
translate avx impl_x86_avx2_256
translate avx impl_x86_avx_256



## Now generate report
report > "${TRANSLATOR_REPORT}"

## Print the report
echo ""
cat "${TRANSLATOR_REPORT}"
