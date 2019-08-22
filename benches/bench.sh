MACHINES_FILE="machines.txt"
SETUP_SCRIPT="../setup.sh"
NSIMD_PATCH="../nsimd.patch"

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

# Check configuration file
if ! [ -f "${MACHINES_FILE}" ]; then
    print_error "\"${MACHINES_FILE}\" file not found"
    exit 1
fi

# Read configuration file
#   - one line per test: 'ssh address' 'vector extension'
while read -u 3 -r line; do
    if [ "${line:0:1}" == "#" ]; then
        continue
    fi

    IFS=" " read -r SSH_HOST SIMD  <<< $line

    echo "${SSH_HOST}"
    echo "${SIMD}"

    # Check root dir for the machine
    BENCH_ROOT_DIR=$(ssh "${SSH_HOST}" "grep GROMACS_ROOT_DIR ~/.config/gromacs-bench/\$(hostname).sh | cut -d= -f 2")
    if [ "${BENCH_ROOT_DIR}" == "" ]; then
        print_error "Could not get configuration for host: ${SSH_HOST}"
        continue
    fi

    if ! ssh "${SSH_HOST}" "[ -d ${BENCH_ROOT_DIR} ]"; then
        print_error "Root directory '${BENCH_ROOT_DIR}' does not exist on ${SSH_HOST}"
        continue
    fi

    ssh "${SSH_HOST}" "mkdir -p ${BENCH_ROOT_DIR}/gromacs/bench" || continue
    rsync "${SETUP_SCRIPT}" "${SSH_HOST}:${BENCH_ROOT_DIR}/gromacs" || continue
    rsync "${NSIMD_PATCH}" "${SSH_HOST}:${BENCH_ROOT_DIR}/gromacs" || continue
    # Use a here string to get an interactive shell
    ssh "${SSH_HOST}" <<< "bash ${BENCH_ROOT_DIR}/gromacs/setup.sh autorun ${SIMD}" || continue
    rsync ${SSH_HOST}:${BENCH_ROOT_DIR}/gromacs/bench/perf-${SIMD}.out . || continue
    rsync "${SSH_HOST}:${BENCH_ROOT_DIR}/gromacs/bench/*.info" . || continue
done 3< "${MACHINES_FILE}"






