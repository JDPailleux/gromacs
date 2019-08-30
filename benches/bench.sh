RESOURCE_ROOT=".."
MACHINES_FILE="${RESOURCE_ROOT}/machines.txt"
SETUP_SCRIPT="${RESOURCE_ROOT}/setup.sh"
NSIMD_PATCH="${RESOURCE_ROOT}/nsimd.patch"

print_error () {
    echo -e "\033[1;31m$@\033[0m" 1>&2
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

    IFS=" " read -r SSH_HOST SIMD BENCH_ROOT_DIR <<< $line

    echo "${SSH_HOST?Could not read ssh host from the ${MACHINES_FILE} file}"
    echo "${SIMD?Could not read the SIMD configuration from the ${MACHINES_FILE} file}"

    REMOTE_HOSTNAME=$(ssh "${SSH_HOST}" "hostname")
    BENCH_ROOT_DIR=$(grep BENCH_ROOT_DIR "${RESOURCE_ROOT}/${REMOTE_HOSTNAME}.sh" | cut -d= -f 2)

    if [ "${BENCH_ROOT_DIR}" == "" ]; then
        print_error "Could not get configuration for host: ${SSH_HOST}"
        print_error "You must create file ${RESOURCE_ROOT}/${REMOTE_HOSTNAME}.sh and define the BENCH_ROOT_DIR variable in it."
        continue
    fi

    if ! ssh "${SSH_HOST}" "[ -d ${BENCH_ROOT_DIR} ]"; then
        print_error "Root directory '${BENCH_ROOT_DIR}' does not exist on ${SSH_HOST}"
        continue
    fi

    if ! [ -z "${REMOTE_HOSTNAME}" ] && [ -e "${RESOURCE_ROOT}/${REMOTE_HOSTNAME}.sh" ]; then
        rsync "${RESOURCE_ROOT}/${REMOTE_HOSTNAME}.sh" "${SSH_HOST}:${BENCH_ROOT_DIR}/gromacs"
    fi

    ssh "${SSH_HOST}" "mkdir -p ${BENCH_ROOT_DIR}/gromacs/bench" || continue
    rsync "${SETUP_SCRIPT}" "${SSH_HOST}:${BENCH_ROOT_DIR}/gromacs" ||  continue
    rsync "${NSIMD_PATCH}" "${SSH_HOST}:${BENCH_ROOT_DIR}/gromacs" || continue
    # Use a here string to get an interactive shell
    ssh "${SSH_HOST}" <<< "bash ${BENCH_ROOT_DIR}/gromacs/setup.sh autorun ${SIMD}" || continue
    rsync "${SSH_HOST}:${BENCH_ROOT_DIR}/gromacs/bench/${SIMD}*" . || continue
done 3< "${MACHINES_FILE}"

python3 ${RESOURCE_ROOT}/graph.py




