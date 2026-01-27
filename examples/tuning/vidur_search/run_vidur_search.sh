#!/bin/bash
# run_vidur_search.sh - Run Vidur config optimizer search

set -e

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDUR_DIR="${VIDUR_DIR:-/cache/simserving/vidur-siminfer}"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/vidur_search_config.yml}"

# Convert OUTPUT_DIR to absolute path (to avoid it being relative to VIDUR_DIR after cd)
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/vidur_search_results}"
OUTPUT_DIR="$(cd "$(dirname "${OUTPUT_DIR}")" 2>/dev/null && pwd)/$(basename "${OUTPUT_DIR}")" || OUTPUT_DIR="${SCRIPT_DIR}/vidur_search_results"

# Search settings
TIME_LIMIT="${TIME_LIMIT:-60}"  # minutes
MAX_ITERATIONS="${MAX_ITERATIONS:-20}"

# ============================================================================
# Functions
# ============================================================================
print_header() {
    echo ""
    echo "============================================================================"
    echo " $1"
    echo "============================================================================"
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  VIDUR_DIR         Path to vidur directory (default: /cache/simserving/vidur-siminfer)"
    echo "  OUTPUT_DIR        Output directory for search results (default: ./vidur_search_results)"
    echo "  CONFIG_FILE       Path to vidur search config (default: ./vidur_search_config.yml)"
    echo "  TIME_LIMIT        Search time limit in minutes (default: 60)"
    echo "  MAX_ITERATIONS    Max search iterations (default: 20)"
    echo ""
    echo "Example:"
    echo "  TIME_LIMIT=30 $0"
}

run_search() {
    print_header "Running Vidur Config Optimizer Search"

    echo "Vidur Dir: ${VIDUR_DIR}"
    echo "Config File: ${CONFIG_FILE}"
    echo "Output Dir: ${OUTPUT_DIR}"
    echo "Time Limit: ${TIME_LIMIT} minutes"
    echo "Max Iterations: ${MAX_ITERATIONS}"
    echo ""

    # Check if config file exists
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "Error: Config file not found: ${CONFIG_FILE}"
        exit 1
    fi

    # Create output directory
    mkdir -p "${OUTPUT_DIR}"

    # Run vidur config optimizer from vidur directory (for relative data paths)
    cd "${VIDUR_DIR}"

    python -m vidur.config_optimizer.config_explorer.main \
        --config-path "${CONFIG_FILE}" \
        --output-dir "${OUTPUT_DIR}" \
        --time-limit "${TIME_LIMIT}" \
        --max-iterations "${MAX_ITERATIONS}"

    print_header "Search Complete"
    echo "Results saved to: ${OUTPUT_DIR}"
}

# ============================================================================
# Main
# ============================================================================
case "${1:-}" in
    -h|--help|help)
        print_usage
        ;;
    *)
        run_search
        ;;
esac
