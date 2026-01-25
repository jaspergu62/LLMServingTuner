#!/bin/bash
# run_vidur_search.sh - Run Vidur config optimizer search
#
# This script helps run the complete vidur search workflow:
# 1. Export benchmark trace to vidur format
# 2. Run vidur config optimizer

set -e

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDUR_DIR="${VIDUR_DIR:-/Users/hyjack/workspace/siminfer/vidur}"
OUTPUT_DIR="${OUTPUT_DIR:-./vidur_search_results}"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/vidur_search_config.yml}"
TRACE_OUTPUT="${TRACE_OUTPUT:-${VIDUR_DIR}/data/traces/benchmark_trace.csv}"

# Dataset settings (can be overridden via environment variables)
DATASET_NAME="${DATASET_NAME:-sharegpt}"
DATASET_PATH="${DATASET_PATH:-/home/airr/hyj/dataset/ShareGPT_V3_unfiltered_cleaned_split.json}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
REQUEST_RATE="${REQUEST_RATE:-100}"
TOKENIZER="${TOKENIZER:-meta-llama/Llama-2-7b-hf}"

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
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  export-trace    Export benchmark dataset to vidur trace format"
    echo "  search          Run vidur config optimizer search"
    echo "  all             Run both export-trace and search (default)"
    echo "  help            Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  VIDUR_DIR       Path to vidur directory (default: /Users/hyjack/workspace/siminfer/vidur)"
    echo "  OUTPUT_DIR      Output directory for search results (default: ./vidur_search_results)"
    echo "  CONFIG_FILE     Path to vidur search config (default: ./vidur_search_config.yml)"
    echo "  TRACE_OUTPUT    Output path for exported trace (default: \$VIDUR_DIR/data/traces/benchmark_trace.csv)"
    echo ""
    echo "  Dataset Settings:"
    echo "  DATASET_NAME    Dataset name: random, sharegpt, vidur, sonnet (default: sharegpt)"
    echo "  DATASET_PATH    Path to dataset file"
    echo "  NUM_PROMPTS     Number of prompts to export (default: 1000)"
    echo "  REQUEST_RATE    Request rate for arrival times (default: 100)"
    echo "  TOKENIZER       Tokenizer name (default: meta-llama/Llama-2-7b-hf)"
    echo ""
    echo "  Search Settings:"
    echo "  TIME_LIMIT      Search time limit in minutes (default: 60)"
    echo "  MAX_ITERATIONS  Max search iterations (default: 20)"
    echo ""
    echo "Examples:"
    echo "  # Run full workflow with default settings"
    echo "  $0 all"
    echo ""
    echo "  # Export random dataset trace"
    echo "  DATASET_NAME=random NUM_PROMPTS=500 $0 export-trace"
    echo ""
    echo "  # Run search only (if trace already exists)"
    echo "  TIME_LIMIT=30 $0 search"
}

export_trace() {
    print_header "Exporting Benchmark Trace"

    echo "Dataset: ${DATASET_NAME}"
    echo "Dataset Path: ${DATASET_PATH}"
    echo "Num Prompts: ${NUM_PROMPTS}"
    echo "Request Rate: ${REQUEST_RATE}"
    echo "Tokenizer: ${TOKENIZER}"
    echo "Output: ${TRACE_OUTPUT}"
    echo ""

    # Create output directory
    mkdir -p "$(dirname "${TRACE_OUTPUT}")"

    # Build export command
    CMD="python ${SCRIPT_DIR}/export_trace_for_vidur.py"
    CMD="${CMD} --dataset-name ${DATASET_NAME}"
    CMD="${CMD} --num-prompts ${NUM_PROMPTS}"
    CMD="${CMD} --request-rate ${REQUEST_RATE}"
    CMD="${CMD} --tokenizer ${TOKENIZER}"
    CMD="${CMD} --output ${TRACE_OUTPUT}"

    if [ -n "${DATASET_PATH}" ] && [ "${DATASET_NAME}" != "random" ]; then
        CMD="${CMD} --dataset-path ${DATASET_PATH}"
    fi

    echo "Running: ${CMD}"
    eval ${CMD}
}

run_search() {
    print_header "Running Vidur Config Optimizer Search"

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

    # Check if trace file exists
    TRACE_FILE=$(grep -m1 "trace_file:" "${CONFIG_FILE}" | awk '{print $2}' | tr -d '"')
    if [ ! -f "${VIDUR_DIR}/${TRACE_FILE}" ] && [ ! -f "${TRACE_FILE}" ]; then
        echo "Warning: Trace file not found: ${TRACE_FILE}"
        echo "You may need to run 'export-trace' first."
        echo ""
    fi

    # Create output directory
    mkdir -p "${OUTPUT_DIR}"

    # Run vidur config optimizer
    cd "${VIDUR_DIR}"

    CMD="python -m vidur.config_optimizer.config_explorer.main"
    CMD="${CMD} --config-path ${CONFIG_FILE}"
    CMD="${CMD} --output-dir ${OUTPUT_DIR}"
    CMD="${CMD} --time-limit ${TIME_LIMIT}"
    CMD="${CMD} --max-iterations ${MAX_ITERATIONS}"

    echo "Running: ${CMD}"
    eval ${CMD}

    print_header "Search Complete"
    echo "Results saved to: ${OUTPUT_DIR}"
}

# ============================================================================
# Main
# ============================================================================
COMMAND="${1:-all}"

case "${COMMAND}" in
    export-trace)
        export_trace
        ;;
    search)
        run_search
        ;;
    all)
        export_trace
        run_search
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo "Unknown command: ${COMMAND}"
        print_usage
        exit 1
        ;;
esac
