#!/bin/bash
# Wrapper around lm-evaluation-harness for running benchmark evaluations.
# Supports vLLM and HF backends.

set -e

# Default values
MODEL="${MODEL:-Qwen/Qwen3-8B}"
TASKS="${TASKS:-mmlu_pro}"
LIMIT="${LIMIT:-10}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
GPU="${GPU:-}"
TOKENIZER="${TOKENIZER:-}"
BACKEND="${BACKEND:-vllm}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --tp)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL       Model name/path (default: Qwen/Qwen3-8B)"
            echo "  --tasks TASKS       Task(s) to run (default: mmlu_pro)"
            echo "  --limit N           Limit samples per task (default: 10)"
            echo "  --batch-size N      Batch size (default: auto)"
            echo "  --tp N              Tensor parallel size (default: 1)"
            echo "  --output DIR        Output directory (default: ./results)"
            echo "  --gpu ID            GPU ID to use, e.g. 0, 1, or 0,1 for multi-GPU (default: all visible)"
            echo "  --tokenizer NAME    Separate tokenizer to use (useful for merged models)"
            echo "  --backend BACKEND   Backend to use: vllm or hf (default: vllm)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  MODEL, TASKS, LIMIT, BATCH_SIZE, TENSOR_PARALLEL, OUTPUT_DIR, GPU, TOKENIZER, BACKEND"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "LM Evaluation Harness - Sanity Test"
echo "============================================"
echo "Model:           $MODEL"
echo "Tasks:           $TASKS"
echo "Limit:           ${LIMIT:-all} samples"
echo "Batch size:      $BATCH_SIZE"
echo "Tensor parallel: $TENSOR_PARALLEL"
echo "GPU:             ${GPU:-all visible}"
echo "Tokenizer:       ${TOKENIZER:-same as model}"
echo "Backend:         $BACKEND"
echo "Output:          $OUTPUT_DIR"
echo "============================================"
echo ""

# Set GPU if specified
if [[ -n "$GPU" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# Build model_args based on backend
if [[ "$BACKEND" == "vllm" ]]; then
    MODEL_ARGS="pretrained=$MODEL,tensor_parallel_size=$TENSOR_PARALLEL,dtype=auto,trust_remote_code=true"
    if [[ -n "$TOKENIZER" ]]; then
        MODEL_ARGS="$MODEL_ARGS,tokenizer=$TOKENIZER"
    fi
elif [[ "$BACKEND" == "hf" ]]; then
    MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16"
    if [[ -n "$TOKENIZER" ]]; then
        MODEL_ARGS="$MODEL_ARGS,tokenizer=$TOKENIZER"
    fi
else
    echo "Error: Unknown backend '$BACKEND'. Use 'vllm' or 'hf'."
    exit 1
fi

# Build limit argument (skip if "all" or empty to run all samples)
LIMIT_ARG=""
if [[ -n "$LIMIT" && "$LIMIT" != "all" && "$LIMIT" != "-1" ]]; then
    LIMIT_ARG="--limit $LIMIT"
fi

# Run evaluation
lm_eval \
    --model "$BACKEND" \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    $LIMIT_ARG \
    --output_path "$OUTPUT_DIR" \
    --log_samples

echo ""
echo "============================================"
echo "Sanity test complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================"
