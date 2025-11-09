#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU Per-Model Layer Sweep for Knowledge Triple Probing Framework
# =====================================================================
# 
# This script performs comprehensive layer analysis across:
# - Multiple probing systems (LRC, Circular, MLP, Logistic)
# - Multiple models (general and biomedical LLMs)
# - Multiple layer combinations
# - One model at a time, with all probing systems running simultaneously on different GPUs
#
# PERFORMANCE FIXES APPLIED:
# - Fixed GPU device specification: now uses "cuda:$GPU_ID" instead of "cuda:0"
# - Removed CUDA_VISIBLE_DEVICES override to prevent GPU confusion
# - Added timing measurements for performance monitoring
# - Reduced sleep delays for faster execution
#
# Execution Strategy:
# 1. Process one model at a time
# 2. For each model, run all probing systems simultaneously on different GPUs
# 3. Wait for all systems to complete before moving to the next model
#
# Parallel Execution Pattern (per model):
# GPU 0 → circular system → all layers for current model
# GPU 1 → mlp system → all layers for current model  
# GPU 2 → logistic system → all layers for current model
# (All 3 systems run simultaneously for the same model)
#
# Usage:
#   ./run_layer_sweep.sh [options]
#
# Options:
#   --systems "lrc circular"     # Space-separated list of systems
#   --models "model1 model2"     # Space-separated list of models
#   --gpus "0 1 2"              # GPU IDs to use for systems
#   --subject-start 10          # Starting subject layer
#   --subject-end 25            # Ending subject layer
#   --object-offset 2           # Object layer = subject + offset
#   --output-dir results/       # Output directory

# Default configuration
SYSTEMS=("circular" "mlp" "logistic") # "lrc" 
GPUS=(0 1 2)  # One GPU per system

# Model configurations - Combined models for sequential execution
ALL_MODELS=(
    # "google/gemma-2-9b"
    # "meta-llama/Meta-Llama-3-8B"
    # "mistralai/Mistral-7B-Instruct-v0.1"
    # "meta-llama/Llama-3.1-8B"
    # "google/gemma-3-4b-pt"
    "OpenMeditron/Meditron3-Gemma2-9B"
    "Henrychur/MMed-Llama-3-8B"
    "BioMistral/BioMistral-7B"
    # "google/medgemma-4b-pt"
    # "google/medgemma-4b-it"
    # "TsinghuaC3I/Llama-3.1-8B-UltraMedical"
)

# For backward compatibility, keep separate arrays but they'll use the combined list
GENERAL_MODELS=("${ALL_MODELS[@]}")
BIOMEDICAL_MODELS=("${ALL_MODELS[@]}")

# Layer sweep parameters
SUBJECT_LAYER_START=10
SUBJECT_LAYER_END=25
LAYER_OFFSET=2  # object_layer = subject_layer + offset

# Output configuration (simplified): save CSVs directly in sota_result at repo root
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="sota_results"
RESULTS_DIR="${OUTPUT_DIR}"  # Direct CSV output here

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --systems)
            IFS=' ' read -ra SYSTEMS <<< "$2"
            shift 2
            ;;
        --models)
            IFS=' ' read -ra MODELS <<< "$2"
            shift 2
            ;;
        --gpus)
            IFS=' ' read -ra GPUS <<< "$2"
            shift 2
            ;;
        --subject-start)
            SUBJECT_LAYER_START="$2"
            shift 2
            ;;
        --subject-end)
            SUBJECT_LAYER_END="$2"
            shift 2
            ;;
        --object-offset)
            LAYER_OFFSET="$2"
            shift 2
            ;;
        --output-dir)
            # Force output to sota_result folder regardless of argument
            OUTPUT_DIR="sota_result"
            RESULTS_DIR="${OUTPUT_DIR}"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate GPU count vs system count
if [ ${#GPUS[@]} -lt ${#SYSTEMS[@]} ]; then
    echo "⚠️ Warning: More systems (${#SYSTEMS[@]}) than GPUs (${#GPUS[@]})"
    echo "   Some systems will share GPUs, which may impact performance"
fi

# Create output directory only
mkdir -p "$RESULTS_DIR"

# No folder scaffolding; only ensure OUTPUT_DIR exists (created above)

# Function to get model display name
get_model_display_name() {
    local model="$1"
    case "$model" in
        "google/gemma-2-9b") echo "Gemma-2-9B" ;;
        "google/gemma-3-4b-pt") echo "Gemma-3-4B" ;;
        "google/medgemma-4b-pt") echo "MedGemma-4B" ;;
        "google/medgemma-4b-it") echo "MedGemma-4B-IT" ;;
        "meta-llama/Meta-Llama-3-8B") echo "Llama-3-8B" ;;
        "meta-llama/Llama-3.1-8B") echo "Llama-3.1-8B" ;;
        "Henrychur/MMed-Llama-3-8B") echo "MMed-Llama-3-8B" ;;
        "TsinghuaC3I/Llama-3.1-8B-UltraMedical") echo "UltraMed-Llama-3.1-8B" ;;
        "mistralai/Mistral-7B-Instruct-v0.1") echo "Mistral-7B-Instruct" ;;
        "BioMistral/BioMistral-7B") echo "BioMistral-7B" ;;
        "OpenMeditron/Meditron3-Gemma2-9B") echo "Meditron3-Gemma2-9B" ;;
        *) echo "${model##*/}" ;;  # Extract last part after /
    esac
}

# Function to run experiments for one system and one model
run_system_model() {
    local SYSTEM="$1"
    local GPU_ID="$2"
    local MODEL="$3"
    local MODEL_SET="$4"  # "general" or "biomedical"
    local MODEL_NUM="$5"  # Model number in the set
    local TOTAL_MODELS="$6"  # Total number of models in this phase
    
    local MODEL_DISPLAY=$(get_model_display_name "$MODEL")
    # Simplified logging: do not create log files
    local SYSTEM_LOG="/dev/null"
    local PHASE_NAME=$(echo "$MODEL_SET" | tr '[:lower:]' '[:upper:]')
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $PHASE_NAME phase: $SYSTEM on GPU $GPU_ID for $MODEL_DISPLAY" | tee -a "$SYSTEM_LOG"
    echo "[INFO] Layer sweep: subject layers ${SUBJECT_LAYER_START}-${SUBJECT_LAYER_END}, object offset +${LAYER_OFFSET}" | tee -a "$SYSTEM_LOG"
    echo "[INFO] Model: $MODEL_DISPLAY (${MODEL_NUM}/${TOTAL_MODELS})" | tee -a "$SYSTEM_LOG"
    
    (
        # Remove CUDA_VISIBLE_DEVICES override to avoid GPU confusion
        export CUDA_LAUNCH_BLOCKING=1  # For better error messages
    
    local TOTAL_LAYERS=$((SUBJECT_LAYER_END - SUBJECT_LAYER_START + 1))
    local EXPERIMENT_COUNT=0
    
    for SUBJECT_LAYER in $(seq $SUBJECT_LAYER_START $SUBJECT_LAYER_END); do
        OBJECT_LAYER=$((SUBJECT_LAYER + LAYER_OFFSET))
        EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
        PROGRESS=$((EXPERIMENT_COUNT * 100 / TOTAL_LAYERS))
        
        echo "" | tee -a "$SYSTEM_LOG"
        echo "[EXPERIMENT ${EXPERIMENT_COUNT}/${TOTAL_LAYERS} (${PROGRESS}%)]" | tee -a "$SYSTEM_LOG"
        echo "  System: $SYSTEM" | tee -a "$SYSTEM_LOG"
        echo "  Model: $MODEL_DISPLAY" | tee -a "$SYSTEM_LOG"
        echo "  Layers: S${SUBJECT_LAYER} → O${OBJECT_LAYER}" | tee -a "$SYSTEM_LOG"
        echo "  GPU: $GPU_ID" | tee -a "$SYSTEM_LOG"
        echo "  Phase: $PHASE_NAME" | tee -a "$SYSTEM_LOG"
        echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$SYSTEM_LOG"
        
        # Run the evaluation with proper GPU specification
        START_TIME=$(date +%s)
        if PYTHONPATH=/home/avv533/knowledge python /home/avv533/knowledge/main_probing.py \
            --system "$SYSTEM" \
            --model "$MODEL" \
            --subject-layers "$SUBJECT_LAYER" \
            --object-layers "$OBJECT_LAYER" \
            --device "cuda:$GPU_ID" \
            --output-dir "$RESULTS_DIR"; then
            
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))
            echo "[SUCCESS] Completed: ${SYSTEM} | ${MODEL_DISPLAY} | S${SUBJECT_LAYER}→O${OBJECT_LAYER} | ${PHASE_NAME} | Duration: ${DURATION}s" | tee -a "$SYSTEM_LOG"
            
            # Check if CSV was generated and log it
            CSV_FILES=$(find "$RESULTS_DIR" -name "*.csv" -newer "$SYSTEM_LOG" 2>/dev/null | wc -l)
            if [ "$CSV_FILES" -gt 0 ]; then
                echo "[CSV] Generated ${CSV_FILES} CSV file(s) for this experiment" | tee -a "$SYSTEM_LOG"
            fi
        else
            echo "[ERROR] Failed: ${SYSTEM} | ${MODEL_DISPLAY} | S${SUBJECT_LAYER}→O${OBJECT_LAYER} | ${PHASE_NAME}" | tee -a "$SYSTEM_LOG"
            echo "[ERROR] Check log for details: $SYSTEM_LOG" | tee -a "$SYSTEM_LOG"
        fi
        
        # Brief pause to avoid overwhelming the system
        sleep 1
    done
        
        echo "" | tee -a "$SYSTEM_LOG"
        echo "[MODEL COMPLETE] $SYSTEM finished all layers for $MODEL_DISPLAY ($PHASE_NAME phase) on GPU $GPU_ID" | tee -a "$SYSTEM_LOG"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed $MODEL_DISPLAY: $SYSTEM" | tee -a "$SYSTEM_LOG"
        
    ) &
}

# Function to wait for all processes to complete (keep minimal status)
wait_for_processes() {
    local PIDS=("$@")
    local PIDS_NAMES=("${@:$((${#PIDS[@]}+1))}")
    
    echo ""
    echo "[INFO] Waiting for ${#PIDS[@]} processes to complete..."
    echo "[INFO] Output CSVs in: ${RESULTS_DIR}"
    
    # Wait for all processes with progress monitoring
    local RUNNING=1
    while [ $RUNNING -eq 1 ]; do
        RUNNING=0
        echo -n "[$(date '+%H:%M:%S')] Status: "
        
        for i in "${!PIDS[@]}"; do
            PID="${PIDS[$i]}"
            NAME="${PIDS_NAMES[$i]}"
            
            if kill -0 "$PID" 2>/dev/null; then
                echo -n "$NAME(running) "
                RUNNING=1
            else
                echo -n "$NAME(done) "
            fi
        done
        
        echo ""
        
        if [ $RUNNING -eq 1 ]; then
            sleep 30  # Check every 30 seconds
        fi
    done
    
    echo ""
    echo "[COMPLETE] All processes finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

# Function to run a phase (general or biomedical models)
run_phase() {
    local PHASE_NAME="$1"  # "general" or "biomedical"
    local MODELS_ARRAY_NAME="$2"  # Name of the array variable
    local PHASE_DISPLAY=$(echo "$PHASE_NAME" | tr '[:lower:]' '[:upper:]')
    
    # Get the actual array from the variable name
    local MODELS_ARRAY
    if [ "$MODELS_ARRAY_NAME" = "GENERAL_MODELS" ]; then
        MODELS_ARRAY=("${GENERAL_MODELS[@]}")
    elif [ "$MODELS_ARRAY_NAME" = "BIOMEDICAL_MODELS" ]; then
        MODELS_ARRAY=("${BIOMEDICAL_MODELS[@]}")
    else
        echo "Error: Unknown model array name: $MODELS_ARRAY_NAME"
        exit 1
    fi
    
    echo ""
    echo "======================================================================"
    echo "🚀 ${PHASE_DISPLAY} PHASE: One model at a time, all systems parallel"
    echo "======================================================================"
    echo "Execution Pattern:"
    echo "  Process one model at a time"
    echo "  For each model, run all systems simultaneously on different GPUs"
    echo "  Total models to process: ${#MODELS_ARRAY[@]}"
    echo "======================================================================"
    
    # Process each model sequentially
    for MODEL_IDX in "${!MODELS_ARRAY[@]}"; do
        MODEL="${MODELS_ARRAY[$MODEL_IDX]}"
        MODEL_DISPLAY=$(get_model_display_name "$MODEL")
        MODEL_NUM=$((MODEL_IDX + 1))
        TOTAL_MODELS_IN_PHASE="${#MODELS_ARRAY[@]}"
        
        echo ""
        echo "🚀 Processing Model ${MODEL_NUM}/${TOTAL_MODELS_IN_PHASE}: $MODEL_DISPLAY"
        echo "   Running all ${#SYSTEMS[@]} systems simultaneously on different GPUs..."
        
        local ALL_PIDS=()
        local ALL_NAMES=()
        
        # Launch all systems simultaneously for this model
        for SYSTEM_IDX in "${!SYSTEMS[@]}"; do
            SYSTEM="${SYSTEMS[$SYSTEM_IDX]}"
            GPU_ID="${GPUS[$SYSTEM_IDX]:-0}"
            
            echo "   🚀 $SYSTEM → GPU $GPU_ID → $MODEL_DISPLAY"
            run_system_model "$SYSTEM" "$GPU_ID" "$MODEL" "$PHASE_NAME" "$MODEL_NUM" "$TOTAL_MODELS_IN_PHASE"
            ALL_PIDS+=("$!")
            ALL_NAMES+=("${SYSTEM}_GPU${GPU_ID}")
            
            # Small stagger to avoid overwhelming
            sleep 1
        done
        
        echo ""
        echo "⏳ All ${#ALL_PIDS[@]} systems launched for $MODEL_DISPLAY. Waiting for completion..."
        
        # Wait for all systems to complete for this model
        wait_for_processes "${ALL_PIDS[@]}" "${ALL_NAMES[@]}"
        
        echo ""
        echo "✅ Completed $MODEL_DISPLAY: All ${#SYSTEMS[@]} systems finished"
        echo "======================================================================"
        
        # Pause between models
        sleep 1
    done
    
    echo ""
    echo "✅ ${PHASE_DISPLAY} PHASE COMPLETE: All models processed"
    echo "======================================================================"
}

# Main execution
echo "======================================================================"
echo "Knowledge Triple Probing - Multi-GPU Per-Model Layer Sweep"
echo "======================================================================"
echo "Execution Strategy:"
echo "  1. Process one model at a time"
echo "  2. For each model, run all systems simultaneously on different GPUs"
echo "  3. Wait for all systems to complete before moving to next model"
echo ""
echo "Parallel Execution Pattern (per model):"
echo "  GPU ${GPUS[0]} → ${SYSTEMS[0]} system → all layers for current model"
echo "  GPU ${GPUS[1]} → ${SYSTEMS[1]} system → all layers for current model"
echo "  GPU ${GPUS[2]} → ${SYSTEMS[2]} system → all layers for current model"
echo "  (All ${#SYSTEMS[@]} systems run simultaneously for the same model)"
echo ""
echo "Configuration:"
echo "  Systems: ${SYSTEMS[*]}"
echo "  General Models: ${#GENERAL_MODELS[@]} models"
echo "  Biomedical Models: ${#BIOMEDICAL_MODELS[@]} models"
echo "  GPUs: ${GPUS[*]}"
echo "  Subject layers: ${SUBJECT_LAYER_START}-${SUBJECT_LAYER_END}"
echo "  Object offset: +${LAYER_OFFSET}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  CSV results: ${RESULTS_DIR}"
echo "======================================================================"

# Skip saving extra config files; keep script output minimal

# PHASE 1: Run GENERAL_MODELS with multi-GPU per-model execution
run_phase "general" "GENERAL_MODELS"

# PHASE 2: Run BIOMEDICAL_MODELS with multi-GPU per-model execution
run_phase "biomedical" "BIOMEDICAL_MODELS"

# Calculate total experiments
TOTAL_EXPERIMENTS=$((${#SYSTEMS[@]} * ${#ALL_MODELS[@]} * 2 * (SUBJECT_LAYER_END - SUBJECT_LAYER_START + 1)))

# Final summary
echo ""
echo "======================================================================"
echo "[COMPLETE] All experiments finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"
echo "Results Summary:"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Result files: ${RESULTS_DIR}"
echo "  CSV results: ${RESULTS_DIR}"
echo "  Log files: None (simplified output)"
echo "  Configuration: Direct execution"
echo "  Total experiments: ${TOTAL_EXPERIMENTS}"
echo ""
echo "To analyze results:"
echo "  python analyze_layer_sweep_results.py --input-dir ${RESULTS_DIR}"
echo "======================================================================"

# Simple end message
echo "======================================================================"
echo "🎉 Layer sweep completed. CSVs saved in: ${OUTPUT_DIR}"
echo "======================================================================"
