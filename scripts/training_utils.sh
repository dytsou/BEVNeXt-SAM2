#!/bin/bash
# BEVNeXt-SAM2 Training Utilities
# Convenient wrapper for all training management utilities

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_help() {
    echo -e "${BLUE}BEVNeXt-SAM2 Training Utilities${NC}"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Training Management:"
    echo "  status             - Show current training status"
    echo "  start [OPTIONS]    - Start training with enhanced resume support"
    echo "  stop               - Stop currently running training"
    echo "  resume             - Detect and manage resume opportunities"
    echo "  monitor            - Monitor training progress"
    echo ""
    echo "Checkpoint Management:"
    echo "  list-checkpoints   - List all available checkpoints"
    echo "  analyze-checkpoint PATH - Analyze specific checkpoint"
    echo "  cleanup-checkpoints - Clean up old checkpoints"
    echo ""
    echo "Analysis & Logs:"
    echo "  logs [N]           - Show last N lines of training logs"
    echo "  analyze-logs PATH  - Analyze training logs with resume detection"
    echo "  plot-metrics PATH  - Plot training metrics with resume markers"
    echo ""
    echo "Start Training Options:"
    echo "  --auto-resume      - Automatically resume from latest checkpoint"
    echo "  --resume PATH      - Resume from specific checkpoint"
    echo "  --no-resume        - Force fresh training"
    echo "  --data-path PATH   - Path to dataset"
    echo "  --epochs N         - Number of epochs"
    echo "  --gpu              - Use GPU"
    echo "  --checkpoint-freq N - Checkpoint frequency"
    echo ""
    echo "Examples:"
    echo "  $0 status"
    echo "  $0 start --auto-resume --gpu"
    echo "  $0 start --resume outputs/checkpoint_latest.pth --gpu"
    echo "  $0 analyze-checkpoint outputs/training/checkpoint_latest.pth"
    echo "  $0 logs 100"
    echo "  $0 monitor"
}

check_requirements() {
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3 is required but not available${NC}"
        return 1
    fi
    
    # Check if Docker is available for training commands
    if [[ "$1" == "start" ]] || [[ "$1" == "stop" ]] || [[ "$1" == "monitor" ]]; then
        if ! command -v docker &> /dev/null; then
            echo -e "${RED}❌ Docker is required for training commands${NC}"
            return 1
        fi
    fi
    
    return 0
}

run_training_manager() {
    local cmd="$1"
    shift
    
    cd "$PROJECT_ROOT"
    python3 utils/training_manager.py "$cmd" "$@"
}

run_checkpoint_analyzer() {
    local checkpoint_path="$1"
    shift
    
    cd "$PROJECT_ROOT"
    python3 utils/checkpoint_analyzer.py "$checkpoint_path" "$@"
}

run_log_analyzer() {
    local log_path="$1"
    shift
    
    cd "$PROJECT_ROOT"
    python3 tools/analysis_tools/analyze_logs.py plot_curve "$log_path" "$@"
}

# Parse command
COMMAND="$1"
shift || true

if [[ -z "$COMMAND" ]]; then
    print_help
    exit 0
fi

# Check requirements
check_requirements "$COMMAND"

case "$COMMAND" in
    status)
        echo -e "${BLUE}📊 Getting training status...${NC}"
        run_training_manager status
        ;;
        
    start)
        echo -e "${GREEN}🚀 Starting training...${NC}"
        
        # Parse start arguments
        ARGS=()
        while [[ $# -gt 0 ]]; do
            case $1 in
                --auto-resume)
                    ARGS+=(--auto-resume)
                    shift
                    ;;
                --resume)
                    ARGS+=(--resume "$2")
                    shift 2
                    ;;
                --no-resume)
                    ARGS+=(--no-resume)
                    shift
                    ;;
                --data-path)
                    ARGS+=(--data-path "$2")
                    shift 2
                    ;;
                --epochs)
                    ARGS+=(--epochs "$2")
                    shift 2
                    ;;
                --gpu)
                    ARGS+=(--gpu)
                    shift
                    ;;
                --checkpoint-freq)
                    ARGS+=(--checkpoint-freq "$2")
                    shift 2
                    ;;
                *)
                    echo -e "${RED}Unknown start option: $1${NC}"
                    exit 1
                    ;;
            esac
        done
        
        run_training_manager start "${ARGS[@]}"
        ;;
        
    stop)
        echo -e "${YELLOW}⏹️  Stopping training...${NC}"
        run_training_manager stop
        ;;
        
    resume)
        echo -e "${CYAN}🔄 Analyzing resume opportunities...${NC}"
        run_training_manager resume "$@"
        ;;
        
    monitor)
        echo -e "${BLUE}📺 Monitoring training...${NC}"
        "$SCRIPT_DIR/monitor_training.sh"
        ;;
        
    list-checkpoints)
        echo -e "${BLUE}📋 Listing checkpoints...${NC}"
        run_training_manager list "$@"
        ;;
        
    analyze-checkpoint)
        if [[ -z "$1" ]]; then
            echo -e "${RED}❌ Please specify checkpoint path${NC}"
            echo "Usage: $0 analyze-checkpoint PATH"
            exit 1
        fi
        
        echo -e "${BLUE}🔍 Analyzing checkpoint: $1${NC}"
        run_checkpoint_analyzer "$1" "${@:2}"
        ;;
        
    cleanup-checkpoints)
        echo -e "${YELLOW}🧹 Cleaning up old checkpoints...${NC}"
        run_training_manager cleanup "$@"
        ;;
        
    logs)
        LINES=${1:-50}
        echo -e "${BLUE}📄 Showing last $LINES lines of training logs...${NC}"
        run_training_manager logs --lines "$LINES"
        ;;
        
    analyze-logs)
        if [[ -z "$1" ]]; then
            echo -e "${RED}❌ Please specify log file path${NC}"
            echo "Usage: $0 analyze-logs PATH [OPTIONS]"
            exit 1
        fi
        
        echo -e "${BLUE}📈 Analyzing training logs: $1${NC}"
        run_log_analyzer "$1" "${@:2}"
        ;;
        
    plot-metrics)
        if [[ -z "$1" ]]; then
            echo -e "${RED}❌ Please specify log file path${NC}"
            echo "Usage: $0 plot-metrics PATH [--keys METRICS] [--out OUTPUT]"
            exit 1
        fi
        
        echo -e "${BLUE}📊 Plotting training metrics: $1${NC}"
        
        # Default to common metrics if not specified
        PLOT_ARGS=("$1")
        if [[ "$*" != *"--keys"* ]]; then
            PLOT_ARGS+=(--keys loss val_loss learning_rate)
        fi
        PLOT_ARGS+=("${@:2}")
        
        run_log_analyzer "${PLOT_ARGS[@]}"
        ;;
        
    --help|-h|help)
        print_help
        ;;
        
    *)
        echo -e "${RED}❌ Unknown command: $COMMAND${NC}"
        echo ""
        print_help
        exit 1
        ;;
esac
