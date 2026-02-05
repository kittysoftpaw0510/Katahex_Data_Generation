#!/bin/bash
# KataHex Rollout Generation Script (Bash version)
# Simple wrapper for generating game rollouts

set -e

# Default values
NUM_GAMES=100
MAX_VISITS=500
NUM_THREADS=4
OUTPUT_DIR="rollouts_output"
KATAHEX_BIN="./build/katahex"
CONFIG_FILE="rollout_config.cfg"
MODEL_FILE="katahex_model_20220618.bin.gz"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Generate game rollouts using KataHex match engine.

Options:
    -n, --num-games NUM      Number of games to generate (default: 100)
    -v, --max-visits NUM     MCTS visits per move (default: 500)
    -t, --threads NUM        Number of parallel game threads (default: 4)
    -o, --output-dir DIR     Output directory (default: rollouts_output)
    -c, --config FILE        Config file to use (default: rollout_config.cfg)
    -m, --model FILE         Model file to use (default: katahex_model_20220618.bin.gz)
    -h, --help              Show this help message

Examples:
    # Generate 100 games with default settings
    $0

    # Generate 50 games with high quality (more visits)
    $0 -n 50 -v 1000

    # Use custom config
    $0 -c my_config.cfg -n 200

    # Use different model
    $0 -m path/to/model.bin.gz -n 100
EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num-games)
            NUM_GAMES="$2"
            shift 2
            ;;
        -v|--max-visits)
            MAX_VISITS="$2"
            shift 2
            ;;
        -t|--threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            ;;
    esac
done

# Check if katahex binary exists
if [ ! -f "$KATAHEX_BIN" ]; then
    echo -e "${RED}Error: KataHex binary not found at $KATAHEX_BIN${NC}"
    echo "Please build KataHex first by running:"
    echo "  cd build && make -j4"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    echo "Please create a config file or use the default rollout_config.cfg"
    exit 1
fi

# Check if model file exists
if [ ! -f "$MODEL_FILE" ]; then
    echo -e "${RED}Error: Model file not found: $MODEL_FILE${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SGF_DIR="$OUTPUT_DIR/sgfs_$TIMESTAMP"
LOG_FILE="$OUTPUT_DIR/match_$TIMESTAMP.log"

mkdir -p "$SGF_DIR"

# Print summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}KataHex Rollout Generation${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Config file:    $CONFIG_FILE"
echo "Model file:     $MODEL_FILE"
echo "Games:          $NUM_GAMES"
echo "Max visits:     $MAX_VISITS"
echo "Parallel games: $NUM_THREADS"
echo "Output dir:     $OUTPUT_DIR"
echo "SGF dir:        $SGF_DIR"
echo "Log file:       $LOG_FILE"
echo -e "${GREEN}========================================${NC}"
echo ""

# Run the match
echo -e "${YELLOW}Starting rollout generation...${NC}"
echo ""

"$KATAHEX_BIN" match \
    -config "$CONFIG_FILE" \
    -log-file "$LOG_FILE" \
    -sgf-output-dir "$SGF_DIR" \
    -override-config "numGamesTotal=$NUM_GAMES, maxVisits=$MAX_VISITS, numGameThreads=$NUM_THREADS"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Rollout generation completed!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo "SGF files: $SGF_DIR"
    echo "Log file:  $LOG_FILE"
    
    # Count SGF files
    SGF_COUNT=$(find "$SGF_DIR" -name "*.sgfs" | wc -l)
    echo "Generated: $SGF_COUNT SGF file(s)"
else
    echo -e "${RED}Error: Rollout generation failed${NC}"
    echo "Check log file: $LOG_FILE"
    exit 1
fi

