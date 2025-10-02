#! /usr/bin/bash
SHORT=$(git rev-parse --short HEAD)
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
PROJECT_NAME=$(basename "$PROJECT_ROOT")
APPTAINER_FILE=Apptainer

# Build the base run image if the definition file has changed
if [[ "$SCRIPT_DIR/ApptainerBaseRun" -nt "$SCRIPT_DIR/base_run.img" || "$PROJECT_ROOT/requirements.txt" -nt "$SCRIPT_DIR/base_run.img" ]]; then
  sudo singularity build "$SCRIPT_DIR/base_run.img" "$SCRIPT_DIR/ApptainerBaseRun"
fi

mkdir -p "$PROJECT_ROOT/build_output"

# Build the apptainer image if the definition file has changed
sudo singularity build \
 "$PROJECT_NAME-$SHORT.img" "$SCRIPT_DIR/$APPTAINER_FILE"

echo "$PROJECT_NAME-$SHORT.img"