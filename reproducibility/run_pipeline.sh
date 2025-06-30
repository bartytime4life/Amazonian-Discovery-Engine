#!/bin/bash
set -euo pipefail
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"; }

if [ "$#" -lt 1 ]; then
  log "ERROR: Mode not specified. Usage: $0 {server|client|standalone} [options]"
  exit 1
fi

MODE=$1
shift

if [ "$MODE" = "server" ]; then
  log "--- Starting ADE 2.0 Server ---"
  python3 /ade/code/federated/server.py "$@"
  exit 0
fi

if [ "$MODE" = "client" ]; then
  log "--- Starting ADE 2.0 Client ---"
  python3 /ade/code/federated/client.py "$@"
  exit 0
fi

if [ "$MODE" = "standalone" ]; then
  TILE_ID=$1; BBOX=$2
  CONFIG="/ade/config/pipeline_settings.yaml"
  OUT="/ade/outputs/${TILE_ID}"; mkdir -p "$OUT"
  FS="/ade/data/predictors/${TILE_ID}_features_stack.tif"
  PAG="$OUT/candidate_pag.gml"
  VAL="$OUT/validated_pag.gml"
  REPORT="$OUT/refutation_report.json"
  log "[2/6] Discovering causal graph"; python3 /ade/code/analysis/discover_causal_graph.py --feature_stack "$FS" --algorithm fci --output_path "$PAG" --config "$CONFIG"
  log "[3/6] Simulating validation"; cp "$PAG" "$VAL"
  log "[4/6] Refutation"; python3 /ade/tests/refutation/run_refutation_suite.py --graph_path "$VAL" --output_report "$REPORT"
  log "--- Pipeline Complete ---"
  exit 0
fi

log "Unknown mode '$MODE'"; exit 1
