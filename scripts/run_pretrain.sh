#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

case "${1:-help}" in
    test)
        shift
        bash "$PROJECT_DIR/scripts/run.sh" pretrain-test "$@"
        ;;
    start)
        shift
        bash "$PROJECT_DIR/scripts/run.sh" pretrain "$@"
        ;;
    tensorboard)
        shift
        bash "$PROJECT_DIR/scripts/run.sh" tensorboard "$@"
        ;;
    help|*)
        echo "Usage: $0 {test|start|tensorboard}"
        ;;
esac
