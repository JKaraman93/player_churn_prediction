#!/bin/sh

set -eu

show_help() {
    cat <<'EOF'
Usage:
  docker run --rm -v "$(pwd):/app" bet-project <command> [args]

Commands:
  help                    Show this message
  bronze                  Generate Bronze layer data
  silver                  Generate Silver layer data
  gold                    Generate Gold layer data
  train                   Train the churn model
  backtest                Run backtesting on the registered model
  inference <YYYY-MM-DD>  Run batch inference for one scoring date
  shell                   Open a shell inside the container

Recommended order:
  bronze -> silver -> gold -> train -> [optional: backtest] -> inference <YYYY-MM-DD>

Important:
  After training, manually assign the desired model version the MLflow alias
  "production" before running inference.
  The backtest and inference commands load:
  SparkLogisticRegression_train@production

Examples:
  docker run --rm -v "$(pwd):/app" bet-project bronze
  docker run --rm -v "$(pwd):/app" bet-project silver
  docker run --rm -v "$(pwd):/app" bet-project gold
  docker run --rm -v "$(pwd):/app" bet-project train
  docker run --rm -v "$(pwd):/app" bet-project backtest
  docker run --rm -v "$(pwd):/app" bet-project inference 2024-06-20
  docker run --rm -it -v "$(pwd):/app" bet-project shell
EOF
}

command="${1:-help}"

case "$command" in
    help|-h|--help)
        show_help
        ;;
    bronze)
        exec python src/bet/pipelines/create_bronze_dataset.py
        ;;
    silver)
        exec python src/bet/pipelines/create_silver_dataset.py
        ;;
    gold)
        exec python src/bet/pipelines/create_gold_dataset.py
        ;;
    train)
        exec python src/bet/models/logistic_regression.py
        ;;
    backtest)
        exec python src/bet/evaluation/backtest.py
        ;;
    inference)
        if [ "$#" -lt 2 ]; then
            echo "Error: inference requires a date argument in YYYY-MM-DD format." >&2
            echo >&2
            show_help >&2
            exit 1
        fi
        exec python src/bet/models/inference.py "$2"
        ;;
    shell)
        exec sh
        ;;
    *)
        echo "Error: unknown command '$command'." >&2
        echo >&2
        show_help >&2
        exit 1
        ;;
esac
