# source this to expose the pinned AKOrN checkout (optional — _bootstrap.py also handles it).
export AKORN_HOME="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/external/akorn"
export PYTHONPATH="$AKORN_HOME:${PYTHONPATH:-}"
echo "AKORN_HOME=$AKORN_HOME"
