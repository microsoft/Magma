#!/bin/bash
# UI Automation App — Run Commands
#
# Usage:
#   bash run.sh server                          # Start FastAPI backend (port 8000)
#   bash run.sh server-mock                     # Start FastAPI backend with mocked model output
#   bash run.sh client                          # Start React frontend (port 4000)
#   bash run.sh agent                           # Start local agent API (port 8010)
#   bash run.sh all                             # Start all 3 in background
#   bash run.sh all-mock                        # Start all 3 with mocked backend model

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "$1" in
    server)
        echo "Starting server on http://localhost:8000 ..."
        cd server && python -m src.main
        ;;
    server-mock)
        echo "Starting server on http://localhost:8000 with USE_MOCK_MODEL=1 ..."
        cd server && USE_MOCK_MODEL=1 python -m src.main
        ;;
    client)
        echo "Starting client on http://localhost:4000 ..."
        cd client && npx vite --port 4000
        ;;
    agent)
        shift  # remove "agent" from args, pass the rest
        echo "Starting agent API on http://localhost:8010 (args: $@) ..."
        python agent/executor.py --server-url http://localhost:8000 --port 8010 "$@"
        ;;
    all)
        echo "Starting all components..."
        echo "  Server: http://localhost:8000"
        echo "  Client: http://localhost:4000"
        echo "  Agent:  http://localhost:8010"
        echo ""
        cd server && python -m src.main &
        SERVER_PID=$!
        cd client && npx vite --port 4000 &
        CLIENT_PID=$!
        sleep 10  # wait for server to load model
        python agent/executor.py --server-url http://localhost:8000 --port 8010 "${@:2}" &
        AGENT_PID=$!
        echo ""
        echo "PIDs: server=$SERVER_PID client=$CLIENT_PID agent=$AGENT_PID"
        echo "Press Ctrl+C to stop all"
        wait
        ;;
    all-mock)
        echo "Starting all components with USE_MOCK_MODEL=1..."
        echo "  Server: http://localhost:8000"
        echo "  Client: http://localhost:4000"
        echo "  Agent:  http://localhost:8010"
        echo ""
        cd server && USE_MOCK_MODEL=1 python -m src.main &
        SERVER_PID=$!
        cd client && npx vite --port 4000 &
        CLIENT_PID=$!
        sleep 2
        python agent/executor.py --server-url http://localhost:8000 --port 8010 "${@:2}" &
        AGENT_PID=$!
        echo ""
        echo "PIDs: server=$SERVER_PID client=$CLIENT_PID agent=$AGENT_PID"
        echo "Press Ctrl+C to stop all"
        wait
        ;;
    *)
        echo "Usage: bash run.sh {server|server-mock|client|agent|all|all-mock}"
        echo ""
        echo "  server                          Start FastAPI backend (port 8000)"
        echo "  server-mock                     Start backend with mocked model output"
        echo "  client                          Start React frontend (port 4000)"
        echo "  agent                           Start local agent API (port 8010)"
        echo "  all                             Start all 3 components"
        echo "  all-mock                        Start all 3 with mocked backend model"
        ;;
esac
