#!/bin/bash
# * USAGE: ./forward.sh [start|stop|status] <port>

REMOTE_USER_HOST="bdmi66"
ACTION=${1:-start}
LOCAL_PORT=${2:-11800}
REMOTE_PORT="${3:-$LOCAL_PORT}"

MATCH="autossh.*-R 0.0.0.0:${REMOTE_PORT}:localhost:${LOCAL_PORT}.*${REMOTE_USER_HOST}"

get_pids() {
    pgrep -f "$MATCH"
}

stop_forward() {
    PIDS=($(get_pids))

    if [ ${#PIDS[@]} -eq 0 ]; then
        echo "No forwarding process found for port $LOCAL_PORT."
        return
    fi

    echo "Stopping forwarding on port $LOCAL_PORT (PID: ${PIDS[*]})..."
    kill "${PIDS[@]}"
}

start_forward() {
    stop_forward
    sleep 1

    echo "Starting port forwarding: local $LOCAL_PORT -> $REMOTE_USER_HOST:$REMOTE_PORT ..."

    autossh -M 0 -f -N \
        -o "ServerAliveInterval 60" \
        -o "ServerAliveCountMax 3" \
        -R 0.0.0.0:${REMOTE_PORT}:localhost:${LOCAL_PORT} \
        ${REMOTE_USER_HOST}

    if [ $? -eq 0 ]; then
        echo "Port $LOCAL_PORT forwarding started successfully."
    else
        echo "Failed to start. Check network or port conflict."
    fi
}

status_forward() {
    PIDS=($(get_pids))

    if [ ${#PIDS[@]} -eq 0 ]; then
        echo "Port $LOCAL_PORT forwarding is NOT running."
    else
        echo "Port $LOCAL_PORT forwarding is running (PID: ${PIDS[*]})."
    fi
}

case "$ACTION" in
    start)
        start_forward
        ;;
    stop)
        stop_forward
        ;;
    status)
        status_forward
        ;;
    *)
        echo "Usage: $0 [start|stop|status] [port]"
        exit 1
        ;;
esac