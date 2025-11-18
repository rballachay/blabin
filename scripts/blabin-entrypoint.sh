#!/bin/bash

# Set the timezone to local time
echo "Setting timezone to local time..."
ln -sf /usr/share/zoneinfo/$(cat /etc/timezone) /etc/localtime

# Start the Prometheus metrics server
echo "Starting Prometheus metrics server..."
python3 -m src.metrics.server &

# Start the systemd service for the main application
echo "Starting blabin-cycle service..."
systemctl start blabin-cycle.service

# Keep the script running to maintain the container's active state
tail -f /dev/null
