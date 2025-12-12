#!/bin/bash

# Setup script for CATHODE reproduction project
# This script sets up the necessary environment variables

# Add the working directory to PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Set the LAW_HOME directory for law cache and configuration
export LAW_HOME="${PWD}/.law"
export LAW_CONFIG_FILE="${PWD}/law.cfg"

# Create necessary directories
mkdir -p results/plots results/models data

echo "Environment setup complete!"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "LAW_HOME: ${LAW_HOME}"
echo "LAW_CONFIG_FILE: ${LAW_CONFIG_FILE}"
