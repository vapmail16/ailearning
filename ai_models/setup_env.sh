#!/bin/bash

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Verify activation
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "Virtual environment activated successfully!"
    
    # Install requirements
    echo "Installing requirements..."
    pip install -r requirements.txt
    
    echo "Setup complete! Virtual environment is active."
    echo "To deactivate later, type 'deactivate'"
else
    echo "Failed to activate virtual environment"
    exit 1
fi 