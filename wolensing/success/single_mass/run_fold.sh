#!/bin/bash

# Define the range for ym values
for ym in $(seq 0.4 0.025 1.7); do
    # Run the Python script, passing ym as an environment variable
    YM_VALUE=$ym python3 ./cusp.py

    # Save the output, using the ym value in the filename
    # Assuming the script writes output to a file (or adjust to save specific output)
done

