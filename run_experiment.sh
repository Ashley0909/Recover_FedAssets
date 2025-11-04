#!/bin/bash

mkdir -p logs
lambda_values=(-4 -5 -13)

for lambda in "${lambda_values[@]}"; do
    echo "Running with EVIL_LAMBDA=$lambda, MALI_LAMBDA=$lambda"

    # Update constant.py
    sed -i "s/^EVIL_LAMBDA = .*/EVIL_LAMBDA = $lambda/" constant.py
    sed -i "s/^MALI_LAMBDA = .*/MALI_LAMBDA = $lambda/" constant.py

    # Run experiment
    nohup python3 main.py > logs/lambda${lambda}.log 2>&1 &
    sleep 2
done


