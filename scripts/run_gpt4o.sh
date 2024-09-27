#!/bin/bash

# for fname in "gpt-4o_easy_8fps.yaml" "gpt-4o_ballsports_10fps.yaml" "gpt-4o_fitness_8fps.yaml" "gpt-4o_music_3fps.yaml" "gpt-4o_surgery_3fps.yaml"; do
for fname in "mode2-gpt-4o_easy_4fps.yaml" "mode2-gpt-4o_ballsports_4fps.yaml" "mode2-gpt-4o_fitness_4fps.yaml" "mode2-gpt-4o_diving_4fps.yaml" "mode2-gpt-4o_music_2fps.yaml" "mode2-gpt-4o_surgery_2fps.yaml" ; do 
# for fname in "mode2-gpt-4o_ballsports_4fps.yaml" "mode2-gpt-4o_fitness_4fps.yaml" ; do 
    # Remove .yaml extension for the log file name
    log_base=${fname%.yaml}
    
    timestamp=$(date +%Y%m%d_%H%M%S)

    log_file="logs/${timestamp}_${log_base}.log"
    
    python_command="python -u lmms/run_lmm.py -c lmms/configs/${fname}"
    echo "Running command: $python_command"

    eval "$python_command" 2>&1 | tee "${log_file}"
    
    # sleep 2
    echo
    echo
done
