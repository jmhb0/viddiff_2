#!/bin/bash

for fname in "mode2-gemini-pro_easy_4fps_1geminifps.yaml" "mode2-gemini-pro_fitness_4fps_1geminifps.yaml" "mode2-gemini-pro_ballsports_5fps_1geminifps.yaml" "mode2-gemini-pro_diving_6fps_1geminifps.yaml" "mode2-gemini-pro_music_2fps_1geminifps.yaml" "mode2-gemini-pro_surgery_2fps_1geminifps.yaml"; do
    # Remove .yaml extension for the log file name
    log_base=${fname%.yaml}
    
    # Create timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Create log file name
    log_file="logs/${timestamp}_${log_base}.log"
    
    echo $log_file

    # Run the Python script and log output
    python_command="python lmms/run_lmm_tmp.py -c lmms/configs/${fname}"
    # python_command="python lmms/run_lmm.py -c lmms/configs/${fname}"
    eval "$python_command" 2>&1 | tee "${log_file}"
    
    echo 
    sleep 1
done
