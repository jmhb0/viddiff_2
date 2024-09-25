#!/bin/bash

# for fname in  "gemini-pro_fitness_8fps_1geminifps.yaml" "gemini-pro_ballsports_10fps_1geminifps.yaml" "gemini-pro_surgery_10fps_1geminifps.yaml" "gemini-pro_music_10fps_1geminifps.yaml"; do 
for fname in  "gemini-pro_easy_8fps_1geminifps.yaml" ; do 
    # Remove .yaml extension for the log file name
    log_base=${fname%.yaml}
    
    # Create timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Create log file name
    log_file="logs/${timestamp}_${log_base}.log"
    
    echo $log_file

    # Run the Python script and log output
    echo "python lmms/run_lmm.py -c lmms/configs/${fname}"
    python lmms/run_lmm.py -c lmms/configs/${fname} 2>&1 | tee "${log_file}"
    
    # Sleep for 60 seconds
    echo 
    sleep 5
done




