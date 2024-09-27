#!/bin/bash

# for fname in "gemini-pro_music_10fps_1geminifps.yaml" "gemini-pro_fitness_8fps_1geminifps.yaml" "gemini-pro_ballsports_10fps_1geminifps.yaml" "gemini-pro_surgery_10fps_1geminifps.yaml" ; do
#for fname in "eval2_easy.yaml" "eval2_fitness.yaml"  "eval2_ballsports.yaml" "eval2_diving.yaml" "eval2_music.yaml" "eval2_surgery.yam" ; do
for fname in  "eval2_diving.yaml" "eval2_music.yaml" "eval2_surgery.yaml" ; do
    # Remove .yaml extension for the log file name
    log_base=${fname%.yaml}
    
    # Create timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Create log file name
    log_file="logs/${timestamp}_${log_base}.log"
    
    echo $log_file

    # Run the Python script and log output
    python_command="python -u viddiff_method/run_viddiff.py -c viddiff_method/configs/${fname}"
    echo $Running cmd:     $python_command
    eval "$python_command" 2>&1 | tee "${log_file}" 
    echo
    echo
    
done




 
