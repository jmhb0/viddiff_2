#!/bin/bash
# for fname in "qwen_ballsports_5fps.yaml" ; do
for fname in "mode2_qwen_ballsports_5fps.yaml" "mode2_qwen_fitness_4fps.yaml" "mode2_qwen_diving_6fps.yaml" "mode2_qwen_music_2fps.yaml"  "mode2_qwen_surgery_2fps.yaml" "qwen_fitness_4fps.yaml" "qwen_ballsports_5fps.yaml" "qwen_diving_6fps.yaml" "qwen_music_2fps.yaml" "qwen_surgery_2fps.yaml" ; do

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
