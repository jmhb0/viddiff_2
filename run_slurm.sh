# experiment on a list ofexisting yamls for fules tat already exist 
account=pasteur
partition=pasteur
email_fail=1

## get the configs
set -e
cur_fname="$(basename $0 .sh)"
script_name=$(basename $0)
code=${2:-1} # the second argument to the script; defaults to 1 if none passed

cmd="bash scripts/run_gpt4o.sh"
# echo $cmd
# eval $cmd

sbatch <<< \
"#!/bin/bash
#SBATCH --job-name=${cur_fname}-${partition}
#SBATCH --output=slurm_logs/${cur_fname}-${partition}-%j-out.txt
#SBATCH --error=slurm_logs/${cur_fname}-${partition}-%j-err.txt
#SBATCH --exclude=pasteur1
#SBATCH --mem=24gb
#SBATCH -c 2
#SBATCH -p $partition 
#SBATCH -A $account 
#SBATCH --time=48:00:00  
#SBATCH --mail-user=jmhb@stanford.edu
#SBATCH --mail-type=FAIL

echo \"$cmd\"
eval \"$cmd\"
"
