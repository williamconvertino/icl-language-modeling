squeue -u $USER | awk '$0 ~ /trans|icl|red/ && $0 !~ /h200/ {print $1}' | xargs -r scancel
squeue -u $USER | awk '$0 ~ /h200/ {print $1}' | xargs -r scancel