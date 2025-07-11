sbatch --job-name=transformer_medium ./train_medium.sh model=transformer_medium
# sbatch --job-name=icl_medium ./train_medium.sh model=icl_medium
# sbatch --job-name=icl_medium_mlp ./train_medium.sh model=icl_medium model.icl_use_mlp=true
# sbatch --job-name=icl_medium_output_mlp ./train_medium.sh model=icl_medium model.use_output_mlp=true