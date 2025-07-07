sbatch --job-name=transformer_8l ./run_experiment.sh model.n_layers=8
sbatch --job-name=transformer_6l ./run_experiment.sh model.n_layers=6
sbatch --job-name=transformer_4l ./run_experiment.sh model.n_layers=4
# sbatch --job-name=icl ./run_experiment.sh model=icl
# sbatch --job-name=icl ./run_experiment.sh model=icl model.use_mlp_out=true
# sbatch --job-name=icl ./run_experiment.sh model=icl model.icl_mlp_out=true