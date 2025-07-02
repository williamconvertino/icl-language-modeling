# Misc Modifications

sbatch --job-name=use_mlp_out ./run_job.sh model=icl model.use_mlp_out=true
sbatch --job-name=icl_use_skip_mlp ./run_job.sh model=icl model.icl_use_skip_mlp=true
sbatch --job-name=icl_use_wv ./run_job.sh model=icl model.icl_use_wv=true
sbatch --job-name=icl_use_ln ./run_job.sh model=icl model.icl_use_ln=true
sbatch --job-name=icl_use_rotary ./run_job.sh model=icl model.icl_use_rotary_embedding=true

# Shared weights

sbatch --job-name=share_covariate_attn ./run_job.sh model=icl model.n_layers=8 model.share_covariate_attn=true
sbatch --job-name=share_covariate_mlp ./run_job.sh model=icl model.n_layers=8 model.share_covariate_mlp=true
sbatch --job-name=share_icl_attn ./run_job.sh model=icl model.n_layers=8 model.share_icl_attn=true
sbatch --job-name=share_icl_mlp ./run_job.sh model=icl model.n_layers=8 model.share_icl_mlp=true

# Alternate Order

sbatch --job-name=alt_1 ./run_job.sh model=icl model.block_order=["T", "T", "I", "T", "T", "I", "T", "I"]
sbatch --job-name=alt_2 ./run_job.sh model=icl model.block_order=["T", "T", "I", "I", "T", "T", "I", "I"]
sbatch --job-name=alt_3 ./run_job.sh model=icl model.block_order=["T", "T", "T", "T", "T", "T", "I", "I"]
sbatch --job-name=alt_4 ./run_job.sh model=icl model.block_order=["T", "T", "T", "I", "T", "T", "T", "I"]
sbatch --job-name=alt_5 ./run_job.sh model=icl model.block_order=["T", "I", "I", "T", "I", "I", "T", "I"]