# Transformer

sbatch --job-name=transformer_8l ./run_job.sh model.n_layers=8
sbatch --job-name=transformer_6l ./run_job.sh model.n_layers=6
sbatch --job-name=transformer_4l ./run_job.sh model.n_layers=4

sbatch --job-name=transformer_8l_goodwiki ./run_job.sh model.n_layers=8 dataset=goodwiki
sbatch --job-name=transformer_6l_goodwiki ./run_job.sh model.n_layers=6 dataset=goodwiki
sbatch --job-name=transformer_4l_goodwiki ./run_job.sh model.n_layers=4 dataset=goodwiki

# ICL

sbatch --job-name=icl_8l ./run_job.sh model=icl model.n_layers=8
sbatch --job-name=icl_6l ./run_job.sh model=icl model.n_layers=6
sbatch --job-name=icl_4l ./run_job.sh model=icl model.n_layers=4

sbatch --job-name=icl_8l_goodwiki ./run_job.sh model=icl model.n_layers=8 dataset=goodwiki
sbatch --job-name=icl_6l_goodwiki ./run_job.sh model=icl model.n_layers=6 dataset=goodwiki
sbatch --job-name=icl_4l_goodwiki ./run_job.sh model=icl model.n_layers=4 dataset=goodwiki

# Other Modifications

sbatch --job-name=icl_use_wv ./run_job.sh model=icl model.icl_use_wv=true
sbatch --job-name=icl_use_ln_mlp ./run_job.sh model=icl model.icl_use_ln_mlp=true
sbatch --job-name=icl_use_skip_mlp ./run_job.sh model=icl model.icl_use_skip_mlp=true
sbatch --job-name=icl_use_ln_v ./run_job.sh model=icl model.icl_use_ln_v=true
sbatch --job-name=icl_use_ln_qk ./run_job.sh model=icl model.icl_use_ln_qk=true

# Shared weights
# sbatch --job-name=share_icl_weights ./run_job.sh model=icl model.n_layers=8 model.share_icl_weights=true
# sbatch --job-name=share_covariate_weights ./run_job.sh model=icl model.n_layers=8 model.share_covariate_weights=true

# Alternate Order

sbatch --job-name=alt_1 ./run_job.sh model=icl model.block_order=["T", "T", "I", "T", "T", "I", "T", "I"]