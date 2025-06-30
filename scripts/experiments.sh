# TinyStories

sbatch --job-name=transformer_8l ./run_job.sh model.n_layers=8
sbatch --job-name=transformer_6l ./run_job.sh model.n_layers=6
sbatch --job-name=transformer_4l ./run_job.sh model.n_layers=4

sbatch --job-name=icl ./run_job.sh model=icl
sbatch --job-name=icl_6l ./run_job.sh model=icl model.n_layers=6
sbatch --job-name=icl_4l ./run_job.sh model=icl model.n_layers=4

# GoodWiki

sbatch --job-name=transformer_8l_goodwiki ./run_job.sh model.n_layers=8 dataset=goodwiki
sbatch --job-name=transformer_6l_goodwiki ./run_job.sh model.n_layers=6 dataset=goodwiki
sbatch --job-name=transformer_4l_goodwiki ./run_job.sh model.n_layers=4 dataset=goodwiki

sbatch --job-name=icl_8l_goodwiki ./run_job.sh model=icl model.n_layers=8 dataset=goodwiki
sbatch --job-name=icl_6l_goodwiki ./run_job.sh model=icl model.n_layers=6 dataset=goodwiki
sbatch --job-name=icl_4l_goodwiki ./run_job.sh model=icl model.n_layers=4 dataset=goodwiki

# TinyStories V2

sbatch --job-name=transformer_8l_v2 ./run_job.sh model.n_layers=8 dataset=tinystories_v2
sbatch --job-name=transformer_6l_v2 ./run_job.sh model.n_layers=6 dataset=tinystories_v2
sbatch --job-name=transformer_4l_v2 ./run_job.sh model.n_layers=4 dataset=tinystories_v2

sbatch --job-name=icl_8l_v2 ./run_job.sh model=icl model.n_layers=8 dataset=tinystories_v2
sbatch --job-name=icl_6l_v2 ./run_job.sh model=icl model.n_layers=6 dataset=tinystories_v2
sbatch --job-name=icl_4l_v2 ./run_job.sh model=icl model.n_layers=4 dataset=tinystories_v2

# Other Modifications

sbatch --job-name=use_mlp_out ./run_job.sh model=icl model.use_mlp_out=true
sbatch --job-name=icl_use_skip_mlp ./run_job.sh model=icl model.icl_use_skip_mlp=true
sbatch --job-name=icl_use_wv ./run_job.sh model=icl model.icl_use_wv=true
sbatch --job-name=icl_use_ln ./run_job.sh model=icl model.icl_use_ln=true
sbatch --job-name=icl_use_rotary ./run_job.sh model=icl model.icl_use_rotary_embedding=true

# Shared weights

# sbatch --job-name=share_covariate_attn ./run_job.sh model=icl model.n_layers=8 model.share_covariate_attn=true
# sbatch --job-name=share_covariate_mlp ./run_job.sh model=icl model.n_layers=8 model.share_covariate_mlp=true
# sbatch --job-name=share_icl_attn ./run_job.sh model=icl model.n_layers=8 model.share_icl_attn=true
# sbatch --job-name=share_icl_mlp ./run_job.sh model=icl model.n_layers=8 model.share_icl_mlp=true

# Alternate Order

# sbatch --job-name=alt_1 ./run_job.sh model=icl model.block_order=["T", "T", "I", "T", "T", "I", "T", "I"]
# sbatch --job-name=alt_2 ./run_job.sh model=icl model.block_order=["T", "T", "I", "I", "T", "T", "I", "I"]
# sbatch --job-name=alt_3 ./run_job.sh model=icl model.block_order=["T", "T", "T", "T", "T", "T", "I", "I"]
# sbatch --job-name=alt_4 ./run_job.sh model=icl model.block_order=["T", "T", "T", "I", "T", "T", "T", "I"]
# sbatch --job-name=alt_5 ./run_job.sh model=icl model.block_order=["T", "I", "I", "T", "I", "I", "T", "I"]