# Misc Modifications

sbatch --job-name=use_mlp_out ./lr_search.sh model=icl model.use_mlp_out=true
sbatch --job-name=icl_use_skip_mlp ./lr_search.sh model=icl model.icl_use_skip_mlp=true
sbatch --job-name=icl_use_wv ./lr_search.sh model=icl model.icl_use_wv=true
sbatch --job-name=icl_use_ln ./lr_search.sh model=icl model.icl_use_ln=true
sbatch --job-name=icl_use_rotary ./lr_search.sh model=icl model.icl_use_rotary_embedding=true

# Shared weights

sbatch --job-name=share_covariate_attn ./lr_search.sh model=icl model.n_layers=8 model.share_covariate_attn=true
sbatch --job-name=share_covariate_mlp ./lr_search.sh model=icl model.n_layers=8 model.share_covariate_mlp=true
sbatch --job-name=share_icl_attn ./lr_search.sh model=icl model.n_layers=8 model.share_icl_attn=true
sbatch --job-name=share_icl_mlp ./lr_search.sh model=icl model.n_layers=8 model.share_icl_mlp=true

# Alternate Order

sbatch --job-name=alt_1 ./lr_search.sh model=icl model.block_order=["T", "T", "I", "T", "T", "I", "T", "I"]
sbatch --job-name=alt_2 ./lr_search.sh model=icl model.block_order=["T", "T", "I", "I", "T", "T", "I", "I"]
sbatch --job-name=alt_3 ./lr_search.sh model=icl model.block_order=["T", "T", "T", "T", "T", "T", "I", "I"]
sbatch --job-name=alt_4 ./lr_search.sh model=icl model.block_order=["T", "T", "T", "I", "T", "T", "T", "I"]
sbatch --job-name=alt_5 ./lr_search.sh model=icl model.block_order=["T", "I", "I", "T", "I", "I", "T", "I"]