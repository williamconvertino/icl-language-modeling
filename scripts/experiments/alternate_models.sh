cd ..

sbatch --job-name=de_icl_goodwiki ./lr_search.sh dataset=goodwiki model=de_icl_small training.batch_size=32
sbatch --job-name=de_icl_mlp_goodwiki ./lr_search.sh dataset=goodwiki model=de_icl_small model.use_output_mlp=true training.batch_size=32
sbatch --job-name=de_icl_full_goodwiki ./lr_search.sh dataset=goodwiki model=de_icl_small model.icl_use_mlp=true training.batch_size=32

sbatch --job-name=red_phi_e_icl_mlp_goodwiki ./lr_search.sh dataset=goodwiki model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 embed_dim_phi=256
sbatch --job-name=red_f_e_icl_mlp_goodwiki ./lr_search.sh dataset=goodwiki model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 embed_dim_f=256

sbatch --job-name=red_phi_hid_icl_mlp_goodwiki ./lr_search.sh dataset=goodwiki model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 hidden_dim_phi=256
sbatch --job-name=red_f_hid_icl_mlp_goodwiki ./lr_search.sh dataset=goodwiki model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 hidden_dim_f=256

sbatch --job-name=red_phi_mlp_icl_mlp_goodwiki ./lr_search.sh dataset=goodwiki model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 mlp_dim_phi=256
sbatch --job-name=red_f_mlp_icl_mlp_goodwiki ./lr_search.sh dataset=goodwiki model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 mlp_dim_f=256
