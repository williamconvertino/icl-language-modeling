cd ..

sbatch --job-name=transformer_goodwiki ./lr_search.sh dataset=goodwiki model=transformer_small training.batch_size=32
sbatch --job-name=icl_goodwiki ./lr_search.sh dataset=goodwiki model=icl_small training.batch_size=32
sbatch --job-name=icl_mlp_goodwiki ./lr_search.sh dataset=goodwiki model=icl_small model.use_output_mlp=true training.batch_size=32
sbatch --job-name=icl_full_goodwiki ./lr_search.sh dataset=goodwiki model=icl_small model.icl_use_mlp=true training.batch_size=32

sbatch --job-name=icl_wv__goodwiki ./lr_search.sh dataset=goodwiki model=icl_small model.icl_use_wv=true training.batch_size=32
sbatch --job-name=icl_wv_mlp_goodwiki ./lr_search.sh dataset=goodwiki model=icl_small model.icl_use_wv=true model.use_output_mlp=true training.batch_size=32
sbatch --job-name=icl_wv_full_goodwiki ./lr_search.sh dataset=goodwiki model=icl_small model.icl_use_wv=true model.icl_use_mlp=true training.batch_size=32
