cd ..

sbatch --job-name=transformer_tinystories_v2 ./lr_search.sh dataset=tinystories_v2 model=transformer_small training.batch_size=32
sbatch --job-name=icl_tinystories_v2 ./lr_search.sh dataset=tinystories_v2 model=icl_small training.batch_size=32
sbatch --job-name=icl_mlp_tinystories_v2 ./lr_search.sh dataset=tinystories_v2 model=icl_small model.use_output_mlp=true training.batch_size=32
sbatch --job-name=icl_full_tinystories_v2 ./lr_search.sh dataset=tinystories_v2 model=icl_small model.icl_use_mlp=true training.batch_size=32

sbatch --job-name=icl_wv__tinystories_v2 ./lr_search.sh dataset=tinystories_v2 model=icl_small model.icl_use_wv=true training.batch_size=32
sbatch --job-name=icl_wv_mlp_tinystories_v2 ./lr_search.sh dataset=tinystories_v2 model=icl_small model.icl_use_wv=true model.use_output_mlp=true training.batch_size=32
sbatch --job-name=icl_wv_full_tinystories_v2 ./lr_search.sh dataset=tinystories_v2 model=icl_small model.icl_use_wv=true model.icl_use_mlp=true training.batch_size=32
