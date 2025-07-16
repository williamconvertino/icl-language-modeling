cd ..

# 0.001

sbatch --job-name=transformer_tinystories_v2 ./train_small.sh dataset=tinystories_v2 model=transformer_small training.batch_size=32 training.epochs=15 training.optimizer.lr=0.001
sbatch --job-name=icl_tinystories_v2 ./train_small.sh dataset=tinystories_v2 model=icl_small training.batch_size=32 training.epochs=15 training.optimizer.lr=0.001
sbatch --job-name=icl_mlp_tinystories_v2 ./train_small.sh dataset=tinystories_v2 model=icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=15 training.optimizer.lr=0.001
sbatch --job-name=icl_full_tinystories_v2 ./train_small.sh dataset=tinystories_v2 model=icl_small model.icl_use_mlp=true training.batch_size=32 training.epochs=15 training.optimizer.lr=0.001

sbatch --job-name=icl_wv_full_tinystories_v2 ./train_small.sh dataset=tinystories_v2 model=icl_small model.icl_use_wv=true model.icl_use_mlp=true training.batch_size=32 training.epochs=15 training.optimizer.lr=0.001

# 0.0003

sbatch --job-name=transformer_tinystories_v2 ./train_small.sh dataset=tinystories_v2 model=transformer_small training.batch_size=32 training.epochs=15 training.optimizer.lr=0.0003
sbatch --job-name=icl_tinystories_v2 ./train_small.sh dataset=tinystories_v2 model=icl_small training.batch_size=32 training.epochs=15 training.optimizer.lr=0.0003
sbatch --job-name=icl_mlp_tinystories_v2 ./train_small.sh dataset=tinystories_v2 model=icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=15 training.optimizer.lr=0.0003
sbatch --job-name=icl_full_tinystories_v2 ./train_small.sh dataset=tinystories_v2 model=icl_small model.icl_use_mlp=true training.batch_size=32 training.epochs=15 training.optimizer.lr=0.0003

sbatch --job-name=icl_wv_full_tinystories_v2 ./train_small.sh dataset=tinystories_v2 model=icl_small model.icl_use_wv=true model.icl_use_mlp=true training.batch_size=32 training.epochs=15 training.optimizer.lr=0.0003

# 0.0001

sbatch --job-name=transformer_tinystories_v2 ./train_small.sh dataset=tinystories_v2 model=transformer_small training.batch_size=32 training.epochs=15 training.optimizer.lr=0.0001
sbatch --job-name=icl_tinystories_v2 ./train_small.sh dataset=tinystories_v2 model=icl_small training.batch_size=32 training.epochs=15 training.optimizer.lr=0.0001
sbatch --job-name=icl_mlp_tinystories_v2 ./train_small.sh dataset=tinystories_v2 model=icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=15 training.optimizer.lr=0.0001
sbatch --job-name=icl_full_tinystories_v2 ./train_small.sh dataset=tinystories_v2 model=icl_small model.icl_use_mlp=true training.batch_size=32 training.epochs=15 training.optimizer.lr=0.0001

sbatch --job-name=icl_wv_full_tinystories_v2 ./train_small.sh dataset=tinystories_v2 model=icl_small model.icl_use_wv=true model.icl_use_mlp=true training.batch_size=32 training.epochs=15 training.optimizer.lr=0.0001

