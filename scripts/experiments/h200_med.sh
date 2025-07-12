cd ..

lr=0.0001

sbatch --job-name=h200_transformer_medium_fast ./train_h200.sh model=transformer_medium_fast model.name=h200_transformer_medium_fast training.batch_size=40 training.optimizer.lr=$lr

sbatch --job-name=h200_icl_medium_fast ./train_h200.sh model=icl_medium_fast model.name=h200_icl_medium_fast training.batch_size=40 training.optimizer.lr=$lr
sbatch --job-name=h200_icl_medium_fast_mlp ./train_h200.sh model=icl_medium_fast model.name=h200_icl_medium_fast training.batch_size=40 training.optimizer.lr=$lr model.use_output_mlp=true 
sbatch --job-name=h200_icl_medium_fast_multi_mlp ./train_h200.sh model=icl_medium_fast model.name=h200_icl_medium_fast training.batch_size=40 training.optimizer.lr=$lr model.icl_use_mlp=true

lr=0.00005

sbatch --job-name=h200_transformer_medium_fast ./train_h200.sh model=transformer_medium_fast model.name=h200_transformer_medium_fast training.batch_size=40 training.optimizer.lr=$lr

sbatch --job-name=h200_icl_medium_fast ./train_h200.sh model=icl_medium_fast model.name=h200_icl_medium_fast training.batch_size=40 training.optimizer.lr=$lr
sbatch --job-name=h200_icl_medium_fast_mlp ./train_h200.sh model=icl_medium_fast model.name=h200_icl_medium_fast training.batch_size=40 training.optimizer.lr=$lr model.use_output_mlp=true 
sbatch --job-name=h200_icl_medium_fast_multi_mlp ./train_h200.sh model=icl_medium_fast model.name=h200_icl_medium_fast training.batch_size=40 training.optimizer.lr=$lr model.icl_use_mlp=true

lr=0.0003

sbatch --job-name=h200_transformer_medium_fast ./train_h200.sh model=transformer_medium_fast model.name=h200_transformer_medium_fast training.batch_size=40 training.optimizer.lr=$lr

sbatch --job-name=h200_icl_medium_fast ./train_h200.sh model=icl_medium_fast model.name=h200_icl_medium_fast training.batch_size=40 training.optimizer.lr=$lr
sbatch --job-name=h200_icl_medium_fast_mlp ./train_h200.sh model=icl_medium_fast model.name=h200_icl_medium_fast training.batch_size=40 training.optimizer.lr=$lr model.use_output_mlp=true 
sbatch --job-name=h200_icl_medium_fast_multi_mlp ./train_h200.sh model=icl_medium_fast model.name=h200_icl_medium_fast training.batch_size=40 training.optimizer.lr=$lr model.icl_use_mlp=true
