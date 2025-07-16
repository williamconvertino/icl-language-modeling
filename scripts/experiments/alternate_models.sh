cd ..

# 0.001

sbatch --job-name=red_phi_e_icl_mlp_tinystories_v2_lr1e-3 ./train_small.sh dataset=tinystories_v2 model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=10 training.optimizer.lr=0.001 model.embed_dim_phi=256 model.name="red_embed_phi"
sbatch --job-name=red_f_e_icl_mlp_tinystories_v2_lr1e-3 ./train_small.sh dataset=tinystories_v2 model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=10 training.optimizer.lr=0.001 model.embed_dim_f=256 model.name="red_embed_f"

sbatch --job-name=red_phi_hid_icl_mlp_tinystories_v2_lr1e-3 ./train_small.sh dataset=tinystories_v2 model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=10 training.optimizer.lr=0.001 model.hidden_dim_phi=256 model.name="red_hidden_phi"
sbatch --job-name=red_f_hid_icl_mlp_tinystories_v2_lr1e-3 ./train_small.sh dataset=tinystories_v2 model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=10 training.optimizer.lr=0.001 model.hidden_dim_f=256 model.name="red_hidden_f"

sbatch --job-name=red_phi_mlp_icl_mlp_tinystories_v2_lr1e-3 ./train_small.sh dataset=tinystories_v2 model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=10 training.optimizer.lr=0.001 model.mlp_dim_phi=256 model.name="red_mlp_phi"
sbatch --job-name=red_f_mlp_icl_mlp_tinystories_v2_lr1e-3 ./train_small.sh dataset=tinystories_v2 model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=10 training.optimizer.lr=0.001 model.mlp_dim_f=256 model.name="red_mlp_f"

# 0.0003

sbatch --job-name=red_phi_e_icl_mlp_tinystories_v2_lr3e-4 ./train_small.sh dataset=tinystories_v2 model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=10 training.optimizer.lr=0.0003 model.embed_dim_phi=256 model.name="red_embed_phi"
sbatch --job-name=red_f_e_icl_mlp_tinystories_v2_lr3e-4 ./train_small.sh dataset=tinystories_v2 model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=10 training.optimizer.lr=0.0003 model.embed_dim_f=256 model.name="red_embed_f"

sbatch --job-name=red_phi_hid_icl_mlp_tinystories_v2_lr3e-4 ./train_small.sh dataset=tinystories_v2 model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=10 training.optimizer.lr=0.0003 model.hidden_dim_phi=256 model.name="red_hidden_phi"
sbatch --job-name=red_f_hid_icl_mlp_tinystories_v2_lr3e-4 ./train_small.sh dataset=tinystories_v2 model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=10 training.optimizer.lr=0.0003 model.hidden_dim_f=256 model.name="red_hidden_f"

sbatch --job-name=red_phi_mlp_icl_mlp_tinystories_v2_lr3e-4 ./train_small.sh dataset=tinystories_v2 model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=10 training.optimizer.lr=0.0003 model.mlp_dim_phi=256 model.name="red_mlp_phi"
sbatch --job-name=red_f_mlp_icl_mlp_tinystories_v2_lr3e-4 ./train_small.sh dataset=tinystories_v2 model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=10 training.optimizer.lr=0.0003 model.mlp_dim_f=256 model.name="red_mlp_f"

# 0.0001

sbatch --job-name=red_phi_e_icl_mlp_tinystories_v2_lr1e-4 ./train_small.sh dataset=tinystories_v2 model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=10 training.optimizer.lr=0.0001 model.embed_dim_phi=256 model.name="red_embed_phi"
sbatch --job-name=red_f_e_icl_mlp_tinystories_v2_lr1e-4 ./train_small.sh dataset=tinystories_v2 model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=10 training.optimizer.lr=0.0001 model.embed_dim_f=256 model.name="red_embed_f"

sbatch --job-name=red_phi_hid_icl_mlp_tinystories_v2_lr1e-4 ./train_small.sh dataset=tinystories_v2 model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=10 training.optimizer.lr=0.0001 model.hidden_dim_phi=256 model.name="red_hidden_phi"
sbatch --job-name=red_f_hid_icl_mlp_tinystories_v2_lr1e-4 ./train_small.sh dataset=tinystories_v2 model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=10 training.optimizer.lr=0.0001 model.hidden_dim_f=256 model.name="red_hidden_f"

sbatch --job-name=red_phi_mlp_icl_mlp_tinystories_v2_lr1e-4 ./train_small.sh dataset=tinystories_v2 model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=10 training.optimizer.lr=0.0001 model.mlp_dim_phi=256 model.name="red_mlp_phi"
sbatch --job-name=red_f_mlp_icl_mlp_tinystories_v2_lr1e-4 ./train_small.sh dataset=tinystories_v2 model=reduced_icl_small model.use_output_mlp=true training.batch_size=32 training.epochs=10 training.optimizer.lr=0.0001 model.mlp_dim_f=256 model.name="red_mlp_f"
