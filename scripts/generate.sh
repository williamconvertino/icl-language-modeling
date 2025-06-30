./run_job.sh model=icl mode=generate
./run_job.sh model=icl model.name=icl_lr0001 training.optimizer.lr=0.0001 mode=generate
./run_job.sh model=icl model.icl_use_wv=true mode=generate
./run_job.sh model=icl model.icl_use_ln_mlp=true model.icl_use_ln_v=true model.icl_use_ln_qk=true mode=generate