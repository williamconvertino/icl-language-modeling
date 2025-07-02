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
