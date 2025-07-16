set -e
source ~/.bashrc
conda activate icl

cd ..

python main.py mode="visualize" "$@"
# python main.py mode="visualize" "$@" metric="perplexity"
# python main.py mode="visualize" "$@" visualization.visualization_mode="model"
# python main.py mode="visualize" "$@" visualization.visualization_mode="lr"