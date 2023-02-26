# Using the training scripts in Google Colab

1. Load all the scripts (`evaluation.py`, `preprocess_data.py`, `tokenize_data.py`, `train.py`) into Google Colab
1. Load the file containing annotated data
1. Run the following commands:
   1. `!pip install transformers datasets evaluate transformers[sentencepiece] seqeval wandb`
   1. `!wandb login` - to track experiments using Weights & Biases
   1. `!huggingface-cli login`
   1. `!python train.py -n <NAME_OF_REPO> -e <EPOCHS> -m <MODEL>` - trains the model and pushes it to your repo
