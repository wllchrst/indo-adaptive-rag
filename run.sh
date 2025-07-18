echo "Creating new environment..."
conda create -n experiment_env python=3.12.9 -y

echo "Activating environment..."
# You need to source the conda script to use `conda activate` in a shell script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate experiment_env

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Installing PyTorch with CUDA 12.6..."
pip3 install torch torchvision torchaudio

echo "Setup complete."