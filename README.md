# Environment
```
conda create -n subworkflow python=3.10 -y                                                                      
conda activate subworkflow
```

check your hardware, you may need to install CUDA toolkit
```
nvidia-smi
nvcc --version
```
Install pytorch with cuda toolkit
```
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install -c pytorch -c nvidia pytorch torchaudio pytorch-cuda=12.4 -y
conda install -c conda-forge cudatoolkit=12.4 cudnn=8 --force-reinstall
```

## Install Demucs
```
conda install demucs
```

## Download model file

```
git lfs install
# 必须拥有海南鸡饭大佬的中文直出模型
git clone https://huggingface.co/litagin/anime-whisper ./models/anime
```

```
conda install -c conda-forge ffmpeg -y 
pip install whisperx

pip install openai
pip install -q -U google-genai
pip install pydantic
pip install dotenv
pip install pysrt

```