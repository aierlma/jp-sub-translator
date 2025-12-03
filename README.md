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
# 必须拥有海南鸡饭大佬的中文直出模型, You must have chickenrice0721 newest model(https://huggingface.co/chickenrice0721/whisper-large-v2-translate-zh-v0.2-st-ct2)
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
pip install ffprobe

```

paramters below work best for anime whisper model
```
generate_kwargs = {
    "language": "Japanese",
    "no_repeat_ngram_size": 0,
    "repetition_penalty": 1.0,
    "temperature": 0.0,
    
    # --- 基础配置 ---
    "num_beams": 5,  # 关键：使用束搜索，大幅提高稳定性，减少幻听
    "no_repeat_ngram_size": 4,  # 禁止任何长度为4的词组重复，有效打断 "あ、あ、あ、あ" 这样的循环

    # --- 解决“漏听”（让模型更敏感）---
    "logprob_threshold": -1.5,  # 适度降低置信度门槛，比-2.0更保守，先捕捉大部分弱语音
    "no_speech_threshold": 0.5, # 适度降低“无语音”判断门槛

    # --- 解决“幻听”（抑制模型的“胡言乱语”）---
    "compression_ratio_threshold": 2.0, # 关键：用压缩比来过滤掉重复性的幻听内容
}

pipe = pipeline(
    "automatic-speech-recognition",
    model="./models/anime",
    device="cuda",
    torch_dtype=torch.float16,    # 半精度
    chunk_length_s=30.0,
    stride_length_s=(5.0, 5.0),      # 两端各重叠 5s
    batch_size=64,                # 建议先从 16 开始调
)
```

# Usage
The simplest way is to drag videos to the .bat file, the one without suffix using gpt api, and the other one using gemini api

Don't forget to add your own "style_guide"(no .txt extension) and "config.ini" and .env(put your gpt and gemini api there)

```config.ini
[Paths]
# 这里填写你的 infer.exe 工具所在的目录
# 注意：末尾不要加斜杠
whisper_tool_dir = <dir-to-your-海南鸡饭model> # like ./whisper-large-v2-translate-zh-v0.1-lt-ct2-v0.7
```

And also remember, change the generation_config.json5 in 海南鸡饭 dir to the one in this repo
