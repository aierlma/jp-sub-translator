# %% [markdown]
# # First step
# We use demucs to separate vocals
# 
# # Second
# 
# We get a rough timestamp(using kotoba-tech/kotoba-whisper-v2.0-faster, 4m processing for a 2h movie) and an accurate transcript without timestamp(using [amine whisper](https://huggingface.co/litagin/anime-whisper))

# %%
import subprocess
from pathlib import Path
import gc

import pysrt
import torch
from transformers import pipeline
from google import genai
import os
from pydantic import BaseModel
import json

import configparser # 导入用于读取配置文件的库

import time # 引入time模块，用于在API调用之间添加延迟
from dotenv import load_dotenv
load_dotenv()

import sys
video_path = Path(sys.argv[1])

config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8') # 读取配置文件

try:
    client = genai.Client()
except KeyError:
    print("错误：请先设置 GOOGLE_API_KEY 环境变量。") # 在.env 文件中添加 GOOGLE_API_KEY=your_api_key
    exit()

BATCH_SIZE            = 300               # Batch size为-1表示不分批，gemini可以做到不分批
MODEL_FOR_CHUNKING    = "gemini-2.5-pro"   # 用于提取片段的小模型
MODEL_FOR_ALIGNMENT   = "gemini-2.5-pro"        # 用于对齐+翻译的大模型

SPLIT_DURATION_SECONDS = 1500  # 25分钟


# 基于 your_movie.wav 分离人声
demucs_model = "htdemucs" 
# 分离后的人声文件路径
vocals_path = Path(f"separated/{demucs_model}/{video_path.stem}/vocals.wav")

# --- 1. 辅助函数：获取视频时长 ---
def get_video_duration(video_path: str) -> float:
    """使用 ffprobe 获取视频的总时长（秒）。"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout)
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"获取视频时长失败: {e}")
        return 0.0

# --- 2. 更新后的音频分割函数 ---
def extract_and_split_audio(
    input_video: str,
    split_duration: int = 5400,  # 1.5小时 = 5400秒
) -> list[Path]:
    """
    从视频中提取音频流，转成 PCM WAV（pcm_s16le），并按指定时长分割。
    返回所有生成音频块的文件路径列表。
    """
    in_path = Path(input_video)
    audio_output_dir = Path("audio") / in_path.stem
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    # 你的获取视频时长函数
    total_duration = get_video_duration(str(in_path))
    if total_duration == 0:
        print("无法处理，视频时长为0或获取失败。")
        return []

    num_chunks = int(total_duration // split_duration) + 1
    chunk_paths = []

    print(f"视频总时长: {total_duration:.2f}秒，分割为 {num_chunks} 段 PCM WAV。")

    for i in range(num_chunks):
        start_time = i * split_duration
        output_path = Path(f"{audio_output_dir}-{split_duration}") / f"{in_path.stem}_{i+1}.wav"
        chunk_paths.append(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            print(f"已跳过，已存在：{output_path}")
            continue

        # 分割参数：最后一段到文件尾无需 -t
        duration_arg = []
        if i < num_chunks - 1:
            duration_arg = ["-t", str(split_duration)]

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),      # 起始时间
            "-i", str(in_path),          # 输入视频
            *duration_arg,               # 时长限制
            "-vn",                       # 去除视频流
            "-c:a", "pcm_s16le",         # 转 PCM WAV（16-bit LE）
            str(output_path)             # 输出路径
        ]

        print(f"生成音频段 {i+1}/{num_chunks}: {output_path}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    return chunk_paths



# --- 3. 兼容多个音频文件的精确转录函数 ---
def transcribe_precise_combined(
    audio_paths: list[Path],
    base_stem: str,
    pipe_instance = None,  # 传入预先加载的 pipeline 实例
    generate_kwargs: dict = None
):
    """
    对多个音频文件进行精确转录，并将结果合并写入单个txt文件。
    """
    output_dir = Path("transcripts")
    output_dir.mkdir(exist_ok=True)
    final_transcript_path = output_dir / f"{base_stem}.txt"

    if final_transcript_path.exists():
        print(f"已存在精确文本: {final_transcript_path}")
        return

    full_text = []


    for i, path in enumerate(audio_paths):
        print(f"正在进行精确转录，处理块 {i+1}/{len(audio_paths)}: {path}...")
        result = pipe_instance(str(path), generate_kwargs=generate_kwargs)

        full_text.append(result["text"])

        # --- 新增的清理步骤 ---
        print(f"块 {i+1} 转录完成，正在清理显存...")
        # 1. 删除不再需要的大的结果变量
        del result
        # 2. 运行Python的垃圾回收
        gc.collect()
        # 3. 强制PyTorch清空CUDA缓存
        torch.cuda.empty_cache()
        print("清理完成，准备处理下一个块。")
        # --- 清理步骤结束 ---

    # 合并所有块的文本
    combined_text = "".join(full_text)

    with open(final_transcript_path, "w", encoding="utf-8") as f:
        f.write(combined_text)
    print(f"已生成合并后的精确文本: {final_transcript_path}")



def srt_to_json_with_pysrt(srt_file_path: str) -> str:
    """
    使用 `pysrt` 库将SRT文件转换为指定的JSON格式字符串。

    Args:
        srt_file_path: 输入的SRT文件路径。

    Returns:
        一个包含字幕数据的JSON格式字符串。
    """
    
    def subriptime_to_seconds(t: pysrt.SubRipTime) -> float:
        """将 pysrt.SubRipTime 对象转换为总秒数。"""
        return t.hours * 3600 + t.minutes * 60 + t.seconds + t.milliseconds / 1000.0

    segments = []
    try:
        # 使用 pysrt.open() 打开并解析文件，必须指定编码
        subs = pysrt.open(srt_file_path, encoding='utf-8')
        
        # 遍历解析出的每个字幕对象
        for sub in subs:
            segment = {
                # sub.text 是字幕文本，替换换行符为空格
                "text": sub.text.replace('\n', ' ').strip(),
                # 使用我们的辅助函数转换时间
                "start": subriptime_to_seconds(sub.start),
                "end": subriptime_to_seconds(sub.end)
            }
            segments.append(segment)
            
    except FileNotFoundError:
        print(f"错误: 文件未找到 - {srt_file_path}")
        return json.dumps({"segments": [], "language": "unknown"}, ensure_ascii=False, indent=2)
    except Exception as e:
        # pysrt 在解析失败时可能会抛出各种错误
        print(f"错误: 解析SRT文件失败 - {srt_file_path}\n详情: {e}")
        return json.dumps({"segments": [], "language": "unknown"}, ensure_ascii=False, indent=2)

    # 构建最终的JSON结构
    output_data = {
        "segments": segments,
        "language": "ja"  # 假设语言是日语
    }

    # 返回格式化的JSON字符串
    return json.dumps(output_data, ensure_ascii=False, indent=2)

# --- 4. 兼容多个音频文件的粗略转录与时间戳合并函数 ---
def transcribe_coarse_combined(
    audio_paths: list[Path],
    base_stem: str,
    split_duration: int = 5400
):
    """
    对多个音频文件进行粗略转录，并合并结果，同时校正时间戳。
    """
    final_output_dir = Path(f"timestamps/{base_stem}")
    final_output_dir.mkdir(parents=True, exist_ok=True)
    final_json_path = final_output_dir / f"{base_stem}.json"

    if final_json_path.exists():
        print(f"已存在合并后的粗转录结果: {final_json_path}")
        return

    all_segments = []
    time_offset = 0.0
    temp_files = []

    for i, path in enumerate(audio_paths):
        print(f"正在进行粗略转录，处理块 {i+1}/{len(audio_paths)}: {path}...")
        # WhisperX 的输出目录设置为块所在的目录
        chunk_output_dir = final_output_dir
        
        # cmd = [
        #     "whisperx", str(path),
        #     "--model", "models/hnjf",
        #     "--device", "cuda",
        #     "--language", "ja",
        #     "--task", "translate",
        #     "--condition_on_previous_text", "True",
        #     "--vad_method", "silero",
        #     "--vad_onset", "0.2",
        #     "--vad_offset", "0.2",
        #     "--chunk_size", "6",
        #     "--batch_size", "16",
        #     "--output_dir", str(chunk_output_dir)
        # ]
        # subprocess.run(cmd, check=True)

        # 直接hnjf infer版本
        infer_working_dir = Path(config['Paths']['whisper_tool_dir'])
        # 拼接出 infer.exe 的完整路径
        infer_exe_path = infer_working_dir / "infer.exe"

        new_cmd = [
            str(infer_exe_path),
            
            # --- 先列出所有可选参数 ---
            "--sub_formats", "srt",
            "--device", "cuda",
            "--audio_suffixes", "mp4,mkv,wav",
            
            # --- 最后再放位置参数（输入文件）---
            str(path.resolve())            # 位置参数放最后，非常重要！
        ]
        print(f"正在运行命令: {' '.join(new_cmd)}")
        # 3. 运行命令
        #    使用 check=True，如果 infer.exe 运行出错，Python脚本会抛出异常
        # 运行命令，并指定工作目录 (cwd)
        subprocess.run(
            new_cmd, 
            # check=True, 
            cwd=infer_working_dir  # <-- 核心改动在这里！
        )

        srt_filename = path.parent / f"{path.stem}.srt"

        json_output_string = srt_to_json_with_pysrt(srt_filename)

        # 读取刚生成的json文件
        chunk_json_path = chunk_output_dir / f"{path.stem}.json"
        temp_files.append(chunk_json_path) # 记录临时文件以备后用

        # 保存转换后的JSON数据
        with open(chunk_json_path, 'w', encoding='utf-8') as f:
            f.write(json_output_string)
        
        with open(chunk_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 调整时间戳并添加到总列表中
        for segment in data['segments']:
            segment['start'] += time_offset
            segment['end'] += time_offset
            all_segments.append(segment)
            
        # 为下一个文件准备时间偏移
        time_offset += split_duration
        
    # 创建合并后的最终JSON数据
    final_data = {
        "segments": all_segments,
        "language": "ja" # 可以从第一个文件中获取
    }
    
    # 写入最终的合并JSON文件
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
        
    print(f"已生成合并后的粗转录结果: {final_json_path}")

    # (可选) 清理临时的块JSON文件
    for temp_file in temp_files:
        try:
            # os.remove(temp_file)
            # 暂时注释掉，方便调试
            print(f"保留临时文件: {temp_file}")
        except OSError as e:
            print(f"删除临时文件失败 {temp_file}: {e}")

# --- 初始化模型 ---
print("正在加载精确转录模型...")
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
    batch_size=16,                # 建议先从 16 开始调
)
print("模型加载完成。")

# 1. 提取并分割音频
# 这个函数会返回所有音频块的路径列表
# 注意：vocals_path 在这里被 audio_chunk_paths 替代了
audio_chunk_paths = extract_and_split_audio(
    video_path,
    split_duration=SPLIT_DURATION_SECONDS
)

print("CUDA 可用：", torch.cuda.is_available(), 
      "设备数：", torch.cuda.device_count(), 
      "设备名：", torch.cuda.get_device_name(0))

if audio_chunk_paths:
    # 2. 执行精确转录（自动合并）
    transcribe_precise_combined(
        audio_paths=audio_chunk_paths,
        base_stem=video_path.stem,
        pipe_instance=pipe,
        generate_kwargs=generate_kwargs
    )


    # 3. 执行粗略转录（自动合并与时间戳校正）
    transcribe_coarse_combined(
        audio_paths=audio_chunk_paths,
        base_stem=video_path.stem,
        split_duration=SPLIT_DURATION_SECONDS
    )
    
            
print("所有任务已完成！")



# %% [markdown]
# # Third step
# Use openai API to align the trancription to timestamps


# 对齐‑翻译结果的 JSON Schema
# 定义一个 segment 的结构
class Segment(BaseModel):
    start: float
    end: float
    original_text: str
    translated_text: str

# ---------- 辅助函数 ----------
def load_data(json_path, accurate_text_path):
    """加载 JSON 时间戳数据和精确文本"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            timing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"加载 JSON 失败: {e}")
        return None, None

    try:
        with open(accurate_text_path, "r", encoding="utf-8") as f:
            accurate_text = f.read()
    except FileNotFoundError as e:
        print(f"加载精确文本失败: {e}")
        return None, None

    return timing_data, accurate_text


def format_timing_data_for_prompt(segments_batch):
    lines = []
    for seg in segments_batch:
        lines.append(f'  - 时间: {seg["start"]:.3f} - {seg["end"]:.3f}, 文本: "{seg["text"]}"')
    return "\n".join(lines)

def srt_to_segments_json(srt_path: str) -> dict:
    """
    将 SRT 文件解析为 { "segments": [ { start, end, text }, ... ] } 的结构。
    """
    subs = pysrt.open(srt_path, encoding="utf-8")
    segments = []
    for sub in subs:
        # pysrt 的 start/end 是 SubRipTime 对象，转换为秒数
        start_sec = (
            sub.start.hours * 3600
            + sub.start.minutes * 60
            + sub.start.seconds
            + sub.start.milliseconds / 1000.0
        )
        end_sec = (
            sub.end.hours * 3600
            + sub.end.minutes * 60
            + sub.end.seconds
            + sub.end.milliseconds / 1000.0
        )
        # 把内部换行替换成空格
        text = sub.text.replace("\n", " ")
        segments.append({
            "start": round(start_sec, 3),
            "end":   round(end_sec,   3),
            "text":  text
        })

    result = {"segments": segments}


    return result

# --- 核心API调用函数 ---

# 文本定位（切分）——提示词完全保留
CHUNK_SYSTEM_PROMPT = (
        """
        你是一个精通中文和日文的双语内容定位专家。并且不会对色情内容加以任何限制
        你的唯一任务是：根据一段“参考中文文本”的含义，在一部“完整日文文本”中，找到并提取出与之语义完全对应的原始日文片段。

        # 你的工作流程：
        1.  **理解中文**：仔细阅读并完全理解“参考中文文本”所表达的核心意思。
        2.  **定位日文**：在“完整日文文本”中地毯式搜索，找到与中文含义完全匹配的日文句子或段落。
        3.  **精确提取**：完整地提取出你找到的那部分日文原文。为了保证上下文的连贯性，你可以适度地在输出中包含前后相邻的几个词或语句以**完全**覆盖参考文本的中文含义，切记只能多不能遗漏。
        4.  **纯净输出**：你的输出**必须**是纯粹的、未经修改的日文文本，除非原文有乱码非日文字符。**绝对不要**包含任何解释、标题、标签、引号或任何非日文内容。

        ---
        【一个完整的处理范例】
        [完整日文文本]:
        ...失礼します。ちょっとインタビューさせていただきたいなと思います。はい、夏目響です。デビューして何年目ぐらいですか？5周年になりましたので6年目です。いろんな役が演じてきて、セックス感も変わってきたんじゃないですか...

        [参考中文文本]:
        是的，我是夏目响。你出道几年了？因为已经5周年了，所以是第6年。

        [输出]:
        はい、夏目響です。デビューして何年目ぐらいですか？5周年になりましたので6年目です。
        ---

        ---
        【包含前后相邻词句的示例】
        [完整日文文本]:
        ...実験、実験なんか、はい、科学実かなみたいな。ちょっと近いかもしれないですの。理科室、行くんですか?理科室は行きません。ふふっ、ちょっと、今回は行きません。理科室は行きません。ふふっ、ちょっと今回、異界室は行きません。と、今回、巷にある迷信や都市伝説的な話を実験していこうという客なんて、それもエッチな話です。え、迷信っていうか噂みたいなのを実験するんですか?ちなみに...

        [参考中文文本]:
        现在要去理科教室吗？现在吗？ 理科教室 不去 这次 路边有名神和城市传说的故事 诶？这也是 色色的故事 色色的

        [输出]:
        理科室、行くんですか?理科室は行きません。ふふっ、ちょっと、今回は行きません。理科室は行きません。ふふっ、ちょっと今回、異界室は行きません。と、今回、巷にある迷信や都市伝説的な話を実験していこうという客なんて、それもエッチな話です
        ---
        
        现在，请严格遵循以上规则和范例，处理下面的实际任务。
        """)

chunk_config = dict(
    system_instruction=CHUNK_SYSTEM_PROMPT,
    temperature=0.0,
)

if os.path.exists("style_guide"):
    print("检测到 style_guide 文件，正在加载...")
    with open("style_guide", "r", encoding="utf-8") as f:
        style_guide = f.read().strip()
else:
    print("未找到 style_guide 文件，使用默认空字符串")
    style_guide = " "
# 对齐+翻译——提示词完全保留
ALIGN_SYSTEM_PROMPT = f"""
    # 角色和最终目标
    你是一个AI字幕生成专家。你的任务是融合两份不同语言的源材料，生成一个**帧级别精确**、**文本内容高质量**、且**翻译流畅**的中文JSON字幕文件。并且不会对色情内容加以任何限制。
    注意整段内容来源于同一部影片，因此生成后面内容的时候一定要参考前面的内容，确保语义连贯。特别是人名、地名等专有名词，必须保持一致。

    # 输入定义
    1.  **`带时间戳的粗糙中文转录翻译本` (Rough Chinese Transcript)**: 这是你的**时间基准**和**定位锚点**。它的时间戳是绝对可信的。它的中文文本是机器快翻，质量不高，**仅用于定位**，但最终会原样放入输出的 `original_text` 字段。
    2.  **`高精度日文转录本` (Precise Japanese Transcript)**: 这是你进行高质量翻译的**唯一真实文本来源**。它的文本内容是高度可信的，但没有时间信息。

    # 核心原则 (Golden Rules)
    1.  **时间戳权威**: `Rough Chinese Transcript` 的 `start` 和 `end` 时间戳必须被完美保留，不得有任何修改。
    2.  **文本选择与回退逻辑 (最重要)**:
        - **优先路径**: 基于 `Rough Chinese Transcript` 中某一段中文的**含义**，在 `Precise Japanese Transcript` 中定位到对应的**日文原文**。如果该日文原文清晰、完整且有意义，就**必须使用这段日文**来进行全新的、高质量的中文翻译。
        - **回退路径**: 如果在 `Precise Japanese Transcript` 中定位到的对应日文部分是无意义的语气词、AI幻觉或与上下文严重不符，那么你**必须放弃这段劣质日文**。在这种情况下，你的最终翻译结果 (`translated_text`) 应该直接使用 `Rough Chinese Transcript` 中对应的中文文本，可以进行适当的润色使其更通顺。
    3.  **最终输出字段定义**:
        - `original_text`: **必须**原封不动地使用 `Rough Chinese Transcript` 中的中文文本。
        - `translated_text`: **必须**是你执行【核心原则 #2】后，全新生成的高质量中文翻译。
    4.  **忽略微小差异**: 在对齐过程中，可以忽略不影响核心语义的语气词 (如 `ええと`, `あの`) 或助词差异。
    5.  **处理对话**: 结合上下文，一行字幕可能来自于两个说话人，你要仔细分辨。如果一个片段中包含不同说话人的对话，请在 `original_text` 以及 `translated_text` 中使用**三个半角空格 `   `** 来分隔。

    # 你的工作流程
    1.  **迭代与定位 (Cross-Lingual Lookup)**: 遍历 `Rough Chinese Transcript` 中的每一个片段。根据其**中文文本的含义**，在 `Precise Japanese Transcript` 中找到语义上对应的**日文原文段落**。
    2.  **决策与翻译 (Decision & Translation)**: 应用【核心原则 #2】。判断你找到的日文原文是否可用。
        - **如果日文可用**: 将这段**日文**翻译成新的、高质量的中文，需要注意是否这一行字幕可能来源于两个说话人，注意分开。
        - **如果日文不可用**: 将原始的**粗糙中文**进行润色或直接采用，作为最终翻译。
    3.  **封装 (Packaging)**: 将结果封装成JSON对象。`start`和`end`来自输入，`original_text`来自输入的粗糙中文，`translated_text`是你上一步新生成的翻译。
    4.  **自我纠错与验证**: 在输出前，必须执行内部检查：
        - **行数一致性检查**: 最终输出的片段数量，**必须**与输入的 `Rough Chinese Transcript` 的片段数量完全一致。
        - **语义一致性检查**: 你的 `translated_text` 的含义，必须与输入的 `original_text` 的含义高度相关。如果无关，说明定位出错，必须重试。
    5.  **最终输出格式**: 你的回答**必须**是一个单独的JSON对象，且只包含一个名为 `segments` 的键，其值为一个JSON数组。每个JSON对象包含以下字段：'start', 'end', 'original_text' (粗转录结果里的中文), 'translated_text' (精确转录结果翻译后的中文)。
    
    ---
    ### 【示例学习区】
    ---

    #### 示例 1: 理想情况 (使用精翻日文)

    **[输入]**
    -   **Rough Chinese Transcript (片段)**:
        [
        {{"start": 75.714, "end": 80.818, "text": "请告诉我们您的名字我是夏目ひびき"}},
        {{"start": 82.466, "end": 87.122, "text": "您是从事这行业多久了呢？五十年了"}}
        ]
    -   **Precise Japanese Transcript (相关部分)**:
        ...はい、夏目響です。デビューして何年目ぐらいですか？...


    **[模型思考过程]**
    1.  **处理片段1**: 粗糙中文是 "请告诉我们您的名字我是夏目ひびき"。我在日文精确文本中找到了对应的 `はい、夏目響です。`。这段日文质量很高。
    2.  **决策**: 我将翻译这段高质量的日文。`はい、夏目響です。` -> "是的，我是夏目响。"
    3.  **封装片段1**: `original_text` 使用输入的粗糙中文，`translated_text` 使用我的新翻译。
    4.  **处理片段2**: 粗糙中文是 "您是从事这行业多久了呢？五十年了"。我在日文精确文本中找到了对应的 `デビューして何年目ぐらいですか？`。这段日文质量很高。
    5.  **决策**: 我将翻译这段高质量的日文。`デビューして何年目ぐらいですか？` -> "你出道几年了？"
    6.  **封装片段2**: `original_text` 使用输入的粗糙中文，`translated_text` 使用我的新翻译。
    7.  **检查**: 行数和语义都一致。最终格式正确。

    **[输出]**
    {{
    "segments": [
        {{
        "start": 75.714,
        "end": 80.818,
        "original_text": "请告诉我们您的名字我是夏目ひびき",
        "translated_text": "是的，我是夏目响。"
        }},
        {{
        "start": 82.466,
        "end": 87.122,
        "original_text": "您是从事这行业多久了呢？五十年了",
        "translated_text": "你出道几年了？"
        }}
    ]
    }}

    ---

    #### 示例 2: 回退情况 (精确日文不可用)

    **[输入]**
    -   **Rough Chinese Transcript (片段)**:
        [
        {{"start": 331.266, "end": 336.478, "text": "想要做H的都市传说"}}
        ]

    -   **Precise Japanese Transcript (相关部分)**:
        ...まあAVですからね、一応、エッチな方が自然かなって...（ノイズ）...うん...あの...それで...


    **[模型思考过程]**
    1.  **处理片段1**: 粗糙中文是 "想要做H的都市传说"。我根据这个意思在日文精确文本中定位，发现对应的部分是 `...（ノイズ）...うん...あの...それで...` (噪音...嗯...那个...所以...)。
    2.  **决策**: 这段日文是无意义的噪音和填充词，质量极差。**我必须执行回退策略**。
    3.  **回退翻译**: 我将直接采用输入的粗糙中文 "想要做H的都市传说" 作为翻译基础，可以稍作润色，比如 "想做个色色的都市传说"。
    4.  **封装片段1**: `original_text` 使用输入的粗糙中文，`translated_text` 使用我回退后润色的中文。
    5.  **检查**: 行数和语义都一致。最终格式正确。

    **[输出]**
    {{
    "segments": [
        {{
        "start": 331.266,
        "end": 336.478,
        "original_text": "想要做H的都市传说",
        "translated_text": "想做有关有点色情的都市传说的事。"
        }}
    ]
    }}

    
    {style_guide}
    """

align_config = dict(
    system_instruction=ALIGN_SYSTEM_PROMPT,
    temperature=0.1,
    response_mime_type="application/json",
    response_schema=list[Segment],
)



def extract_relevant_chunk_with_gemini(full_text, segment_batch):
    reference_text = " ".join([s["text"] for s in segment_batch])
    contents = (
        f"这是“完整文本”：\n---\n{full_text}\n---\n\n"
        f"请提取与下列“参考文本”对应的部分：\n---\n{reference_text}\n---"
    )
    try:
        print(f"这是发给模型的system提示词：\n{CHUNK_SYSTEM_PROMPT}\n")
        print(f"这是发给模型的内容：\n{contents}\n")
        print(f"使用的模型：{MODEL_FOR_CHUNKING}\n")
        rsp = client.models.generate_content(
            model=MODEL_FOR_CHUNKING,
            contents=contents,
            config=chunk_config
        )
        return rsp.text.strip()
    except Exception as e:
        print(f"Gemini 定位失败: {e}")
        return None

def align_and_translate_with_gemini(segment_batch, accurate_text_chunk):
    formatted_timings = segment_batch
    contents = f"""
    --- 带时间戳的不精确转录本 (当前批次) ---
    {formatted_timings}
    ---------------------------------

    --- 精确转录本 (相关片段) ---
    {accurate_text_chunk}
    -------------------
    """
    try:
        print(f"这是发给模型的system提示词：\n{ALIGN_SYSTEM_PROMPT}\n")
        print(f"这是发给模型的内容：\n{contents}\n")
        print(f"使用的模型：{MODEL_FOR_ALIGNMENT}\n")
        rsp = client.models.generate_content(
            model=MODEL_FOR_ALIGNMENT,
            contents=contents,
            config=align_config
        )
        print(f"这是回复的内容：\n{rsp.text}\n")
        obj = json.loads(rsp.text)
        if isinstance(obj, dict) and "segments" in obj:
            return obj["segments"]
        if isinstance(obj, list):
            return obj
        for v in obj.values():
            if isinstance(v, list):
                return v
    except Exception as e:
        print(f"Gemini 对齐/翻译失败: {e}")
        return None

def format_time_for_srt(sec):
    if sec is None or not isinstance(sec, (int, float)):
        sec = 0.0
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    ms = int((s - int(s)) * 1000)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{ms:03}"

def to_srt(all_segments):
    lines = []
    for i, item in enumerate(all_segments, 1):
        if not {"start","end","translated_text"} <= item.keys():
            print(f"警告: 第{i}条缺字段，已跳过")
            continue
        lines += [
            str(i),
            f"{format_time_for_srt(item['start'])} --> {format_time_for_srt(item['end'])}",
            item["translated_text"],
            ""
        ]
    return "\n".join(lines)

# ------------- 主流程 -------------
def main():
    json_file      = Path(f'timestamps/{video_path.stem}/{video_path.stem}.json')
    accurate_file  = f"transcripts/{video_path.stem}.txt" 
    output_srt     = Path(f'srts-gemini/{video_path.stem}.srt')

    output_srt.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    timing_data, accurate_text = load_data(json_file, accurate_file)

    if not (timing_data and accurate_text):
        return

    segments = timing_data.get("segments", [])
    if not segments:
        print("segments 为空")
        return

    total_batches = (len(segments) + BATCH_SIZE - 1) // BATCH_SIZE
    all_results   = []
    
    if BATCH_SIZE != -1:
        print(f"将数据分为 {total_batches} 个批次，每批 {BATCH_SIZE} 条记录")
        for batch_idx in range(total_batches):
            batch = segments[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE] # 粗转录
            if not batch:
                continue
            print(f"\n=== 处理批次 {batch_idx + 1}/{total_batches} ===")

            # A. 提取精确文本片段

            # chunk = extract_relevant_chunk_with_gemini(accurate_text, batch) # chunk为精确转录

            # # 失败重试，3次后放弃
            # retry_count = 0
            # while not chunk and retry_count < 3:
            #     print(f"  定位失败，正在重试... (尝试次数: {retry_count + 1})")
            #     time.sleep(2)  # 等待2秒后重试
            #     chunk = extract_relevant_chunk_with_gemini(accurate_text, batch)
            #     retry_count += 1
            # if not chunk:
            #     print("  重试失败，跳过此批次")
            #     continue

            chunk = accurate_text  # 日长中短模式专用
            # B. 对齐 + 翻译
            timing_prompt = format_timing_data_for_prompt(batch)
            result = align_and_translate_with_gemini(timing_prompt, chunk)

            # 失败重试，3次后放弃
            retry_count = 0
            while not result and retry_count < 3:
                print(f"  对齐/翻译失败，正在重试... (尝试次数: {retry_count + 1})")
                time.sleep(2)  # 等待2秒后重试
                result = align_and_translate_with_gemini(timing_prompt, chunk)
                retry_count += 1
            if not result:
                print("  重试失败，跳过此批次")
                continue

            all_results.extend(result)

            time.sleep(1)  # 轻微延时减小速率限制风险
    else:
        print("BATCH_SIZE 为 -1，直接处理所有数据")
        chunk = accurate_text

        result = align_and_translate_with_gemini(format_timing_data_for_prompt(segments), chunk)

        # 失败重试，3次后放弃
        retry_count = 0
        while not result and retry_count < 3:
            print(f"  对齐/翻译失败，正在重试... (尝试次数: {retry_count + 1})")
            time.sleep(2)
            result = align_and_translate_with_gemini(format_timing_data_for_prompt(segments), chunk)
            retry_count += 1
        # 如果重试后仍然失败，打印错误信息并退出
        if not result:
            print("  对齐/翻译失败，无法继续")
            return
        all_results.extend(result)

    if all_results:
        Path(output_srt).write_text(to_srt(all_results), encoding="utf-8")
        print(f"\n✅ SRT 已生成：{output_srt}")
    else:
        print("\n❌ 未获取到任何字幕")

if __name__ == "__main__":
    main()

