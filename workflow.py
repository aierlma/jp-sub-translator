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

import pysrt
import torch
from transformers import pipeline

import os
import openai
import json

import time # 引入time模块，用于在API调用之间添加延迟
from dotenv import load_dotenv
load_dotenv()

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY2"]) # Ensure you have set your OpenAI API key in the environment variable or .env file
BATCH_SIZE = 30
MODEL_FOR_ALIGNMENT = "gpt-4.1"  # gpt4.1 接受nsfw文本
MODEL_FOR_CHUNKING  = "gpt-4.1"

video_path = Path("D:/Downloads/x.mp4")
# 基于 your_movie.wav 分离人声
demucs_model = "htdemucs" 
# 分离后的人声文件路径
vocals_path = Path(f"separated/{demucs_model}/{video_path.stem}/vocals.wav")



def extract_audio(input_video: str,
                sample_rate: int = 16000,
                channels: int = 1,
                codec: str = "pcm_s16le") -> Path:
    """
    从任意视频文件中提取音频，输出 WAV 并返回输出文件路径。
    文件名：保留原始 stem，后缀改为 .wav
    """
    in_path = Path(input_video)
    stem = in_path.stem
    audio_output = in_path.with_name(f"{stem}.wav")

    cmd = [
        "ffmpeg",
        "-y",  # 如果输出已存在则覆盖
        "-i", str(in_path),
        "-vn",
        "-acodec", codec,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        str(audio_output)
    ]
    subprocess.run(cmd, check=True)
    return audio_output

def separate_vocals(audio_input: Path,
                        stems: str = "vocals",
                        model: str = "htdemucs") -> None:
        """
        对提取后的音频做人声分离，两声道模式下只保留 vocals。
        Demucs 会在当前目录下创建一个 separated/<model>/<stem> 文件夹。
        """
        cmd = [
            "python", "-m", "demucs",
            f"--two-stems={stems}",
            "-n", model,
            str(audio_input)
        ]
        subprocess.run(cmd, check=True)

# 若已经存在分离后的人声文件，则不再重复分离
if vocals_path.exists():
    print(f"已存在分离后的人声文件: {vocals_path}")
else:
    # 提取音频，自动生成 your_movie.wav
    wav_path = extract_audio(video_path,
                                sample_rate=16000,
                                channels=1)

    # 基于 your_movie.wav 分离人声
    separate_vocals(wav_path,
                    stems="vocals",
                    model=demucs_model)

    # 删除原始音频文件
    if wav_path.exists():
        delete_cmd = ["rm", str(wav_path)]
        print(f"删除原始音频文件: {wav_path}")
        subprocess.run(delete_cmd, check=True)


# %%
generate_kwargs = {
    "language": "Japanese",
    "no_repeat_ngram_size": 0,
    "repetition_penalty": 1.0,
}
pipe = pipeline(
    "automatic-speech-recognition",
    model="./models/anime",
    device="cuda",

    torch_dtype=torch.float16,
    chunk_length_s=30.0,
    batch_size=64,
)

os.makedirs("transcripts", exist_ok=True)
if os.path.exists(f"transcripts/{video_path.stem}.txt"):
    print(f"已存在精确文本: transcripts/{video_path.stem}.txt")
else:
    result = pipe(str(vocals_path), generate_kwargs=generate_kwargs)

    with open(f"transcripts/{video_path.stem}.txt", "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"已生成精确文本: transcripts/{video_path.stem}.txt")

# %%



out_dir = Path(f"timestamps/{video_path.stem}")
os.makedirs(out_dir, exist_ok=True)

if os.path.exists(out_dir / "vocals.json"):
    print(f"已存在粗转录结果: {out_dir / 'vocals.json'}")
else:
    cmd = [
        "whisperx", str(vocals_path),
        "--model", "models/hnjf",   # 海南鸡饭大佬的转中文模型
        "--device", "cuda",
        "--language", "ja",
        "--task", "translate",
        "--condition_on_previous_text", "True",
        "--vad_method", "silero",
        "--chunk_size", "6",
        "--batch_size", "16",
        "--output_dir", str(out_dir)
    ]
    subprocess.run(cmd, check=True)
    print(f"已生成粗转录结果: {out_dir / 'vocals.json'}")


# %% [markdown]
# # Third step
# Use openai API to align the trancription to timestamps


# 对齐‑翻译结果的 JSON Schema
ALIGN_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start":           {"type": "number"},
                    "end":             {"type": "number"},
                    "original_text":   {"type": "string"},
                    "translated_text": {"type": "string"},
                },
                # 这里必须列出所有属性
                "required": ["start", "end", "original_text", "translated_text"],
                "additionalProperties": False
            }
        }
    },
    "required": ["segments"],
    "additionalProperties": False
}

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

def extract_relevant_chunk_with_api(full_accurate_text, segment_batch):
    """用 Responses API 从长文本中定位与当前批次对应的片段"""
    instructions = (
        """
        你是一个精通中文和日文的双语内容定位专家。
        你的唯一任务是：根据一段“参考中文文本”的含义，在一部“完整日文文本”中，找到并提取出与之语义完全对应的原始日文片段。

        # 你的工作流程：
        1.  **理解中文**：仔细阅读并完全理解“参考中文文本”所表达的核心意思。
        2.  **定位日文**：在“完整日文文本”中地毯式搜索，找到与中文含义完全匹配的日文句子或段落。
        3.  **精确提取**：完整地提取出你找到的那部分日文原文。为了保证上下文的连贯性，你可以适度地包含前后相邻的几个词或语句，切记只能多不能遗漏。
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

        现在，请严格遵循以上规则和范例，处理下面的实际任务。
        """
    )
    reference_text = " ".join([s["text"] for s in segment_batch])
    user_input = (
        f"这是“完整文本”：\n---\n{full_accurate_text}\n---\n\n"
        f"请提取与下列“参考文本”对应的部分：\n---\n{reference_text}\n---"
    )

    print(" -> 正在定位精确文本片段…")
    try:
        rsp = client.responses.create(
            model=MODEL_FOR_CHUNKING,
            instructions=instructions,
            input=user_input,
            temperature=0.0,
        )
        return rsp.output_text.strip()
    except Exception as e:
        print(f"定位失败: {e}")
        return None

def align_and_translate(formatted_timings, accurate_text_chunk):
    """对齐 + 翻译，要求模型按 JSON Schema 输出"""
    if os.path.exists("style_guide"):
        print("检测到 style_guide 文件，正在加载...")
        with open("style_guide", "r", encoding="utf-8") as f:
            style_guide = f.read().strip()
    else:
        print("未找到 style_guide 文件，使用默认空字符串")
        style_guide = " "
    instructions = f"""
    # 角色和最终目标
    你是一个AI字幕生成专家。你的任务是融合两份不同语言的源材料，生成一个**帧级别精确**、**文本内容高质量**、且**翻译流畅**的中文JSON字幕文件。

    # 输入定义
    1.  **`带时间戳的粗糙中文转录本` (Rough Chinese Transcript)**: 这是你的**时间基准**和**定位锚点**。它的时间戳是绝对可信的。它的中文文本是机器快翻，质量不高，**仅用于定位**，但最终会原样放入输出的 `original_text` 字段。
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
    5.  **处理对话**: 如果一个片段中包含不同说话人的对话，请在 `original_text` 以及 `translated_text` 中使用**三个半角空格 `   `** 来分隔。

    # 你的工作流程
    1.  **迭代与定位 (Cross-Lingual Lookup)**: 遍历 `Rough Chinese Transcript` 中的每一个片段。根据其**中文文本的含义**，在 `Precise Japanese Transcript` 中找到语义上对应的**日文原文段落**。
    2.  **决策与翻译 (Decision & Translation)**: 应用【核心原则 #2】。判断你找到的日文原文是否可用。
        - **如果日文可用**: 将这段**日文**翻译成新的、高质量的中文。
        - **如果日文不可用**: 将原始的**粗糙中文**进行润色或直接采用，作为最终翻译。
    3.  **封装 (Packaging)**: 将结果封装成JSON对象。`start`和`end`来自输入，`original_text`来自输入的粗糙中文，`translated_text`是你上一步新生成的翻译。
    4.  **自我纠错与验证**: 在输出前，必须执行内部检查：
        - **行数一致性检查**: 最终输出的片段数量，**必须**与输入的 `Rough Chinese Transcript` 的片段数量完全一致。此处应为{BATCH_SIZE}个片段。
        - **语义一致性检查**: 你的 `translated_text` 的含义，必须与输入的 `original_text` 的含义高度相关。如果无关，说明定位出错，必须重试。
    5.  **最终输出格式**: 你的回答**必须**是一个单独的JSON对象，且只包含一个名为 `segments` 的键，其值为一个JSON数组。每个JSON对象包含以下字段：'start', 'end', 'original_text' (粗转录结果里的中文), 'translated_text' (精确转录结果翻译后的中文)。
    
    ---
    ### 【示例学习区】
    ---

    #### 示例 1: 理想情况 (使用精翻日文)

    **[输入]**
    -   **Rough Chinese Transcript (片段)**:
        ```json
        [
        {{"start": 75.714, "end": 80.818, "text": "请告诉我们您的名字我是夏目ひびき"}},
        {{"start": 82.466, "end": 87.122, "text": "您是从事这行业多久了呢？五十年了"}}
        ]
        ```
    -   **Precise Japanese Transcript (相关部分)**:
        ```text
        ...はい、夏目響です。デビューして何年目ぐらいですか？...
        ```

    **[模型思考过程]**
    1.  **处理片段1**: 粗糙中文是 "请告诉我们您的名字我是夏目ひびき"。我在日文精确文本中找到了对应的 `はい、夏目響です。`。这段日文质量很高。
    2.  **决策**: 我将翻译这段高质量的日文。`はい、夏目響です。` -> "是的，我是夏目响。"
    3.  **封装片段1**: `original_text` 使用输入的粗糙中文，`translated_text` 使用我的新翻译。
    4.  **处理片段2**: 粗糙中文是 "您是从事这行业多久了呢？五十年了"。我在日文精确文本中找到了对应的 `デビューして何年目ぐらいですか？`。这段日文质量很高。
    5.  **决策**: 我将翻译这段高质量的日文。`デビューして何年目ぐらいですか？` -> "你出道几年了？"
    6.  **封装片段2**: `original_text` 使用输入的粗糙中文，`translated_text` 使用我的新翻译。
    7.  **检查**: 行数和语义都一致。最终格式正确。

    **[输出]**
    ```json
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
    ```

    ---

    #### 示例 2: 回退情况 (精确日文不可用)

    **[输入]**
    -   **Rough Chinese Transcript (片段)**:
        ```json
        [
        {{"start": 331.266, "end": 336.478, "text": "想要做H的都市传说"}}
        ]
        ```
    -   **Precise Japanese Transcript (相关部分)**:
        ```text
        ...まあAVですからね、一応、エッチな方が自然かなって...（ノイズ）...うん...あの...それで...
        ```

    **[模型思考过程]**
    1.  **处理片段1**: 粗糙中文是 "想要做H的都市传说"。我根据这个意思在日文精确文本中定位，发现对应的部分是 `...（ノイズ）...うん...あの...それで...` (噪音...嗯...那个...所以...)。
    2.  **决策**: 这段日文是无意义的噪音和填充词，质量极差。**我必须执行回退策略**。
    3.  **回退翻译**: 我将直接采用输入的粗糙中文 "想要做H的都市传说" 作为翻译基础，可以稍作润色，比如 "想做个色色的都市传说"。
    4.  **封装片段1**: `original_text` 使用输入的粗糙中文，`translated_text` 使用我回退后润色的中文。
    5.  **检查**: 行数和语义都一致。最终格式正确。

    **[输出]**```json
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
    ```
    
    {style_guide}
    """

    user_input = f"""
    请处理以下数据：

    --- 带时间戳的不精确转录本 (当前批次) ---
    {formatted_timings}
    ---------------------------------

    --- 精确转录本 (相关片段) ---
    {accurate_text_chunk}
    -------------------

    请严格按照指示，完成对齐和翻译，并仅返回JSON格式的输出。
    """
    
    print("正在调用OpenAI API处理当前批次，请稍候...")
    
    try:
        rsp = client.responses.create(
            model=MODEL_FOR_ALIGNMENT,
            instructions=instructions,
            input=user_input,
            text={                    # 这里才是关键
            "format": {
            "type":   "json_schema",
            "name":   "aligned_segments",
            "schema": ALIGN_SCHEMA,   # 你的 JSON Schema
            "strict": True            # 建议打开严格模式
        }
    }
        )
        out_obj  = json.loads(rsp.output_text)   # rsp.output_text 是字符串
        seg_list = out_obj["segments"]           # 取出真正的字幕列表

        return seg_list                          # 主流程依然收到 list

    except Exception as e:
        print(f"调用OpenAI API时发生错误: {e}")
        return None
# --- SRT生成函数 (来自你的代码，做了一点小优化) ---

def format_time_for_srt(sec):
    if sec is None:
        sec = 0.0
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    ms = int((s - int(s)) * 1000)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{ms:03}"

def to_srt(data):
    lines = []
    for i, item in enumerate(data, 1):
        if not {"start", "end", "translated_text"} <= item.keys():
            print(f"警告: 第 {i} 条数据缺字段，已跳过")
            continue
        lines += [
            str(i),
            f"{format_time_for_srt(item['start'])} --> {format_time_for_srt(item['end'])}",
            item["translated_text"],
            "",
        ]
    return "\n".join(lines)


# --- 主逻辑函数 (重构后) ---

def main():
    # --- 文件路径 ---
    json_file      = Path(f'timestamps/{video_path.stem}/vocals.json')
    accurate_file  = f"transcripts/{video_path.stem}.txt" 
    output_srt     = f'srts/{video_path.stem}.srt'

    os.makedirs('srts', exist_ok=True)  # 确保输出目录存在


    timing_data, accurate_text = load_data(json_file, accurate_file)

    if not (timing_data and accurate_text):
        return

    segments = timing_data.get("segments", [])
    if not segments:
        print("segments 为空")
        return

    total_batches = (len(segments) + BATCH_SIZE - 1) // BATCH_SIZE
    all_results   = []

    for batch_idx in range(total_batches):
        batch = segments[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        if not batch:
            continue
        print(f"\n=== 处理批次 {batch_idx + 1}/{total_batches} ===")

        # A. 提取精确文本片段
        chunk = extract_relevant_chunk_with_api(accurate_text, batch)
        if not chunk:
            print("  无法提取片段，跳过")
            continue

        # B. 对齐 + 翻译
        timing_prompt = format_timing_data_for_prompt(batch)
        result = align_and_translate(timing_prompt, chunk)
        if result:
            all_results.extend(result)
            print(f"  成功获得 {len(result)} 条字幕")
        else:
            print("  对齐/翻译失败")

        time.sleep(1)  # 轻微延时减小速率限制风险

    if all_results:
        Path(output_srt).write_text(to_srt(all_results), encoding="utf-8")
        print(f"\n✅ 已生成 SRT: {output_srt}")
    else:
        print("\n❌ 未获得任何有效字幕")


if __name__ == "__main__":
    main()


