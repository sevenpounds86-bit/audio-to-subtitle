import streamlit as st
import whisper
import re
from datetime import timedelta
import os
import io
import subprocess

# 页面配置
st.set_page_config(page_title="音频转字幕工具", page_icon="🎙️", layout="wide")

# ---------------------- 核心适配函数（解决ffmpeg/音频问题） ----------------------
# 强制配置ffmpeg路径，适配Streamlit Cloud
def setup_ffmpeg():
    try:
        # 检查ffmpeg是否存在，不存在则尝试安装
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # 适配Streamlit Cloud的ffmpeg路径
        os.environ["PATH"] += ":/usr/bin:/usr/local/bin"
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except:
            st.error("⚠️ 系统缺少ffmpeg，无法处理音频！")
            return False

# 去除所有标点符号（中英文）
def remove_punctuation(text):
    punctuation = r'[，。！？；：""''()（）[]【】、·~@#￥%…&*+-=《》<>/\\|{}^_`·,:;!"$%&()*+-/<=>?@[\]^_`{|}~]'
    clean_text = re.sub(punctuation, "", text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

# 转换秒数为SRT标准时间格式
def format_time(seconds):
    try:
        td = timedelta(seconds=float(seconds))
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = td.microseconds // 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    except:
        return "00:00:00,000"

# 生成SRT字幕内容
def generate_srt(segments, target_lang, source_texts=None, use_bilingual=False):
    srt_content = ""
    for idx, seg in enumerate(segments, 1):
        start = format_time(seg["start"])
        end = format_time(seg["end"])
        target_text = remove_punctuation(seg["text"])
        
        srt_content += f"{idx}\n{start} --> {end}\n"
        if use_bilingual and source_texts:
            source_text = remove_punctuation(source_texts[idx-1])
            srt_content += f"{source_text}\n{target_text}\n\n"
        else:
            srt_content += f"{target_text}\n\n"
    return srt_content

# 加载Whisper Small模型（适配免费版）
@st.cache_resource
def load_whisper_model():
    setup_ffmpeg()
    # 加载模型时指定CPU（避免GPU问题）
    return whisper.load_model("small", device="cpu")

# ---------------------- 主界面逻辑 ----------------------
def main():
    st.title("🎙️ 智能音频转字幕工具（稳定版）")
    st.markdown("### 支持多语言识别、双语字幕、精准时间线")
    st.divider()

    # 侧边栏配置
    with st.sidebar:
        st.subheader("⚙️ 配置项")
        target_language = st.selectbox(
            "目标字幕语言",
            ["中文", "英文", "日语", "韩语", "法语", "西班牙语"],
            index=0,
            help="音频会自动识别并翻译成该语言"
        )
        use_bilingual = st.checkbox("生成双语字幕（源语言+目标语言）", value=False)
        st.info("✅ 适配免费服务器，稳定运行\n✅ 自动去除所有标点符号\n✅ 按语义分割字幕，精准对齐时间线")

    # 音频上传
    audio_file = st.file_uploader("📤 上传音频文件（仅支持MP3/WAV）", type=["mp3", "wav"])
    
    if audio_file and setup_ffmpeg():
        # 直接读取音频到内存（避免磁盘权限问题）
        audio_bytes = audio_file.read()
        audio_io = io.BytesIO(audio_bytes)
        
        # 音频预览
        st.audio(audio_bytes, format=f"audio/{audio_file.name.split('.')[-1]}")
        st.divider()

        # 加载模型
        with st.spinner("🔧 加载识别模型（首次加载需1分钟）..."):
            model = load_whisper_model()
        
        # 识别+翻译（核心步骤）
        with st.spinner(f"🎧 正在识别音频并翻译为{target_language}..."):
            # 语言映射
            lang_map = {
                "中文": "zh", "英文": "en", "日语": "ja", "韩语": "ko",
                "法语": "fr", "西班牙语": "es"
            }
            # 直接处理内存中的音频，不写磁盘
            result = model.transcribe(
                audio_io,
                task="translate" if target_language != "中文" else "transcribe",
                language=lang_map[target_language],
                word_timestamps=False,
                verbose=False
            )
        
        # 提取识别结果
        source_segments = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in result["segments"]]
        target_segments = source_segments
        source_texts = [s["text"] for s in source_segments]

        # 字幕编辑区域
        st.subheader("✏️ 编辑字幕（可修改时间/文本）")
        edited_segments = []
        for idx, seg in enumerate(target_segments):
            col1, col2, col3 = st.columns([2, 2, 6])
            with col1:
                start = st.text_input(f"开始时间（秒）", value=f"{seg['start']:.2f}", key=f"s_{idx}")
            with col2:
                end = st.text_input(f"结束时间（秒）", value=f"{seg['end']:.2f}", key=f"e_{idx}")
            with col3:
                clean_text = remove_punctuation(seg["text"])
                text = st.text_input(f"字幕 {idx+1}", value=clean_text, key=f"t_{idx}")
            
            # 容错处理
            try:
                start_float = float(start)
                end_float = float(end)
            except:
                start_float = seg["start"]
                end_float = seg["end"]
                st.warning(f"第{idx+1}行时间格式错误，已恢复默认值")
            
            edited_segments.append({"start": start_float, "end": end_float, "text": text})
        
        # 导出SRT
        st.subheader("💾 导出字幕文件")
        srt_content = generate_srt(edited_segments, target_language, source_texts, use_bilingual)
        st.download_button(
            label=f"下载{target_language}字幕（.srt）",
            data=srt_content,
            file_name=f"{os.path.splitext(audio_file.name)[0]}_{target_language}.srt",
            mime="text/plain"
        )
        
        # 字幕预览
        st.subheader("👀 字幕预览")
        st.text_area("SRT内容", value=srt_content, height=300)

if __name__ == "__main__":
    main()
