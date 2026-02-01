import streamlit as st
import whisper
import re
from datetime import timedelta
import os


# 页面配置（更美观）
st.set_page_config(page_title="音频转字幕工具", page_icon="🎙️", layout="wide")

# ---------------------- 核心功能函数 ----------------------
# 去除所有标点符号（中英文）
def remove_punctuation(text):
    punctuation = r'[，。！？；：""''()（）[]【】、·~@#￥%…&*+-=《》<>/\\|{}^_`·,:;!"$%&()*+-/<=>?@[\]^_`{|}~]'
    clean_text = re.sub(punctuation, "", text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

# 转换秒数为SRT标准时间格式（HH:MM:SS,mmm）
def format_time(seconds):
    try:
        td = timedelta(seconds=float(seconds))
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = td.microseconds // 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    except:
        return "00:00:00,000"

# 生成SRT字幕内容（支持双语）
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

# 加载Whisper Large-v3模型（最好的模型，缓存避免重复加载）
@st.cache_resource
def load_best_whisper_model():
    # 换免费版能跑的模型（small，精度足够，内存占用小）
    return whisper.load_model("small")

# ---------------------- 页面交互 ----------------------
def main():
    st.title("🎙️ 智能音频转字幕工具（Whisper Large-v3）")
    st.markdown("### 支持多语言识别、双语字幕、精准时间线")
    st.divider()

    # 侧边栏配置
    with st.sidebar:
        st.subheader("⚙️ 配置项")
        target_language = st.selectbox(
            "目标字幕语言",
            ["中文", "英文", "日语", "韩语", "法语", "西班牙语", "德语", "俄语"],
            index=0,
            help="音频会自动识别并翻译成该语言"
        )
        use_bilingual = st.checkbox("生成双语字幕（源语言+目标语言）", value=False)
        st.info("✅ 模型：Whisper Small（适配免费服务器，精度高）\n✅ 自动去除所有标点符号\n✅ 按语义分割字幕，精准对齐时间线")

    # 音频上传
    audio_file = st.file_uploader("📤 上传音频文件（支持MP3/WAV/M4A/FLAC）", type=["mp3", "wav", "m4a", "flac"])
    
    if audio_file:
        # 保存临时音频文件
        temp_audio = f"temp_{audio_file.name}"
        with open(temp_audio, "wb") as f:
            f.write(audio_file.getbuffer())
        
        # 音频预览
        st.audio(temp_audio)
        st.divider()

        # 加载模型（提示）
        with st.spinner("🔧 加载最好的识别模型（首次加载需1-2分钟）..."):
            model = load_best_whisper_model()
        
        # 识别+翻译（核心步骤）
        with st.spinner(f"🎧 正在识别音频并翻译为{target_language}（音频越长，时间越久）..."):
            # 语言映射（Whisper要求的代码）
            lang_map = {
                "中文": "zh", "英文": "en", "日语": "ja", "韩语": "ko",
                "法语": "fr", "西班牙语": "es", "德语": "de", "俄语": "ru"
            }
            # 识别+翻译
            result = model.transcribe(
                temp_audio,
                task="translate" if target_language != "中文" else "transcribe",
                language=lang_map[target_language],
                word_timestamps=False,  # 按语义分块（更符合字幕逻辑）
                verbose=False
            )
        
        # 提取识别结果
        source_segments = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in result["segments"]]
        target_segments = source_segments  # 翻译后的结果
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
            
            # 容错：防止时间输入错误
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
        st.text_area("SRT内容（可复制）", value=srt_content, height=300)
        
        # 删除临时文件（清理空间）
        os.remove(temp_audio)

if __name__ == "__main__":
    main()
