import streamlit as st
import whisper
import re
from datetime import timedelta
import os
import tempfile
from opencc import OpenCC  # 简繁转换，需额外安装

# 页面配置
st.set_page_config(page_title="高精度音频转字幕工具", page_icon="🎙️", layout="wide")

# 简繁转换初始化
@st.cache_resource
def load_converters():
    try:
        # 简→繁  繁→简
        t2s = OpenCC('t2s')
        s2t = OpenCC('s2t')
        return t2s, s2t
    except:
        return None, None

t2s_conv, s2t_conv = load_converters()

# ---------------------- 工具函数 ----------------------
# 中英文标点清洗与规范化
def remove_punctuation(text):
    # 修复转义序列，使用三重单引号避免语法错误
    punctuation = r'''[，。！？；：""''()（）[]【】、·~@#￥%…&*+-=《》<>/\\|{}^_`·,:;!"$%&()*+-/<=>?@[\]^_`{|}~]'''
    clean_text = re.sub(punctuation, "", text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

# 秒数转SRT标准时间格式
def format_time(seconds):
    try:
        td = timedelta(seconds=float(seconds))
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = td.microseconds // 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    except:
        return "00:00:00,000"

# 简繁统一转换
def convert_zh_text(text, target_type):
    if not t2s_conv or not s2t_conv:
        return text
    if target_type == "简体中文":
        return t2s_conv.convert(text)
    elif target_type == "繁体中文":
        return s2t_conv.convert(text)
    return text

# 生成SRT（支持简繁、双语、格式规范）
def generate_srt(segments, target_lang, source_texts=None, use_bilingual=False):
    srt_content = ""
    for idx, seg in enumerate(segments, 1):
        start = format_time(seg["start"])
        end = format_time(seg["end"])
        target_text = seg["text"].strip()
        
        # 简繁转换
        if target_lang in ["简体中文", "繁体中文"]:
            target_text = convert_zh_text(target_text, target_lang)
        
        target_text = remove_punctuation(target_text)
        
        srt_content += f"{idx}\n{start} --> {end}\n"
        if use_bilingual and source_texts:
            source_text = source_texts[idx-1].strip()
            source_text = remove_punctuation(source_text)
            srt_content += f"{source_text}\n{target_text}\n\n"
        else:
            srt_content += f"{target_text}\n\n"
    return srt_content

# ---------------------- 模型加载（高精度版本） ----------------------
@st.cache_resource
def load_whisper_model(model_size="medium"):
    """
    模型选择（精度从低到高）：
    tiny / base / small / medium / large-v3
    推荐：medium（平衡精度速度），large-v3（最高中文精度）
    """
    try:
        # 优先使用GPU，无GPU则用CPU
        return whisper.load_model(model_size, device="cuda" if whisper.cuda.is_available() else "cpu")
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None

# ---------------------- 主界面 ----------------------
def main():
    st.title("🎙️ 高精度音频转字幕工具（简繁分离版）")
    st.markdown("### 优化中文识别率 | 简体/繁体独立选项 | 双语SRT导出 | 降噪预处理")
    st.divider()

    # 侧边栏配置
    with st.sidebar:
        st.subheader("⚙️ 核心配置")
        
        # 1. 模型选择（直接影响准确率）
        model_choice = st.selectbox(
            "识别模型（越大越准越慢）",
            ["small", "medium", "large-v3"],
            index=1,
            help="medium平衡精度与速度，large-v3中文最强精度"
        )
        
        # 2. 语言分离：简体中文 / 繁体中文 / 其他语言
        lang_option = st.selectbox(
            "输出语言类型",
            ["简体中文", "繁体中文", "英文", "日语", "韩语", "法语", "西班牙语"],
            index=0
        )
        
        # 3. 双语字幕
        use_bilingual = st.checkbox("生成双语字幕（原文本+目标文本）", value=False)
        
        # 4. 抗噪增强
        enhance_noise = st.checkbox("开启音频降噪预处理", value=True)
        
        st.info("💡 建议：普通话清晰音频选medium，口音/嘈杂环境选large-v3")

    # 音频上传
    audio_file = st.file_uploader("📤 上传音频文件（MP3/WAV）", type=["mp3", "wav"])
    
    if audio_file:
        # 加载模型
        with st.spinner(f"🔧 加载 {model_choice} 模型（首次下载耗时较久）..."):
            model = load_whisper_model(model_choice)
            if not model:
                st.stop()

        # 写入临时音频文件
        suffix = os.path.splitext(audio_file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio_file.read())
            temp_audio_path = temp_file.name

        # 音频预览
        st.audio(temp_audio_path)
        st.divider()

        # 语言映射（简繁统一用zh，Whisper原生支持，输出再做简繁转换）
        lang_map = {
            "简体中文": "zh",
            "繁体中文": "zh",
            "英文": "en",
            "日语": "ja",
            "韩语": "ko",
            "法语": "fr",
            "西班牙语": "es"
        }
        lang_code = lang_map[lang_option]
        
        # 核心转录（高精度参数）
        with st.spinner("🎧 高精度识别中，请勿刷新页面..."):
            transcribe_kwargs = {
                "audio": temp_audio_path,
                "language": lang_code,
                "task": "transcribe",  # 固定为转录，不强制翻译，避免识别失真
                "verbose": False,
                "word_timestamps": False,
                "temperature": 0.0,  # 低温度更稳定，高噪声可设0.2
                "condition_on_previous_text": True,  # 上下文关联，提升语句连贯性
                "no_speech_threshold": 0.6,
                "logprob_threshold": -1.0
            }
            # 静音裁剪
            if enhance_noise:
                transcribe_kwargs["vad_filter"] = True
                transcribe_kwargs["vad_threshold"] = 0.5
            
            result = model.transcribe(**transcribe_kwargs)

        # 删除临时文件
        os.unlink(temp_audio_path)

        # 提取分段
        segments_raw = [
            {
                "start": s["start"],
                "end": s["end"],
                "text": s["text"].strip()
            } for s in result["segments"]
        ]
        source_texts = [s["text"] for s in segments_raw]

        # 字幕在线编辑
        st.subheader("✏️ 校对与编辑字幕")
        edited_segments = []
        for idx, seg in enumerate(segments_raw):
            col1, col2, col3 = st.columns([2, 2, 6])
            with col1:
                start_val = st.text_input(f"开始时间(s)", f"{seg['start']:.2f}", key=f"s_{idx}")
            with col2:
                end_val = st.text_input(f"结束时间(s)", f"{seg['end']:.2f}", key=f"e_{idx}")
            with col3:
                # 先简繁转换再展示
                disp_text = convert_zh_text(seg["text"], lang_option) if lang_option in ["简体中文", "繁体中文"] else seg["text"]
                text_val = st.text_input(f"字幕{idx+1}", disp_text.strip(), key=f"t_{idx}")

            # 时间格式容错
            try:
                start_f = float(start_val)
                end_f = float(end_val)
            except:
                start_f, end_f = seg["start"], seg["end"]
                st.warning(f"第{idx+1}行时间格式错误，已恢复默认")

            edited_segments.append({
                "start": start_f,
                "end": end_f,
                "text": text_val
            })

        # 生成与导出
        st.subheader("💾 导出SRT字幕文件")
        srt_content = generate_srt(edited_segments, lang_option, source_texts, use_bilingual)
        
        # 下载按钮
        base_name = os.path.splitext(audio_file.name)[0]
        st.download_button(
            label=f"📥 下载{lang_option}字幕(.srt)",
            data=srt_content.encode("utf-8"),
            file_name=f"{base_name}_{lang_option}.srt",
            mime="text/plain"
        )

        # 预览
        st.subheader("👀 字幕内容预览")
        st.text_area("SRT预览", srt_content, height=350)

if __name__ == "__main__":
    main()
