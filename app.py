import streamlit as st
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
import tempfile, base64, os

# ------------------ CONFIG ------------------
st.set_page_config(page_title="AI Voice Chat Demo", page_icon="🧠", layout="centered")

# Load API key from secrets.toml
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ Please add your OPENAI_API_KEY in .streamlit/secrets.toml")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ------------------ SESSION STATE ------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a concise, friendly English AI assistant. Always reply in under 2 short lines."},
        {"role": "assistant", "content": "Hi 👋 I’m your AI voice assistant. How are you today?"}
    ]

# ------------------ FUNCTIONS ------------------
def transcribe_audio(audio_bytes: bytes) -> str:
    """Speech → Text using Whisper"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            with open(tmp.name, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
        os.remove(tmp.name)
        return transcript.text.strip()
    except Exception as e:
        st.error(f"Whisper error: {e}")
        return ""


def generate_response(user_text: str) -> str:
    """Text → Concise AI reply (fast)"""
    try:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        history.append({"role": "user", "content": user_text})

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            temperature=0.7
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Chat error: {e}")
        return "Sorry, something went wrong."


def speak_text(text: str) -> bytes:
    """Text → Speech using OpenAI TTS (fast)"""
    try:
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text
        )
        return speech.read()
    except Exception as e:
        st.error(f"TTS error: {e}")
        return b""


def render_audio(audio_bytes: bytes):
    """Display audio player inline"""
    if not audio_bytes:
        return
    audio_b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(
        f"""
        <audio autoplay controls style="width:100%; margin-top: 10px;">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True,
    )


# ------------------ UI STYLING ------------------
st.markdown("""
<style>
.stApp {background-color: #f9fafc;}
.stChatMessage {font-size: 1rem; line-height: 1.5;}
div[data-testid="stChatInput"] textarea {border-radius: 10px;}
audio {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 AI Voice Chat Demo")
st.caption("🎤 Speak or type to chat — powered by Whisper + GPT-4o + OpenAI TTS")

# ------------------ DISPLAY CHAT ------------------
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

st.divider()

# ------------------ INPUT AREA ------------------
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.chat_input("Type your message here...")
with col2:
    audio = mic_recorder(start_prompt="🎙️ Speak", stop_prompt="⏹️ Stop", just_once=True)

# ------------------ HANDLE USER INPUT ------------------
if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.spinner("🤖 Thinking..."):
        ai_response = generate_response(user_input)

    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)
        audio_bytes = speak_text(ai_response)
        render_audio(audio_bytes)


elif audio and audio["bytes"]:
    # Handle voice input
    with st.spinner("🎧 Transcribing..."):
        user_text = transcribe_audio(audio["bytes"])

    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        with st.spinner("🤖 Thinking..."):
            ai_response = generate_response(user_text)

        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        with st.chat_message("assistant"):
            st.markdown(ai_response)
            audio_bytes = speak_text(ai_response)
            render_audio(audio_bytes)




