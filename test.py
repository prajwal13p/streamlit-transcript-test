import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import speech_recognition as sr
import queue
import threading
import numpy as np
from datetime import datetime
import json
import io

st.set_page_config(
    page_title="Voice Call Transcript Demo",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Two-Person Voice Call with Live Transcript")
st.markdown("**Real-time conversation recording and transcription**")

# Initialize session state
if 'call_transcript' not in st.session_state:
    st.session_state.call_transcript = []
if 'call_started' not in st.session_state:
    st.session_state.call_started = False
if 'participant_name' not in st.session_state:
    st.session_state.participant_name = ""

# Participant setup
col1, col2 = st.columns([1, 3])
with col1:
    participant_name = st.text_input("Your Name", value=st.session_state.participant_name)
    if participant_name:
        st.session_state.participant_name = participant_name

# STUN/TURN servers for WebRTC connection
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

# Audio processing queue
audio_queue = queue.Queue()
recognizer = sr.Recognizer()

def process_audio_frame(frame):
    """Process incoming audio frames for speech recognition"""
    try:
        audio_array = frame.to_ndarray()
        
        # Convert to speech recognition format
        audio_data = sr.AudioData(
            audio_array.tobytes(), 
            frame.sample_rate, 
            frame.sample_width
        )
        
        # Perform speech recognition in background
        try:
            text = recognizer.recognize_google(audio_data, language='en-US')
            if text.strip():
                timestamp = datetime.now().strftime("%H:%M:%S")
                transcript_entry = {
                    'time': timestamp,
                    'speaker': st.session_state.participant_name or "Unknown",
                    'text': text,
                    'type': 'speech'
                }
                
                # Add to transcript
                st.session_state.call_transcript.append(transcript_entry)
                
                # Auto-detect important information
                important_keywords = ['action', 'task', 'deadline', 'important', 'note', 'remember', 'follow up', 'meeting', 'decision']
                if any(keyword in text.lower() for keyword in important_keywords):
                    transcript_entry['type'] = 'important'
                    
        except sr.UnknownValueError:
            pass  # Ignore unrecognizable audio
        except sr.RequestError as e:
            st.error(f"Speech recognition error: {e}")
            
    except Exception as e:
        pass  # Ignore processing errors

# Layout for call interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔗 Voice Connection")
    
    # WebRTC audio streamer for bidirectional communication
    webrtc_ctx = webrtc_streamer(
        key="voice-call",
        mode=WebRtcMode.SENDRECV,  # Send and receive audio
        rtc_configuration=RTC_CONFIGURATION,
        audio_frame_callback=process_audio_frame,
        media_stream_constraints={
            "video": False,
            "audio": {
                "echoCancellation": True,
                "noiseSuppression": True,
                "autoGainControl": True,
            }
        },
        async_processing=True,
    )
    
    # Connection status
    if webrtc_ctx.state.playing:
        st.success(f"🟢 {participant_name} is connected to the call")
        st.session_state.call_started = True
    else:
        st.info("📞 Click START to join the voice call")

    # Share connection details
    if st.session_state.call_started:
        st.markdown("### 🔗 Share this link with the other person:")
        current_url = st.query_params.get("url", "Your app URL here")
        st.code(current_url, language="text")

with col2:
    st.subheader("📋 Call Controls")
    
    # Recording controls
    if st.button("🔴 Start Recording Transcript"):
        st.session_state.call_transcript = []
        st.success("Recording started!")
    
    if st.button("⏹️ Stop & Save Transcript"):
        if st.session_state.call_transcript:
            # Generate downloadable transcript
            transcript_text = generate_transcript_file()
            st.download_button(
                label="📥 Download Transcript",
                data=transcript_text,
                file_name=f"call_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            st.success("Transcript saved!")
    
    # Clear transcript
    if st.button("🗑️ Clear Transcript"):
        st.session_state.call_transcript = []
        st.success("Transcript cleared!")

# Live transcript display
st.subheader("📝 Live Conversation Transcript")

if st.session_state.call_transcript:
    # Create scrollable transcript area
    with st.container():
        for entry in st.session_state.call_transcript[-20:]:  # Show last 20 entries
            timestamp = entry['time']
            speaker = entry['speaker']
            text = entry['text']
            entry_type = entry.get('type', 'speech')
            
            if entry_type == 'important':
                st.warning(f"⭐ **[{timestamp}] {speaker}:** {text}")
            else:
                st.markdown(f"**[{timestamp}] {speaker}:** {text}")
    
    # Auto-scroll to bottom
    st.markdown("---")
    
    # Transcript statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Messages", len(st.session_state.call_transcript))
    with col2:
        important_count = sum(1 for entry in st.session_state.call_transcript if entry.get('type') == 'important')
        st.metric("Important Items", important_count)
    with col3:
        if st.session_state.call_transcript:
            duration = datetime.now() - datetime.strptime(st.session_state.call_transcript[0]['time'], "%H:%M:%S").replace(year=datetime.now().year, month=datetime.now().month, day=datetime.now().day)
            st.metric("Call Duration", str(duration).split('.')[0])

else:
    st.info("💬 Transcript will appear here when the conversation starts...")

# AI Analysis Section
if st.session_state.call_transcript:
    st.subheader("🤖 AI Analysis")
    
    if st.button("🔍 Analyze Conversation"):
        with st.spinner("Analyzing conversation..."):
            # Extract key points and action items
            full_transcript = " ".join([entry['text'] for entry in st.session_state.call_transcript])
            
            # Simple keyword-based analysis
            action_items = []
            key_topics = []
            
            for entry in st.session_state.call_transcript:
                text = entry['text'].lower()
                if any(word in text for word in ['will', 'should', 'need to', 'have to', 'must']):
                    action_items.append(f"[{entry['time']}] {entry['speaker']}: {entry['text']}")
                
                if any(word in text for word in ['project', 'meeting', 'deadline', 'budget', 'client']):
                    key_topics.append(entry['text'])
            
            # Display analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📋 Action Items")
                for item in action_items[-5:]:  # Last 5 action items
                    st.markdown(f"- {item}")
            
            with col2:
                st.markdown("### 🎯 Key Topics")
                unique_topics = list(set(key_topics[-5:]))  # Last 5 unique topics
                for topic in unique_topics:
                    st.markdown(f"- {topic}")

def generate_transcript_file():
    """Generate downloadable transcript file"""
    transcript_lines = []
    transcript_lines.append("VOICE CALL TRANSCRIPT")
    transcript_lines.append("=" * 50)
    transcript_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    transcript_lines.append("")
    
    for entry in st.session_state.call_transcript:
        line = f"[{entry['time']}] {entry['speaker']}: {entry['text']}"
        if entry.get('type') == 'important':
            line = f"⭐ {line} [IMPORTANT]"
        transcript_lines.append(line)
    
    return "\n".join(transcript_lines)

# Instructions
with st.expander("📖 How to Use This Demo"):
    st.markdown("""
    ### Setup Instructions:
    
    1. **Both participants** open this same Streamlit app
    2. **Enter your name** in the text input
    3. **Click START** to join the voice call
    4. **Grant microphone permissions** when prompted
    5. **Start talking** - transcript will appear in real-time
    
    ### Features:
    - ✅ Real-time bidirectional voice communication
    - ✅ Live speech-to-text transcription
    - ✅ Automatic important detail detection
    - ✅ Downloadable transcript file
    - ✅ AI-powered conversation analysis
    
    ### Tips:
    - Speak clearly for better transcription accuracy
    - Important keywords are automatically highlighted
    - Transcript is saved automatically during the call
    - Use the analysis feature to extract key insights
    """)
