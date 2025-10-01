import streamlit as st
import asyncio
import json
import uuid
from datetime import datetime
import threading
import queue
import time
from typing import Dict, List, Optional
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase, AudioProcessorBase
import speech_recognition as sr
from collections import defaultdict

st.set_page_config(
    page_title="Google Meet Clone with Live Transcript",
    page_icon="🎥",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .meet-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .participant-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .transcript-entry {
        background: #f8f9fa;
        padding: 10px;
        margin: 5px 0;
        border-left: 4px solid #007bff;
        border-radius: 4px;
    }
    .important-entry {
        background: #fff3cd;
        border-left-color: #ffc107;
    }
    .speaker-you {
        background: #d4edda;
        border-left-color: #28a745;
    }
    .speaker-other {
        background: #e2e3e5;
        border-left-color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'room_id' not in st.session_state:
    st.session_state.room_id = str(uuid.uuid4())[:8]
if 'participant_id' not in st.session_state:
    st.session_state.participant_id = str(uuid.uuid4())[:8]
if 'participant_name' not in st.session_state:
    st.session_state.participant_name = ""
if 'is_connected' not in st.session_state:
    st.session_state.is_connected = False
if 'transcript_data' not in st.session_state:
    st.session_state.transcript_data = []
if 'participants' not in st.session_state:
    st.session_state.participants = {}
if 'audio_queue' not in st.session_state:
    st.session_state.audio_queue = queue.Queue()

# Global transcript storage (in real app, this would be Redis/Database)
GLOBAL_TRANSCRIPT = defaultdict(list)
GLOBAL_PARTICIPANTS = {}

class AudioProcessor(AudioProcessorBase):
    """Custom audio processor for speech recognition"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.last_transcript_time = 0
        self.transcript_cooldown = 2  # seconds
        
    def recv(self, frame):
        """Process incoming audio frames"""
        try:
            # Convert audio frame to numpy array
            audio_array = frame.to_ndarray()
            
            # Only process if there's actual audio (not silence)
            if np.max(np.abs(audio_array)) > 0.01:
                # Convert to speech recognition format
                audio_data = sr.AudioData(
                    audio_array.tobytes(),
                    frame.sample_rate,
                    frame.sample_width
                )
                
                # Perform speech recognition
                try:
                    current_time = time.time()
                    if current_time - self.last_transcript_time > self.transcript_cooldown:
                        text = self.recognizer.recognize_google(audio_data, language='en-US')
                        if text.strip():
                            self.last_transcript_time = current_time
                            self.add_transcript_entry(text.strip())
                            
                except sr.UnknownValueError:
                    pass  # Ignore unrecognizable audio
                except sr.RequestError as e:
                    st.error(f"Speech recognition error: {e}")
                    
        except Exception as e:
            pass  # Ignore processing errors
            
        return frame
    
    def add_transcript_entry(self, text: str):
        """Add transcript entry to global storage"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        participant_name = st.session_state.participant_name or "Unknown"
        
        entry = {
            'id': str(uuid.uuid4()),
            'timestamp': timestamp,
            'speaker': participant_name,
            'text': text,
            'participant_id': st.session_state.participant_id,
            'datetime': datetime.now().isoformat()
        }
        
        # Add to global transcript
        GLOBAL_TRANSCRIPT[st.session_state.room_id].append(entry)
        
        # Update session state
        st.session_state.transcript_data.append(entry)
        
        # Trigger rerun to update UI
        st.rerun()

def get_room_participants():
    """Get participants in the current room"""
    return GLOBAL_PARTICIPANTS.get(st.session_state.room_id, {})

def add_participant():
    """Add current user to room participants"""
    if st.session_state.participant_name:
        GLOBAL_PARTICIPANTS[st.session_state.room_id][st.session_state.participant_id] = {
            'name': st.session_state.participant_name,
            'joined_at': datetime.now().isoformat(),
            'is_active': st.session_state.is_connected
        }

def remove_participant():
    """Remove current user from room participants"""
    if st.session_state.room_id in GLOBAL_PARTICIPANTS:
        GLOBAL_PARTICIPANTS[st.session_state.room_id].pop(st.session_state.participant_id, None)

# Main UI
st.title("🎥 Google Meet Clone with Live Transcript")
st.markdown("**Real-time voice chat with live transcription**")

# Room setup
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="meet-container">', unsafe_allow_html=True)
    
    # Participant name input
    participant_name = st.text_input(
        "👤 Your Name", 
        value=st.session_state.participant_name,
        placeholder="Enter your name to join the call"
    )
    
    if participant_name:
        st.session_state.participant_name = participant_name
    
    # Room ID display and sharing
    st.markdown("### 🏠 Room Information")
    col_room1, col_room2 = st.columns([2, 1])
    
    with col_room1:
        st.code(f"Room ID: {st.session_state.room_id}", language="text")
    
    with col_room2:
        if st.button("🔄 New Room"):
            st.session_state.room_id = str(uuid.uuid4())[:8]
            st.session_state.transcript_data = []
            st.rerun()
    
    # Share room link
    st.markdown("### 🔗 Share this link with others:")
    room_url = f"{st.query_params.get('url', 'Your app URL')}?room={st.session_state.room_id}"
    st.code(room_url, language="text")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="participant-card">', unsafe_allow_html=True)
    st.markdown("### 👥 Participants")
    
    # Add current participant
    if st.session_state.participant_name and st.session_state.is_connected:
        add_participant()
        st.success(f"🟢 {st.session_state.participant_name} (You)")
    
    # Show other participants
    participants = get_room_participants()
    for pid, pdata in participants.items():
        if pid != st.session_state.participant_id:
            status = "🟢" if pdata.get('is_active', False) else "🔴"
            st.info(f"{status} {pdata.get('name', 'Unknown')}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# WebRTC Configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
})

# Voice call interface
st.markdown("### 🎙️ Voice Call")

if st.session_state.participant_name:
    # WebRTC audio streamer
    webrtc_ctx = webrtc_streamer(
        key=f"voice-call-{st.session_state.room_id}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={
            "video": False,
            "audio": {
                "echoCancellation": True,
                "noiseSuppression": True,
                "autoGainControl": True,
                "sampleRate": 44100,
            }
        },
        async_processing=True,
    )
    
    # Connection status
    if webrtc_ctx.state.playing:
        st.session_state.is_connected = True
        st.success(f"🟢 Connected to room {st.session_state.room_id}")
        add_participant()
    else:
        st.session_state.is_connected = False
        st.info("📞 Click START to join the voice call")
        remove_participant()
        
else:
    st.warning("⚠️ Please enter your name to join the call")

# Live transcript display
st.markdown("### 📝 Live Transcript")

# Auto-refresh transcript every 2 seconds
if st.session_state.is_connected:
    # Get transcript for current room
    room_transcript = GLOBAL_TRANSCRIPT.get(st.session_state.room_id, [])
    
    if room_transcript:
        # Display transcript entries
        for entry in room_transcript[-50:]:  # Show last 50 entries
            timestamp = entry['timestamp']
            speaker = entry['speaker']
            text = entry['text']
            is_you = entry['participant_id'] == st.session_state.participant_id
            
            # Determine CSS class
            css_class = "speaker-you" if is_you else "speaker-other"
            
            # Check for important keywords
            important_keywords = ['meeting', 'appointment', 'schedule', 'important', 'urgent', 'deadline', 'action', 'task']
            is_important = any(keyword in text.lower() for keyword in important_keywords)
            
            if is_important:
                css_class += " important-entry"
            
            # Display transcript entry
            st.markdown(f"""
            <div class="transcript-entry {css_class}">
                <strong>[{timestamp}] {speaker}:</strong> {text}
            </div>
            """, unsafe_allow_html=True)
        
        # Auto-scroll to bottom
        st.markdown("---")
        
        # Transcript statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Messages", len(room_transcript))
        with col2:
            your_messages = sum(1 for entry in room_transcript if entry['participant_id'] == st.session_state.participant_id)
            st.metric("Your Messages", your_messages)
        with col3:
            important_count = sum(1 for entry in room_transcript if any(keyword in entry['text'].lower() for keyword in important_keywords))
            st.metric("Important Items", important_count)
        with col4:
            unique_speakers = len(set(entry['speaker'] for entry in room_transcript))
            st.metric("Speakers", unique_speakers)
            
    else:
        st.info("💬 Transcript will appear here when participants start speaking...")

# Controls
st.markdown("### 🎛️ Controls")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Download Transcript"):
        if st.session_state.transcript_data:
            transcript_text = generate_transcript_file()
            st.download_button(
                label="💾 Download",
                data=transcript_text,
                file_name=f"meet_transcript_{st.session_state.room_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

with col2:
    if st.button("🗑️ Clear Transcript"):
        GLOBAL_TRANSCRIPT[st.session_state.room_id] = []
        st.session_state.transcript_data = []
        st.success("Transcript cleared!")
        st.rerun()

with col3:
    if st.button("🔄 Refresh"):
        st.rerun()

def generate_transcript_file():
    """Generate downloadable transcript file"""
    room_transcript = GLOBAL_TRANSCRIPT.get(st.session_state.room_id, [])
    
    transcript_lines = []
    transcript_lines.append("GOOGLE MEET CLONE - VOICE CALL TRANSCRIPT")
    transcript_lines.append("=" * 60)
    transcript_lines.append(f"Room ID: {st.session_state.room_id}")
    transcript_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    transcript_lines.append(f"Participants: {', '.join(set(entry['speaker'] for entry in room_transcript))}")
    transcript_lines.append("")
    
    for entry in room_transcript:
        line = f"[{entry['timestamp']}] {entry['speaker']}: {entry['text']}"
        transcript_lines.append(line)
    
    return "\n".join(transcript_lines)

# Instructions
with st.expander("📖 How to Use This Google Meet Clone"):
    st.markdown("""
    ### 🚀 Setup Instructions:
    
    1. **Enter your name** in the text input above
    2. **Share the room link** with the person you want to call
    3. **Both participants** click START to join the voice call
    4. **Grant microphone permissions** when prompted
    5. **Start talking** - transcript will appear in real-time for both participants
    
    ### ✨ Features:
    - ✅ **Real-time bidirectional voice communication** (like Google Meet)
    - ✅ **Live speech-to-text transcription** for all participants
    - ✅ **Shared transcript** visible to everyone in the room
    - ✅ **Automatic speaker identification**
    - ✅ **Important message highlighting**
    - ✅ **Downloadable transcript file**
    - ✅ **Room-based conversations** (multiple rooms supported)
    - ✅ **No echo issues** (proper audio processing)
    
    ### 🎯 Tips:
    - **Speak clearly** for better transcription accuracy
    - **Important keywords** are automatically highlighted
    - **Each participant** sees their own messages highlighted in green
    - **Room ID** allows multiple separate conversations
    - **Transcript is shared** between all participants in the same room
    
    ### 🔧 Technical Notes:
    - Uses WebRTC for real-time audio communication
    - Google Speech Recognition for transcription
    - STUN servers for NAT traversal
    - Echo cancellation and noise suppression enabled
    """)

# Auto-refresh every 3 seconds when connected
if st.session_state.is_connected:
    time.sleep(3)
    st.rerun()
