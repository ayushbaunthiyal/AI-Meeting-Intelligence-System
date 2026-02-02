"""
Streamlit UI for AI Meeting Intelligence System

This is the main entry point for the Streamlit application.
"""

import os
import sys

import httpx
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AI Meeting Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Backend API URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")
API_BASE = f"{BACKEND_URL}/api/v1"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #F9FAFB;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #E5E7EB;
    }
    .decision-card {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
    }
    .action-card {
        background-color: #F0FDF4;
        border-left: 4px solid #22C55E;
    }
    .topic-tag {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "meetings" not in st.session_state:
        st.session_state.meetings = []
    if "selected_meeting" not in st.session_state:
        st.session_state.selected_meeting = None
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def fetch_meetings():
    """Fetch list of meetings from backend."""
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{API_BASE}/meetings")
            if response.status_code == 200:
                st.session_state.meetings = response.json()
    except Exception as e:
        st.error(f"Failed to fetch meetings: {e}")


def upload_transcript(title: str, transcript: str):
    """Upload a transcript to the backend."""
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{API_BASE}/transcripts/upload",
                json={"title": title, "transcript": transcript},
            )
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Upload failed: {response.text}")
                return None
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None


def upload_audio(file, title: str = None, language: str = None):
    """Upload and transcribe an audio file."""
    try:
        with httpx.Client(timeout=300.0) as client:
            files = {"file": (file.name, file.getvalue(), file.type)}
            data = {}
            if title:
                data["title"] = title
            if language:
                data["language"] = language
            
            response = client.post(
                f"{API_BASE}/audio/transcribe",
                files=files,
                data=data,
            )
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Transcription failed: {response.text}")
                return None
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None


def analyze_meeting(meeting_id: str):
    """Run analysis on a meeting."""
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{API_BASE}/meetings/{meeting_id}/analyze")
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Analysis failed: {response.text}")
                return None
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return None


def ask_question(meeting_id: str, question: str):
    """Ask a question about a meeting."""
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{API_BASE}/meetings/{meeting_id}/ask",
                json={"meeting_id": meeting_id, "question": question},
            )
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Question failed: {response.text}")
                return None
    except Exception as e:
        st.error(f"Question failed: {e}")
        return None


def render_sidebar():
    """Render the sidebar with meeting list and upload options."""
    with st.sidebar:
        st.markdown("## 📁 Meetings")
        
        # Upload section
        with st.expander("📤 Upload New", expanded=True):
            upload_type = st.radio(
                "Upload Type",
                ["Text Transcript", "Audio File"],
                horizontal=True,
            )
            
            if upload_type == "Text Transcript":
                title = st.text_input("Meeting Title", key="upload_title")
                transcript = st.text_area(
                    "Paste Transcript",
                    height=200,
                    placeholder="[00:00] Speaker: Welcome everyone...",
                )
                
                if st.button("Upload Transcript", type="primary", disabled=not (title and transcript)):
                    with st.spinner("Uploading..."):
                        result = upload_transcript(title, transcript)
                        if result:
                            st.success(f"✅ Uploaded: {result['segment_count']} segments")
                            fetch_meetings()
            
            else:  # Audio File
                title = st.text_input("Meeting Title (optional)", key="audio_title")
                audio_file = st.file_uploader(
                    "Upload Audio",
                    type=["mp3", "wav", "m4a", "ogg", "flac"],
                )
                language = st.selectbox(
                    "Language (optional)",
                    ["Auto-detect", "en", "es", "fr", "de", "zh", "ja"],
                )
                
                if st.button("Transcribe Audio", type="primary", disabled=not audio_file):
                    with st.spinner("Transcribing... ⏳ This may take a while"):
                        lang = None if language == "Auto-detect" else language
                        result = upload_audio(audio_file, title or None, lang)
                        if result:
                            st.success(f"✅ Transcribed: {len(result['segments'])} segments")
                            fetch_meetings()
        
        st.divider()
        
        # Meeting list
        if st.button("🔄 Refresh", use_container_width=True):
            fetch_meetings()
        
        for meeting in st.session_state.meetings:
            is_selected = (
                st.session_state.selected_meeting 
                and st.session_state.selected_meeting.get("id") == meeting["id"]
            )
            
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(
                    f"{'📊' if meeting.get('has_analysis') else '📝'} {meeting['title'][:25]}...",
                    key=f"meeting_{meeting['id']}",
                    type="primary" if is_selected else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.selected_meeting = meeting
                    st.session_state.analysis_result = None
                    st.session_state.chat_history = []
                    st.rerun()


def render_main_content():
    """Render the main content area."""
    st.markdown('<h1 class="main-header">🎯 AI Meeting Intelligence</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Analyze meeting transcripts with AI-powered insights</p>',
        unsafe_allow_html=True,
    )
    
    if not st.session_state.selected_meeting:
        # Welcome screen
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📤 Upload
            Upload text transcripts or audio files to get started.
            """)
        
        with col2:
            st.markdown("""
            ### 🔍 Analyze
            AI extracts summaries, decisions, and action items.
            """)
        
        with col3:
            st.markdown("""
            ### 💬 Ask
            Ask questions about your meetings naturally.
            """)
        
        st.info("👈 Upload a transcript or select a meeting from the sidebar to begin")
        return
    
    meeting = st.session_state.selected_meeting
    
    # Meeting header
    st.markdown(f"## {meeting['title']}")
    st.caption(f"ID: {meeting['id']} | Participants: {len(meeting.get('participants', []))}")
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔍 Analyze Meeting", type="primary", use_container_width=True):
            with st.spinner("Analyzing meeting... ⏳"):
                result = analyze_meeting(meeting["id"])
                if result:
                    st.session_state.analysis_result = result
                    st.success("✅ Analysis complete!")
    
    with col2:
        if st.button("🗑️ Delete Meeting", type="secondary", use_container_width=True):
            try:
                with httpx.Client(timeout=30.0) as client:
                    client.delete(f"{API_BASE}/meetings/{meeting['id']}")
                st.session_state.selected_meeting = None
                st.session_state.analysis_result = None
                fetch_meetings()
                st.rerun()
            except Exception as e:
                st.error(f"Failed to delete: {e}")
    
    st.divider()
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Analysis", "💬 Q&A Chat", "📄 Transcript"])
    
    with tab1:
        render_analysis_tab()
    
    with tab2:
        render_chat_tab()
    
    with tab3:
        render_transcript_tab()


def render_analysis_tab():
    """Render the analysis results tab."""
    result = st.session_state.analysis_result
    
    if not result:
        st.info("Click 'Analyze Meeting' to generate insights")
        return
    
    summary = result.get("summary", {})
    
    # Summary section
    st.markdown("### 📝 Summary")
    st.markdown(summary.get("overview", "No summary available"))
    
    # Key Topics
    topics = summary.get("key_topics", [])
    if topics:
        st.markdown("### 🏷️ Key Topics")
        topic_html = " ".join([f'<span class="topic-tag">{t}</span>' for t in topics])
        st.markdown(f'<div>{topic_html}</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Decisions and Action Items in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚖️ Decisions")
        decisions = summary.get("decisions", [])
        if decisions:
            for d in decisions:
                with st.container():
                    st.markdown(f"""
                    <div class="card decision-card">
                        <strong>{d.get('decision', 'Unknown decision')}</strong>
                        <br><small>Made by: {d.get('made_by', 'Not specified')}</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.caption("No decisions extracted")
    
    with col2:
        st.markdown("### ✅ Action Items")
        actions = summary.get("action_items", [])
        if actions:
            for a in actions:
                priority = a.get("priority", "medium")
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
                
                with st.container():
                    st.markdown(f"""
                    <div class="card action-card">
                        {priority_emoji} <strong>{a.get('task', 'Unknown task')}</strong>
                        <br><small>Owner: {a.get('owner', 'Unassigned')} | 
                        Deadline: {a.get('deadline', 'Not set')}</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.caption("No action items extracted")


def render_chat_tab():
    """Render the Q&A chat tab."""
    meeting = st.session_state.selected_meeting
    
    st.markdown("### 💬 Ask Questions About the Meeting")
    
    # Chat history
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            st.chat_message("user").markdown(content)
        else:
            st.chat_message("assistant").markdown(content)
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for i, source in enumerate(msg["sources"], 1):
                        st.caption(f"{i}. {source[:200]}...")
    
    # Chat input
    question = st.chat_input("Ask a question about this meeting...")
    
    if question:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.chat_message("user").markdown(question)
        
        # Get answer
        with st.spinner("Thinking..."):
            result = ask_question(meeting["id"], question)
        
        if result:
            answer = result["answer"]
            sources = result.get("sources", [])
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
            
            st.chat_message("assistant").markdown(answer)
            if sources:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(sources, 1):
                        st.caption(f"{i}. {source[:200]}...")
        
        st.rerun()


def render_transcript_tab():
    """Render the transcript view tab."""
    meeting = st.session_state.selected_meeting
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{API_BASE}/meetings/{meeting['id']}")
            if response.status_code == 200:
                data = response.json()
                transcript = data.get("raw_transcript", "No transcript available")
                
                st.markdown("### 📄 Full Transcript")
                st.text_area(
                    "Transcript",
                    transcript,
                    height=500,
                    disabled=True,
                    label_visibility="collapsed",
                )
    except Exception as e:
        st.error(f"Failed to load transcript: {e}")


def main():
    """Main application entry point."""
    init_session_state()
    
    # Initial fetch
    if not st.session_state.meetings:
        fetch_meetings()
    
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
