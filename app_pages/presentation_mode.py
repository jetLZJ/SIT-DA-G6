"""
Presentation Mode Controller
Manages the presentation mode state, navigation, and rendering.
"""
import streamlit as st
from typing import Optional
import sqlalchemy
from app_pages import presentation_slides


# Presentation configuration
SLIDES_PER_ACT = {1: 4, 2: 4, 3: 4, 4: 4}
TOTAL_ACTS = 4
ACT_NAMES = {
    1: "ACT I: INTRODUCTION",
    2: "ACT II: PREPARATION",
    3: "ACT III: ANALYSIS",
    4: "ACT IV: PREDICTION & PROPOSITION"
}


def initialize_presentation_state():
    """Initialize session state for presentation mode"""
    if 'presentation_mode' not in st.session_state:
        st.session_state.presentation_mode = False
    if 'current_act' not in st.session_state:
        st.session_state.current_act = 1
    if 'current_slide' not in st.session_state:
        st.session_state.current_slide = 1


def toggle_presentation_mode():
    """Toggle between report and presentation mode"""
    st.session_state.presentation_mode = not st.session_state.presentation_mode
    # Reset to first slide when entering presentation mode
    if st.session_state.presentation_mode:
        st.session_state.current_act = 1
        st.session_state.current_slide = 1


def next_slide():
    """Navigate to next slide"""
    current_act = st.session_state.current_act
    current_slide = st.session_state.current_slide
    max_slides = SLIDES_PER_ACT[current_act]
    
    if current_slide < max_slides:
        st.session_state.current_slide += 1
    elif current_act < TOTAL_ACTS:
        st.session_state.current_act += 1
        st.session_state.current_slide = 1


def previous_slide():
    """Navigate to previous slide"""
    current_act = st.session_state.current_act
    current_slide = st.session_state.current_slide
    
    if current_slide > 1:
        st.session_state.current_slide -= 1
    elif current_act > 1:
        st.session_state.current_act -= 1
        st.session_state.current_slide = SLIDES_PER_ACT[st.session_state.current_act]


def goto_act(act_number: int):
    """Jump to specific act"""
    if 1 <= act_number <= TOTAL_ACTS:
        st.session_state.current_act = act_number
        st.session_state.current_slide = 1


def get_slide_progress() -> dict:
    """Calculate slide progress information"""
    current_act = st.session_state.current_act
    current_slide = st.session_state.current_slide
    
    # Calculate total slide number
    total_slide_number = sum(SLIDES_PER_ACT[i] for i in range(1, current_act)) + current_slide
    total_slides = sum(SLIDES_PER_ACT.values())
    
    return {
        'act': current_act,
        'slide': current_slide,
        'slide_in_act': f"{current_slide}/{SLIDES_PER_ACT[current_act]}",
        'total_slide': f"{total_slide_number}/{total_slides}",
        'progress': total_slide_number / total_slides,
        'act_name': ACT_NAMES[current_act]
    }


def render_presentation_controls():
    """Render the presentation navigation controls - compact version"""
    progress = get_slide_progress()
    
    # Compact divider
    st.markdown("<div style='margin: 15px 0 10px 0; border-top: 1px solid rgba(128, 128, 128, 0.3);'></div>", unsafe_allow_html=True)
    
    # Progress bar with label
    col_prog1, col_prog2 = st.columns([4, 1])
    with col_prog1:
        st.progress(progress['progress'])
    with col_prog2:
        st.caption(f"{progress['total_slide']}")
    
    # Control layout - more compact
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("⬅️ Prev", use_container_width=True, disabled=(progress['act'] == 1 and progress['slide'] == 1)):
            previous_slide()
            st.rerun()
    
    with col2:
        if st.button("Next ➡️", use_container_width=True, disabled=(progress['act'] == TOTAL_ACTS and progress['slide'] == SLIDES_PER_ACT[TOTAL_ACTS])):
            next_slide()
            st.rerun()
    
    with col3:
        st.markdown(f"<div style='text-align: center; padding: 4px; font-size: 0.9em;'><b>{progress['act_name']}</b><br/>Slide {progress['slide_in_act']}</div>", unsafe_allow_html=True)
    
    with col4:
        # Act selector
        selected_act = st.selectbox(
            "Jump to Act",
            options=list(range(1, TOTAL_ACTS + 1)),
            format_func=lambda x: f"Act {x}",
            index=progress['act'] - 1,
            key="act_selector",
            label_visibility="collapsed"
        )
        if selected_act != progress['act']:
            goto_act(selected_act)
            st.rerun()
    
    with col5:
        if st.button("🚪 Exit", use_container_width=True, help="Exit presentation mode"):
            toggle_presentation_mode()
            st.rerun()


def render_presentation_header():
    """Render presentation mode header - compact version"""
    st.markdown("""
        <style>
        .presentation-header {
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 12px 20px;
            border-radius: 8px;
            margin-bottom: 15px;
            color: white;
        }
        .presentation-title {
            font-size: 1.5em;
            font-weight: bold;
            margin: 0;
        }
        .presentation-subtitle {
            font-size: 0.9em;
            margin: 3px 0 0 0;
            opacity: 0.9;
        }
        </style>
        <div class="presentation-header">
            <div class="presentation-title">Singapore Labour Force Analysis</div>
            <div class="presentation-subtitle">Unemployment Insights & Predictive Analytics (2014-2024)</div>
        </div>
    """, unsafe_allow_html=True)


def render_presentation_mode(engine: Optional[sqlalchemy.engine.Engine]):
    """
    Main presentation mode renderer
    
    Args:
        engine: Database engine (optional)
    """
    # Apply presentation styling - theme-adaptive
    st.markdown("""
        <style>
        /* Hide streamlit default elements in presentation mode */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Ensure sidebar toggle button remains visible */
        button[kind="header"] {
            visibility: visible !important;
            display: block !important;
        }
        
        /* Keep sidebar collapse button visible */
        [data-testid="collapsedControl"] {
            visibility: visible !important;
            display: flex !important;
        }
        
        /* Force visibility of sidebar toggle elements */
        .css-1d391kg, .css-1rs6os, .css-17lntkn {
            visibility: visible !important;
            display: flex !important;
        }
        
        /* Streamlit sidebar toggle button */
        button[aria-label="Open sidebar"] {
            visibility: visible !important;
            display: block !important;
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            z-index: 999999 !important;
        }
        
        /* Presentation slide styling - adaptive to theme */
        .stApp {
            /* Background will adapt to user's theme preference */
        }
        
        /* Main content container - subtle styling that works with both themes */
        div[data-testid="stVerticalBlock"] > div:has(div.element-container) {
            background-color: var(--background-color, rgba(255, 255, 255, 0.05));
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(128, 128, 128, 0.2);
        }
        
        /* Improve text readability - adaptive colors */
        .stMarkdown h1 {
            color: #3b82f6;
            margin-top: 0;
            margin-bottom: 0.5em;
            font-size: 2em;
        }
        .stMarkdown h2 {
            color: #60a5fa;
            margin-top: 0.5em;
            margin-bottom: 0.5em;
            font-size: 1.5em;
        }
        .stMarkdown h3 {
            color: #93c5fd;
            margin-top: 0.4em;
            margin-bottom: 0.4em;
            font-size: 1.2em;
        }
        
        /* Compact spacing for better vertical usage */
        .stMarkdown p {
            margin-bottom: 0.5em;
        }
        .stMarkdown ul, .stMarkdown ol {
            margin-top: 0.3em;
            margin-bottom: 0.5em;
        }
        
        /* Make metrics more prominent */
        div[data-testid="stMetricValue"] {
            font-size: 1.8em;
        }
        
        /* Compact dataframes */
        div[data-testid="stDataFrame"] {
            font-size: 0.9em;
        }
        
        /* Better info boxes - theme adaptive */
        div[data-testid="stAlert"] {
            padding: 12px;
            margin: 10px 0;
            border-radius: 6px;
        }
        
        /* Code blocks - more compact */
        .stCodeBlock {
            margin: 10px 0;
        }
        pre {
            font-size: 0.85em;
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Render header
    render_presentation_header()
    
    # Render current slide
    current_act = st.session_state.current_act
    current_slide = st.session_state.current_slide
    
    # Slide content container
    with st.container():
        presentation_slides.render_slide(current_act, current_slide, engine)
    
    # Render controls
    render_presentation_controls()
    
    # Auto-scroll to top on slide navigation
    st.markdown("""
        <script>
        window.parent.document.querySelector('section.main').scrollTo(0, 0);
        </script>
    """, unsafe_allow_html=True)


def render_mode_toggle_button():
    """Render the presentation mode toggle button in sidebar with logo"""
    # Show logo and caption in sidebar when in presentation mode
    if st.session_state.get('presentation_mode', False):
        from pathlib import Path
        logo_path = Path(__file__).parent.parent / 'assets' / '4C LogoSIT Learn Lock UP logo_4C.png'
        if logo_path.exists():
            st.sidebar.image(str(logo_path), caption='Data Analytics Group 6', width='stretch')
        st.sidebar.markdown("---")
    else:
        st.sidebar.markdown("---")
    
    if st.session_state.get('presentation_mode', False):
        button_label = "📊 Switch to Report Mode"
        button_help = "Exit presentation and return to full report view"
    else:
        button_label = "🎤 Switch to Presentation Mode"
        button_help = "Enter slide-based presentation view"
    
    if st.sidebar.button(button_label, use_container_width=True, help=button_help):
        toggle_presentation_mode()
        st.rerun()
