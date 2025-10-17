"""
Presentation Mode Controller
Manages the presentation mode state, navigation, and rendering.
"""
import streamlit as st
from typing import Optional
import sqlalchemy


# Presentation configuration
SLIDES_PER_ACT = {1: 4, 2: 3, 3: 4, 4: 3, 5: 1}
TOTAL_ACTS = 5
ACT_NAMES = {
    1: "ACT I: INTRODUCTION",
    2: "ACT II: PREPARATION", 
    3: "ACT III: ANALYSIS",
    4: "ACT IV: PREDICTION & PROPOSITION",
    5: "ACT V: ENDING"
}
def initialize_presentation_state():
    """Initialize session state for presentation mode"""
    if 'presentation_mode' not in st.session_state:
        st.session_state.presentation_mode = False
    if 'current_act' not in st.session_state:
        st.session_state.current_act = 1
    if 'current_slide' not in st.session_state:
        st.session_state.current_slide = 1
    if 'scroll_to_top' not in st.session_state:
        st.session_state.scroll_to_top = False


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
    # Special case: move from last slide of Act IV to Act V
    elif current_act == 4 and current_slide == SLIDES_PER_ACT[4]:
        st.session_state.current_act = 5
        st.session_state.current_slide = 1
    
    # Flag to trigger scroll to top
    st.session_state.scroll_to_top = True


def previous_slide():
    """Navigate to previous slide"""
    current_act = st.session_state.current_act
    current_slide = st.session_state.current_slide
    
    if current_slide > 1:
        st.session_state.current_slide -= 1
    elif current_act > 1:
        st.session_state.current_act -= 1
        st.session_state.current_slide = SLIDES_PER_ACT[st.session_state.current_act]
    
    # Flag to trigger scroll to top
    st.session_state.scroll_to_top = True


def goto_act(act_number: int):
    """Jump to specific act"""
    if 1 <= act_number <= TOTAL_ACTS:
        st.session_state.current_act = act_number
        st.session_state.current_slide = 1
        # Flag to trigger scroll to top
        st.session_state.scroll_to_top = True


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
    
    # Add CSS to style progress bar to black
    st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: black;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Compact divider
    st.markdown("<div style='margin: 10px 0 8px 0; border-top: 1px solid rgba(128, 128, 128, 0.3);'></div>", unsafe_allow_html=True)
    
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
    # Add an invisible anchor for scroll-to-top
    st.markdown('<div id="presentation-top"></div>', unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        .presentation-header {
            background: linear-gradient(90deg, #1a1a1a 0%, #2d2d2d 100%);
            padding: 12px 20px;
            border-radius: 12px;
            margin-bottom: 16px;
            color: white;
            box-shadow: 0 4px 16px rgba(26, 26, 26, 0.25);
            border: 1px solid rgba(45, 45, 45, 0.3);
            border-left: 6px solid #e74c3c;
        }
        .presentation-title {
            font-size: 1.6em;
            font-weight: 700;
            margin: 0;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        .presentation-subtitle {
            font-size: 0.9em;
            margin: 4px 0 0 0;
            opacity: 0.95;
            font-weight: 400;
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
        
        /* Main content container - SIT Learn branding */
        div[data-testid="stVerticalBlock"] > div:has(div.element-container) {
            background-color: var(--background-color, rgba(255, 255, 255, 0.98));
            padding: 25px;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(26, 26, 26, 0.08);
            border: 2px solid rgba(231, 76, 60, 0.1);
            border-left: 6px solid #e74c3c;
        }
        
        /* Dark theme adaptations */
        @media (prefers-color-scheme: dark) {
            div[data-testid="stVerticalBlock"] > div:has(div.element-container) {
                background-color: rgba(26, 26, 26, 0.05);
                border-color: rgba(231, 76, 60, 0.25);
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            }
        }
        
        /* SIT Learn typography - authentic colors */
        .stMarkdown h1 {
            color: #1a1a1a;
            margin-top: 0;
            margin-bottom: 0.3em;
            font-size: 2.1em;
            font-weight: 700;
            border-bottom: 3px solid #e74c3c;
            padding-bottom: 8px;
        }
        .stMarkdown h2 {
            color: #2d2d2d;
            margin-top: 0.5em;
            margin-bottom: 0.3em;
            font-size: 1.5em;
            font-weight: 600;
        }
        .stMarkdown h3 {
            color: #1a1a1a;
            margin-top: 0.4em;
            margin-bottom: 0.2em;
            font-size: 1.2em;
            font-weight: 500;
        }
        .stMarkdown h4 {
            color: #2d2d2d;
            margin-top: 0.3em;
            margin-bottom: 0.2em;
            font-size: 1.05em;
            font-weight: 500;
        }
        .stMarkdown h5 {
            color: #e74c3c;
            margin-top: 0.2em;
            margin-bottom: 0.2em;
            font-size: 1.0em;
            font-weight: 600;
        }
        
        /* Dark theme typography adaptations */
        @media (prefers-color-scheme: dark) {
            .stMarkdown h1 {
                color: #f8f9fa;
                border-bottom-color: #ff6b5b;
            }
            .stMarkdown h2 {
                color: #e9ecef;
            }
            .stMarkdown h3 {
                color: #f8f9fa;
            }
            .stMarkdown h4 {
                color: #e9ecef;
            }
            .stMarkdown h5 {
                color: #ff6b5b;
            }
        }
        
        /* Compact spacing for better vertical usage */
        .stMarkdown p {
            margin-bottom: 0.3em;
        }
        .stMarkdown ul, .stMarkdown ol {
            margin-top: 0.2em;
            margin-bottom: 0.3em;
        }
        
        /* Compact horizontal rules */
        .stMarkdown hr {
            margin: 0.8em 0;
            border: none;
            border-top: 2px solid rgba(231, 76, 60, 0.2);
        }
        
        /* Compact info boxes */
        div[data-testid="stAlert"] {
            padding: 8px 12px;
            margin: 8px 0;
            border-radius: 6px;
        }
        
        /* Compact code blocks */
        .stCodeBlock {
            margin: 8px 0;
        }
        pre {
            font-size: 0.85em;
            padding: 8px;
        }
        
        /* SIT Learn authentic metrics */
        div[data-testid="stMetricValue"] {
            font-size: 2.1em;
            color: #1a1a1a;
            font-weight: 700;
        }
        div[data-testid="stMetricLabel"] {
            color: #2d2d2d;
            font-weight: 600;
            font-size: 1.1em;
        }
        div[data-testid="stMetricDelta"] {
            font-weight: 500;
        }
        
        /* Dark theme metrics */
        @media (prefers-color-scheme: dark) {
            div[data-testid="stMetricValue"] {
                color: #f8f9fa;
            }
            div[data-testid="stMetricLabel"] {
                color: #e9ecef;
            }
        }
        
        /* SIT Learn branded dataframes */
        div[data-testid="stDataFrame"] {
            font-size: 0.95em;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 3px 12px rgba(26, 26, 26, 0.08);
            border: 1px solid rgba(231, 76, 60, 0.15);
        }
        
        /* Dark theme dataframes */
        @media (prefers-color-scheme: dark) {
            div[data-testid="stDataFrame"] {
                box-shadow: 0 3px 12px rgba(0, 0, 0, 0.4);
                border-color: rgba(231, 76, 60, 0.3);
            }
        }
        
        /* SIT Learn info boxes */
        div[data-testid="stAlert"] {
            padding: 18px;
            margin: 18px 0;
            border-radius: 12px;
            border-left: 5px solid #e74c3c;
            background-color: rgba(231, 76, 60, 0.05);
        }
        
        /* Dark theme info boxes */
        @media (prefers-color-scheme: dark) {
            div[data-testid="stAlert"] {
                background-color: rgba(231, 76, 60, 0.1);
                border-left-color: #ff6b5b;
            }
        }
        
        /* SIT Learn containers */
        div[data-testid="stContainer"] {
            border: 1px solid rgba(231, 76, 60, 0.2);
            border-radius: 12px;
            padding: 22px;
            background-color: rgba(231, 76, 60, 0.02);
        }
        
        /* Dark theme containers */
        @media (prefers-color-scheme: dark) {
            div[data-testid="stContainer"] {
                background-color: rgba(231, 76, 60, 0.05);
                border-color: rgba(231, 76, 60, 0.25);
            }
        }
        
        /* SIT Learn accent elements */
        .stMarkdown strong {
            color: #e74c3c;
            font-weight: 600;
        }
        
        @media (prefers-color-scheme: dark) {
            .stMarkdown strong {
                color: #ff6b5b;
            }
        }
        
        /* Enhanced bullet points */
        .stMarkdown li {
            margin-bottom: 0.4em;
            color: inherit;
        }
        
        /* SIT Learn code blocks */
        .stCodeBlock {
            margin: 18px 0;
            border-left: 4px solid #e74c3c;
        }
        
        @media (prefers-color-scheme: dark) {
            .stCodeBlock {
                border-left-color: #ff6b5b;
            }
        }
        
        pre {
            font-size: 0.9em;
            padding: 16px;
            background-color: rgba(26, 26, 26, 0.03);
            border-radius: 10px;
        }
        
        @media (prefers-color-scheme: dark) {
            pre {
                background-color: rgba(26, 26, 26, 0.2);
            }
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Render header
    render_presentation_header()
    
    # Render current slide
    current_act = st.session_state.current_act
    current_slide = st.session_state.current_slide
    
    # Add scroll trigger before slide content
    if st.session_state.get('scroll_to_top', False):
        # Use HTML with autofocus to force scroll
        st.markdown(
            '<input type="text" id="scroll-trigger" style="position:absolute;top:-1000px;left:-1000px;" autofocus>',
            unsafe_allow_html=True
        )
    
    # Slide content container
    with st.container():
        # Import here to avoid circular imports
        from app_pages import presentation_slides
        presentation_slides.render_slide(current_act, current_slide, engine)
    
    # Render controls
    render_presentation_controls()
    
    # Force scroll to top using multiple methods
    if st.session_state.get('scroll_to_top', False):
        st.markdown("""
            <script>
            // Multiple scroll-to-top methods for better compatibility
            setTimeout(function() {
                // Method 1: Scroll to anchor if available
                const topAnchor = window.parent.document.getElementById('presentation-top');
                if (topAnchor) {
                    topAnchor.scrollIntoView({ behavior: 'instant', block: 'start' });
                }
                
                // Method 2: Scroll main content area
                const mainContainer = window.parent.document.querySelector('.main .block-container');
                if (mainContainer) mainContainer.scrollTop = 0;
                
                // Method 3: Scroll main section
                const mainSection = window.parent.document.querySelector('section.main');
                if (mainSection) mainSection.scrollTo(0, 0);
                
                // Method 4: Scroll entire window
                window.parent.scrollTo(0, 0);
                
                // Method 5: Scroll document body
                if (window.parent.document.body) window.parent.document.body.scrollTop = 0;
                if (window.parent.document.documentElement) window.parent.document.documentElement.scrollTop = 0;
            }, 50);
            </script>
        """, unsafe_allow_html=True)
        # Reset the flag
        st.session_state.scroll_to_top = False


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
