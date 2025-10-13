# Presentation Mode Implementation

## Overview
The presentation mode transforms the SIT-DA-G6 labour force analytics application into a slide-based stakeholder presentation with 16 slides across 4 acts.

## Architecture

### File Structure
```
app_pages/
├── presentation_mode.py      # Main controller & navigation
├── presentation_slides.py    # Individual slide renderers (Act I & II implemented)
└── ...

streamlit_app.py              # Main app with mode toggle integration
Presentation_Mode_Architecture.md  # Complete slide content specification
```

### Components

#### 1. **presentation_mode.py** - Controller
- `initialize_presentation_state()`: Sets up session state variables
- `toggle_presentation_mode()`: Switches between report/presentation modes
- `next_slide()` / `previous_slide()`: Navigation logic
- `render_presentation_controls()`: Navigation UI (progress bar, buttons, act selector)
- `render_presentation_mode()`: Main presentation renderer with styling

#### 2. **presentation_slides.py** - Slide Renderers
- Individual slide functions for each of the 16 slides
- `render_slide(act, slide, engine)`: Routes to appropriate slide renderer
- Currently implemented: Act I (4 slides) & Act II (4 slides)

## Story Arc (4 Acts × 4 Slides = 16 Total)

### **ACT I: INTRODUCTION** ✅ Implemented
1. Project Opening & Context
2. Power BI Dashboard Preview
3. Research Framework
4. Analytic Strategy

### **ACT II: PREPARATION** ✅ Implemented
1. Data Sourcing & Architecture
2. Data Quality & Preliminary Analysis (with SQL period-based analysis)
3. Data Pipeline & ETL Architecture
4. Analytics-Ready Master Dataset

### **ACT III: ANALYSIS** 🚧 To Be Implemented
1. Trend Lens — Persistent Pressure & COVID Impact
2. Human Capital Lens — Demographics Mediate Risk
3. Comparative Lens — PMET Resilience
4. Analysis Summary — Synthesis & Bridge to Prediction

### **ACT IV: PREDICTION & PROPOSITION** 🚧 To Be Implemented
1. Predictive Modeling — From Patterns to Forecasts
2. 2025 Forecasts — The Verdict (KNN + Logistic convergence)
3. The 12-Month Window — Act Now or Pay Later
4. Strategic Recommendations — The Action Plan

## Usage

### Accessing Presentation Mode
1. Run the Streamlit app: `streamlit run streamlit_app.py`
2. Click **"🎤 Switch to Presentation Mode"** button in the sidebar
3. Use navigation controls to move through slides

### Navigation Controls
- **⬅️ Previous**: Go to previous slide
- **Next ➡️**: Go to next slide
- **Jump to Act**: Dropdown to skip to specific act
- **🚪 Exit**: Return to report mode
- **Progress Bar**: Visual indicator of presentation progress

### Session State Variables
```python
st.session_state.presentation_mode  # Boolean: True = presentation, False = report
st.session_state.current_act        # Integer: 1-4 (current act number)
st.session_state.current_slide      # Integer: 1-4 (slide within current act)
```

## Styling Features

### Presentation Mode Styling
- Dark background (#0e1117) with white slide cards
- Gradient blue header
- Enhanced readability with hierarchical color scheme:
  - H1: #1e3a8a (dark blue)
  - H2: #3b82f6 (medium blue)
  - H3: #60a5fa (light blue)
- Larger metrics (2em font size)
- Rounded corners and shadows for depth

### Responsive Design
- Horizontal navigation (previous/next)
- Progress tracking (slide X of Y, overall progress bar)
- Act-based quick navigation

## Next Steps for Completion

### Act III Implementation (Analysis)
```python
# In presentation_slides.py, add:
def slide_3_1_trend_lens():
    # Which occupations show persistent pressure?
    # COVID-19 structural break analysis
    
def slide_3_2_human_capital_lens():
    # Demographics mediate risk within occupations
    # Age × Education × Occupation heatmaps
    
def slide_3_3_comparative_lens():
    # PMET vs Non-PMET resilience comparison
    # Correlation analysis
    
def slide_3_4_analysis_summary():
    # Synthesis of three lenses
    # Bridge to prediction
```

### Act IV Implementation (Prediction)
```python
# In presentation_slides.py, add:
def slide_4_1_predictive_modeling():
    # KNN + Logistic approach
    # Feature engineering overview
    
def slide_4_2_forecasts_2025():
    # Model convergence on top 3 groups
    # 99%+ probability findings
    
def slide_4_3_twelve_month_window():
    # Intervention timing economics
    # Cost of delay analysis
    
def slide_4_4_recommendations():
    # 4 strategic priorities
    # ROI calculation
    # Call to action
```

### Integration with Existing Pages
Consider extracting visualizations and content from:
- `cleaning_eda.py` → For Act III analysis slides
- `module_4_machine_learning.py` → For Act IV prediction slides

## Design Decisions

### Why 4 Acts × 4 Slides?
- **Balance**: Each act tells a complete story
- **Digestibility**: 4 slides per act prevent cognitive overload
- **Symmetry**: Clean 16-slide structure for 45-60 minute presentations

### Why Separate from Report Mode?
- **Different audiences**: Report mode for analysts, presentation for stakeholders
- **Different narratives**: Report is exploratory, presentation is persuasive
- **Flexibility**: Toggle preserves both modes without duplication

### Content Alignment
All slide content is extracted from actual analysis in:
- Module 1 (SQL analysis) → Act II slides
- Module 2 & 3 (EDA) → Act III slides
- Module 4 (ML) → Act IV slides
- Strategic brief from overview → Act I slides

## Configuration

### Power BI Integration
Set in `.streamlit/secrets.toml`:
```toml
POWERBI_EMBED_URL = "https://app.powerbi.com/view?r=..."
```

### Database Connection
```toml
DB_CONNECTION_STRING = "sqlite:///path/to/database.db"
```

## Known Limitations
- Acts III & IV not yet implemented (slide stubs return "not yet implemented")
- Power BI embed requires external URL configuration
- No slide transitions/animations (future enhancement)
- No speaker notes view (future enhancement)

## Future Enhancements
1. **Slide Transitions**: Add fade/slide animations between slides
2. **Speaker Notes**: Separate view with narrative text for presenter
3. **PDF Export**: Generate PDF from presentation slides
4. **Slide Thumbnails**: Mini-map navigation view
5. **Timing Mode**: Auto-advance slides for rehearsal
6. **Annotation Layer**: Real-time drawing on slides during presentation

## Testing
```bash
# Run the app
streamlit run streamlit_app.py

# Navigate to any page in report mode
# Click "Switch to Presentation Mode"
# Verify:
# - Slide 1.1 renders correctly
# - Navigation buttons work
# - Progress bar updates
# - Act selector jumps correctly
# - Exit button returns to report mode
```

## Credits
**Author**: SIT Data Analytics Group 6  
**Project**: Labour Force Unemployment Insights (2014-2024)  
**Date**: October 2025
