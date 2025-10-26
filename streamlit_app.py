from pathlib import Path
from typing import Optional

import sqlalchemy
import streamlit as st

from app import data_loader
from app_pages import overview, data_schema, cleaning_eda, dashboard, module_4_machine_learning, presentation_mode


PAGE_TITLE = 'SIT-DA Capstone — Labor Force Trends'
ASSETS_DIR = Path(__file__).parent / 'assets'
MODULES_DIR = Path(__file__).parent / 'modules'
APP_LOGO_PATH = ASSETS_DIR / '4C LogoSIT Learn Lock UP logo_4C.png'
GROUP_PHOTO_PATH = ASSETS_DIR / 'MVIMG_20251011_153643_1.jpg'
APPENDIX_FILES = [
    {
        'label': 'Module 1 Appendix — Create Scripts (SQL)',
        'path': MODULES_DIR / 'm1_appendix_create.sql',
        'mime': 'text/sql',
    },
    {
        'label': 'Module 1 Appendix — Transform Scripts (SQL)',
        'path': MODULES_DIR / 'm1_appendix_transform.sql',
        'mime': 'text/sql',
    },
    {
        'label': 'Module 1 — Data Fundamentals (Deck)',
        'path': MODULES_DIR / 'M1 Data Fundamentals and SQL G6 v4.docx',
        'mime': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    },
    {
        'label': 'Module 2 & 3 — EDA and Visualisation Notebook',
        'path': MODULES_DIR / 'M2 M3 EDA and Visualisation.ipynb',
        'mime': 'application/x-ipynb+json',
    },
    {
        'label': 'Module 4 — Machine Learning Notebook',
        'path': MODULES_DIR / 'M4 Machine Learning.ipynb',
        'mime': 'application/x-ipynb+json',
    },
]


def get_db_engine() -> Optional[sqlalchemy.engine.Engine]:
    conn = st.secrets.get('DB_CONNECTION_STRING')
    if not conn:
        return None
    return data_loader.engine_from_connection_string(conn)


# pillar-focused renderers


def page_hypothesis(engine: Optional[sqlalchemy.engine.Engine]) -> None:
    """Frame the strategic hypothesis with the organised brief and problem narrative."""
    overview.page_overview()


def page_data_processing_and_analysis_methodology(engine: Optional[sqlalchemy.engine.Engine]) -> None:
    """Bundle the end-to-end data engineering and exploratory analysis workflow across Modules 1 to 3."""
    st.header('Data Processing and Analysis Methodology')
    st.caption('From raw labour-force extracts through diagnostics to analytics-ready tables.')
    with st.expander('Module 1 — Data Fundamentals & SQL (Schema, provenance, validation)', expanded=False):
        data_schema.page_data_and_schema(engine)
    with st.expander('Module 2 — Data cleaning and checking', expanded=False):
        cleaning_eda.page_cleaning_module_two(engine)
    with st.expander('Module 3 — Exploratory Diagnostics & Visualisation', expanded=False):
        cleaning_eda.page_visualisation_module_three(engine)


def page_modelling_methodology(engine: Optional[sqlalchemy.engine.Engine]) -> None:
    """Showcase the machine learning experimentation and evaluation flow."""
    module_4_machine_learning.module_4_page(engine)


def page_learnings(engine: Optional[sqlalchemy.engine.Engine]) -> None:
    """Distil the cross-module takeaways and provide follow-on assets."""
    st.header('Learnings')
    
    with st.expander('📋 Cross-module takeaways', expanded=False):
        st.markdown(
            """
            - **Hypothesis validation:** Occupation-level vulnerability remains concentrated in service, clerical, and certain professional tracks, confirming the strategic brief while highlighting the 2020 shock as an inflection point with 7.15% COVID peak in clerical roles.
            - **Data readiness:** Module 1 transformations plus Modules 2–3 quality gates establish a reproducible long-format warehouse with demographic enrichments for downstream analytics, enabling comprehensive 11-year MOM data analysis.
            - **Model efficacy:** The Module 4 pipeline delivers both point forecasts (KNN MAE: 0.34pp) and risk classification (logistic regression 0.82 predictability correlation), giving planners actionable forward-looking insight.
            - **Strategic transformation:** From reactive unemployment response to predictive intervention strategy, with precision-targeted programs reaching 800,000 workers and delivering 6:1 ROI within a 12-month action window.
            """
        )

    with st.expander('🎯 Key discoveries and impact metrics', expanded=False):
        st.markdown(
            """
            - **Occupation vulnerability gap:** 4.3x risk difference between cleaners and managers, confirming hypothesis about skill-based unemployment disparities
            - **Educational dynamics:** 40.7% of unemployment among degree holders, revealing complexity beyond traditional assumptions
            - **Predictive capability:** 0.82 past-to-future correlation enables reliable forecasting for proactive policy intervention
            - **Investment efficiency:** S$85M strategic investment targeting 120,000 high-risk workers with measurable economic returns
            """
        )

    with st.expander('🚀 Recommended next steps', expanded=False):
        st.markdown(
            """
            1. Automate quarterly ingestion from the Ministry of Manpower feeds and re-run feature engineering health checks.
            2. Integrate macroeconomic covariates (e.g., PMI, trade exposure) to stress-test model resilience.
            3. Deploy intervention playbooks for high-risk occupation clusters surfaced by the risk models.
            4. Establish continuous monitoring dashboard with real-time predictive alerts for policy makers.
            5. Implement evidence-driven policy framework with measurable KPIs and quarterly assessment cycles.
            """
        )

    st.markdown('### Closing reflections')
    st.markdown(
        """
        The capstone journey brought multiple disciplines together—SQL engineering, exploratory diagnostics,
        predictive modelling, stakeholder storytelling and BI tooling. Thank you to every mentor who supported the push from raw extracts to a decision-ready analytics asset.
        Also, a big thank you to all our classmates who walked beside us through the highs and lows — your support made this journey unforgettable.
        """
    )

    if GROUP_PHOTO_PATH.exists():
        st.image(
            str(GROUP_PHOTO_PATH),
            caption='SIT Data Analytics Group 6 — October 2025',
            width='stretch',
        )
    else:
        st.info(f'Group photo not found at {GROUP_PHOTO_PATH}.')

    st.markdown('### Appendix & downloads')
    with st.expander('Capstone artefacts', expanded=False):
        for artifact in APPENDIX_FILES:
            file_path = artifact['path']
            if file_path.exists():
                file_bytes = file_path.read_bytes()
                st.download_button(
                    label=f"Download {artifact['label']}",
                    data=file_bytes,
                    file_name=file_path.name,
                    mime=artifact.get('mime', 'application/octet-stream'),
                    key=f"download_{file_path.stem}",
                )
            else:
                st.warning(f"{artifact['label']} not found at {file_path}.")

def main():
    if APP_LOGO_PATH.exists():
        st.set_page_config(page_title=PAGE_TITLE, layout='wide', page_icon=str(APP_LOGO_PATH))
    else:
        st.set_page_config(page_title=PAGE_TITLE, layout='wide')
    
    # Initialize presentation mode state
    presentation_mode.initialize_presentation_state()
    
    # Get database engine
    engine = get_db_engine()
    
    # Check if in presentation mode
    if st.session_state.get('presentation_mode', False):
        # Render presentation mode
        presentation_mode.render_presentation_mode(engine)
    else:
        # Regular report mode
        st.title(PAGE_TITLE)

        if APP_LOGO_PATH.exists():
            st.sidebar.image(str(APP_LOGO_PATH), caption='Data Analytics Group 6', width='stretch')

        st.sidebar.markdown('## Navigation')
        page = st.sidebar.radio(
            'Go to',
            [
                'Overview',
                'Data Processing and Analysis Methodology',
                'Modelling Methodology',
                'Learnings',
            ],
        )

        if page == 'Overview':
            page_hypothesis(engine)
        elif page == 'Data Processing and Analysis Methodology':
            page_data_processing_and_analysis_methodology(engine)
        elif page == 'Modelling Methodology':
            page_modelling_methodology(engine)
        elif page == 'Learnings':
            page_learnings(engine)
    
    # Render mode toggle button in sidebar
    presentation_mode.render_mode_toggle_button()


if __name__ == '__main__':
    main()