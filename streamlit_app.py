from pathlib import Path
from typing import Optional

import sqlalchemy
import streamlit as st

from app import data_loader
from app_pages import overview, data_schema, cleaning_eda, dashboard, module_4_machine_learning


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
        'path': MODULES_DIR / 'M1 Data Fundamentals and SQL G6 v2.docx',
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
    st.markdown(
        """
        ### Cross-module takeaways
        - **Hypothesis validation:** Occupation-level vulnerability remains concentrated in service, clerical, and certain professional tracks, confirming the strategic brief while highlighting the 2020 shock as an inflection point.
        - **Data readiness:** Module 1 transformations plus Modules 2–3 quality gates establish a reproducible long-format warehouse with demographic enrichments for downstream analytics.
        - **Model efficacy:** The Module 4 pipeline delivers both point forecasts (KNN ≈ 9.8% MAPE) and risk classification (logistic regression >70% ROC-AUC), giving planners actionable forward-looking insight.

        ### Recommended next steps
        1. Automate quarterly ingestion from the Ministry of Manpower feeds and re-run feature engineering health checks.
        2. Integrate macroeconomic covariates (e.g., PMI, trade exposure) to stress-test model resilience.
        3. Deploy intervention playbooks for high-risk occupation clusters surfaced by the risk models.
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
    st.title(PAGE_TITLE)

    if APP_LOGO_PATH.exists():
        st.sidebar.image(str(APP_LOGO_PATH), caption='Data Analytics Group 6', width='stretch')

    engine = get_db_engine()

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


if __name__ == '__main__':
    main()