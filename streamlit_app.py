import streamlit as st
from typing import Optional
import sqlalchemy

from app import data_loader
from app_pages import overview, data_schema, cleaning_eda, dashboard, module_4_machine_learning


PAGE_TITLE = 'SIT-DA Capstone — Labor Force Trends'


def get_db_engine() -> Optional[sqlalchemy.engine.Engine]:
    conn = st.secrets.get('DB_CONNECTION_STRING')
    if not conn:
        return None
    return data_loader.engine_from_connection_string(conn)


# pillar-focused renderers


def page_hypothesis(engine: Optional[sqlalchemy.engine.Engine]) -> None:
    """Frame the strategic hypothesis with the organised brief and problem narrative."""
    st.header('Hypothesis')
    st.caption('Strategic question and analytical framing derived from the industry brief.')
    overview.page_overview()


def page_data_processing_methodology(engine: Optional[sqlalchemy.engine.Engine]) -> None:
    """Bundle the end-to-end data engineering workflow across Modules 1 to 3."""
    st.header('Data Processing Methodology')
    st.caption('From raw labour-force extracts to trusted analytics-ready tables.')
    with st.expander('Module 1 — Data Fundamentals & SQL (Schema, provenance, validation)', expanded=True):
        data_schema.page_data_and_schema(engine)
    with st.expander('Modules 2 & 3 — Cleaning, EDA & Feature Preparation', expanded=True):
        cleaning_eda.page_cleaning_and_eda(engine)


def page_modelling_methodology(engine: Optional[sqlalchemy.engine.Engine]) -> None:
    """Showcase the machine learning experimentation and evaluation flow."""
    st.header('Modelling Methodology')
    st.caption('Forecasting and risk classification pipeline aligned to the unemployment hypothesis.')
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
        - **Operational enablement:** Shared data contracts, Power BI dashboards, and Streamlit diagnostics allow quarterly refreshes without rework.

        ### Recommended next steps
        1. Automate quarterly ingestion from the Ministry of Manpower feeds and re-run feature engineering health checks.
        2. Integrate macroeconomic covariates (e.g., PMI, trade exposure) to stress-test model resilience.
        3. Deploy intervention playbooks for high-risk occupation clusters surfaced by the risk models.
        """
    )
    with st.expander('Companion dashboard & deliverables (Module 4)', expanded=False):
        dashboard.page_dashboard_and_deliverables(engine)
def main():
    st.set_page_config(page_title=PAGE_TITLE, layout='wide')
    st.title(PAGE_TITLE)

    engine = get_db_engine()

    st.sidebar.markdown('## Navigation')
    page = st.sidebar.radio(
        'Go to',
        [
            'Hypothesis',
            'Data Processing Methodology',
            'Modelling Methodology',
            'Learnings',
        ],
    )

    if page == 'Hypothesis':
        page_hypothesis(engine)
    elif page == 'Data Processing Methodology':
        page_data_processing_methodology(engine)
    elif page == 'Modelling Methodology':
        page_modelling_methodology(engine)
    elif page == 'Learnings':
        page_learnings(engine)


if __name__ == '__main__':
    main()