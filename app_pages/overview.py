import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional


def _load_problem_statement() -> Optional[str]:
    potential_paths = [
        Path.cwd() / 'modules' / 'Problem statement.md',
        Path(__file__).parent.parent / 'modules' / 'Problem statement.md',
    ]
    for path in potential_paths:
        try:
            if path.exists():
                return path.read_text(encoding='utf-8')
        except Exception:
            continue
    return None


def page_overview():
    st.header('Project overview')
    st.caption('Singapore labour market resilience and unemployment risk signals')

    problem_statement_md = _load_problem_statement()

    with st.expander('Strategic brief', expanded=False):
        st.markdown(
            """
            - **Why now:** Structural shifts (automation, macro shocks, post-pandemic recovery) are widening unemployment
              gaps across Singapore's workforce.
            - **Business goal:** Deliver evidence-backed recommendations on which occupations demand immediate reskilling
              and policy support.
            - **Outcome:** A decision-grade analytics asset covering descriptive, predictive, and prescriptive views.
            """
        )

    with st.expander('Objectives & hypothesis', expanded=False):
        col_left, col_right = st.columns(2)
        with col_left:
            st.metric(label='Primary research question', value='Which occupations & industries drive unemployment swings?')
            st.markdown(
                """
                **Objectives**
                1. Flag consistently high or rising unemployment pockets.
                2. Quantify demographic/education levers shaping labour outcomes.
                3. Surface resilient versus vulnerable sectors.
                4. Generate forward-looking risk signals and reskilling targets.
                """
            )
        with col_right:
            st.markdown(
                """
                **Working hypothesis**
                > Lower-skilled occupations (service, sales, clerical, manual labour) exhibit higher and more volatile
                > unemployment than professional and managerial cohorts.

                **Why it matters**
                - Prioritises training budgets toward at-risk worker groups.
                - Anchors labour policy conversations with defensible evidence.
                - Builds the foundation for a reusable labour market monitoring tool.
                """
            )

    with st.expander('Analytic angles & guiding questions', expanded=False):
        st.markdown(
            """
            - **Trend lens:** Which occupations show persistent unemployment pressure? How did COVID-19 reshape trajectories?
            - **Human capital lens:** How do education tiers, gender and age groups mediate unemployment risk within each
              occupation family?
            - **Comparative lens:** Are high-skill/PMET roles structurally more resilient than lower-skill roles?
            - **Technology lens:** Do automation-exposed jobs display disproportionate risk?
            """
        )

    with st.expander('Data requirements & readiness checklist', expanded=False):
        inventory = pd.DataFrame(
            [
                {
                    'Dimension': 'Time coverage',
                    'Module source': 'Module 1 — Data Fundamentals',
                    'Database assets': '`unemployment_rate_by_occupation_long`, `unemployed_by_age_sex_long`, `unemployed_by_qualification_sex_long`',
                    'Status / notes': 'Years 2014-2024 already normalised into long tables (year column ready for trend & forecasting).',
                },
                {
                    'Dimension': 'Labour structure',
                    'Module source': 'Module 1 + Module 2',
                    'Database assets': '`unemployed_by_previous_occupation_sex_long`, `unemployed_pmets_by_age_long`, `long_term_unemployed_pmets_by_age_long`',
                    'Status / notes': 'Occupation / industry breakdown with unemployment_rate, unemployed_count, labour_force_count (Module 2 scripts recompute rates when needed).',
                },
                {
                    'Dimension': 'Demographics',
                    'Module source': 'Module 2 — Cleaning & EDA',
                    'Database assets': 'Same long tables joined with education_level, gender, age_group fields via notebook transformations',
                    'Status / notes': 'Ready for stratified diagnostics; Module 2 notebooks validate dtypes/imputation.',
                },
                {
                    'Dimension': 'Derived features',
                    'Module source': 'Module 3 — Visualisation & feature engineering',
                    'Database assets': 'In-app data marts assembled from canonical long tables (trend slopes, volatility, lagged rates)',
                    'Status / notes': 'Features computed on top of Module 1/2 tables; cached in session for modelling and risk scoring.',
                },
                {
                    'Dimension': 'Optional enrichments',
                    'Module source': 'Module 2 notebooks + external lookups',
                    'Database assets': 'Joins for `avg_wage`, `automation_risk_score` (if provided) staged in Module 2 ingestion cells',
                    'Status / notes': 'Optional CSV/DB tables can be merged through the same pipelines when available.',
                },
            ]
        )
        st.table(inventory)

    with st.expander('Planned analytics playbook', expanded=False):
        st.markdown(
            """
            1. **Data hygiene:** Validate joins, normalise taxonomies, reconcile unemployment-rate calculations.
            2. **Exploratory visuals:** Multi-year industry/occupation trendlines, heatmaps, volatility profiles.
            3. **Stratified diagnostics:** Compare unemployment rates across education, gender, and age segments.
            4. **Risk scoring:** Build volatility and trend slope metrics to isolate vulnerable cohorts.
            5. **Predictive layer:** Deploy time-series forecasts and classifiers to anticipate 12-month unemployment shifts.
            6. **Prescriptive output:** Rank reskilling priorities and map pathways to resilient sectors.
            """
        )

    if not problem_statement_md:
        st.info('Full problem statement file not found. Check `modules/Problem statement.md`.')
