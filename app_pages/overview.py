import streamlit as st
import streamlit.components.v1 as components
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

    problem_statement_md = _load_problem_statement()

    st.subheader('Singapore Labour Force (Unemployment Insights)')
    default_embed_url = st.secrets.get('POWERBI_EMBED_URL', '')
    embed_url = st.text_input('Power BI embed URL', value=default_embed_url, placeholder='https://app.powerbi.com/...', key='overview_powerbi_url')

    if embed_url:
        components.iframe(embed_url, height=780)
    else:
        st.info('Provide a Power BI embed URL (from Publish to web or a secure embed token flow) to display the report here.')

    # with st.expander('How to generate the embed URL', expanded=False):
    #     st.markdown(
    #         '''
    #         1. In Power BI Service, open the report in your workspace.
    #         2. Select **File → Embed report** and choose either **Publish to web** (public data only) or **Website or portal** for secure embeds.
    #         3. Copy the generated embed URL. For secure embeds, ensure the consuming account has permission to view the report.
    #         4. Paste the URL above or store it in `st.secrets["POWERBI_EMBED_URL"]` for persistence.
    #         '''
    #     )

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
            st.markdown('**Primary research question**')
            st.markdown(
                """
                <p style="font-size:1.6rem;font-weight:600;margin-top:0.25rem;margin-bottom:1.5rem;">
                    Which occupations &amp; industries drive unemployment swings?
                </p>
                """,
                unsafe_allow_html=True,
            )
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
            """
        )

    with st.expander('Data requirements & readiness checklist', expanded=False):
        inventory = pd.DataFrame(
            [
                {
                    'Module & focus': 'Module 1 — Data Fundamentals & SQL',
                    'Key data assets': '`unemployment_rate_by_occupation_long`, `unemployed_by_age_sex_long`, `unemployed_by_qualification_sex_long`, `unemployed_by_previous_occupation_sex_long`',
                    'Readiness checks': 'SQL loaders executed; 2014-2024 coverage verified; column naming adheres to ingestion templates; stored in project database or upload-ready CSVs.',
                },
                {
                    'Module & focus': 'Module 2 — Cleaning & EDA',
                    'Key data assets': 'Long-format tables with unemployed_count + labour_force_count (or precomputed unemployment_rate) plus year fields for canonicalisation.',
                    'Readiness checks': 'Auto-detection maps resolve occupation/year columns; counts convertible to numeric; missing values <5% or flagged; year cast succeeds to `year_yr` and `year_dt`.',
                },
                {
                    'Module & focus': 'Module 3 — Visualisation & diagnostics',
                    'Key data assets': 'Module 2 cleaned DataFrame (`module23_clean_df`) with `occupation`, `year`, `unemployment_rate`, plus demographic splits for share-of-burden charts.',
                    'Readiness checks': 'Demographic columns present for `prepare_demographic_share`; computed `share_pct` free of nulls; long tables cached in session for trend/compare lenses.',
                },
                {
                    'Module & focus': 'Module 4 — Machine learning & risk scoring',
                    'Key data assets': 'Master modelling frame merged from long tables with lagged unemployment, PMET mix, qualification shares, demographic indicators, and 2025 scaffold.',
                    'Readiness checks': 'Each occupation has ≥6 years history (2014-2024); engineered features persist without nulls; classification labels available or fall back to notebook baselines.',
                },
            ]
        )
        st.table(inventory)

    with st.expander('Planned analytics playbook', expanded=False):
        st.markdown(
            """
            1. **Data hygiene:** Auto-detect canonical columns, derive unemployment rates from counts, recover year fields, and surface data-quality/outlier alerts.
            2. **Exploratory visuals:** Deploy trend, share-of-burden, and comparative lenses to narrate decade-long occupation trajectories.
            3. **Stratified diagnostics:** Quantify demographic exposure by education tier, gender, and age across occupation families.
            4. **Risk scoring:** Pair volatility indicators with logistic regression probabilities to surface high-risk occupations.
            5. **Predictive layer:** Operationalise KNN forecasts using 2014-2024 history with time-aware validation for 2025 outlooks.
            6. **Prescriptive output:** Blend forecast + risk signals to prioritise reskilling and contingency placement support.
            """
        )

    if not problem_statement_md:
        st.info('Full problem statement file not found. Check `modules/Problem statement.md`.')
