import streamlit as st
import pandas as pd
from pathlib import Path
import sqlalchemy

from app import data_loader


PERIOD_ORDER = [
    "2014-2016",
    "2017-2019",
    "2020-2021",
    "2022-2024"
]


def _year_to_period(year: int) -> str:
    if year <= 2016:
        return "2014-2016"
    if year <= 2019:
        return "2017-2019"
    if year <= 2021:
        return "2020-2021"
    return "2022-2024"


def _get_table_from_session(engine: sqlalchemy.engine.Engine | None, table_name: str) -> pd.DataFrame | None:
    key = f"m1_{table_name}"
    if key in st.session_state:
        return st.session_state[key]
    if engine is None:
        return None
    try:
        df = data_loader.read_table(engine, table_name)
    except Exception as exc:
        st.warning(f"Unable to read `{table_name}` from the database: {exc}")
        return None
    st.session_state[key] = df
    return df


def _occupation_period_summary(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty or "unemployed_rate" not in df.columns:
        return None
    tidy = df.copy()
    tidy["period"] = tidy["year"].astype(int).apply(_year_to_period)
    grouped = (
        tidy.groupby(["occupation", "period"], as_index=False)["unemployed_rate"].mean()
    )
    pivot = grouped.pivot(index="occupation", columns="period", values="unemployed_rate")
    pivot = pivot.reindex(columns=[col for col in PERIOD_ORDER if col in pivot.columns])
    if pivot.empty:
        return None
    sort_col = next((col for col in ["2020-2021", "2022-2024", "2017-2019", "2014-2016"] if col in pivot.columns), pivot.columns[0])
    return pivot.sort_values(by=sort_col, ascending=False)


def _static_occupation_summary() -> pd.DataFrame:
    data = {
        "Occupation": [
            "Associate Professionals & Technicians",
            "Cleaners, Labourers & Related Workers",
            "Clerical Support Workers",
            "Craftsmen & Related Trades Workers",
            "Managers & Administrators (Incl. Prop.)",
            "Plant & Machine Operators & Assemblers",
            "Professionals",
            "Service & Sales Workers",
        ],
        "2014-2016": [3.23, 4.00, 5.33, 3.00, 2.60, 3.20, 2.77, 5.17],
        "2017-2019": [3.30, 3.97, 5.67, 3.43, 2.63, 3.13, 2.90, 5.40],
        "2020-2021": [4.00, 5.60, 7.15, 3.95, 2.80, 3.85, 3.45, 7.05],
        "2022-2024": [2.77, 3.57, 5.47, 2.50, 2.23, 2.73, 2.57, 4.10],
    }
    return pd.DataFrame(data).set_index("Occupation")


def _static_education_summary() -> pd.DataFrame:
    data = {
        "Education": [
            "Below Secondary",
            "Secondary",
            "Post-Secondary (Non-Tertiary)",
            "Diploma & Professional Qualification",
            "Degree",
        ],
        "2014-2016": [3.55, 3.97, 4.04, 3.92, 3.90],
        "2017-2019": [3.59, 4.08, 4.83, 4.34, 3.94],
        "2020-2021": [4.87, 5.44, 5.68, 5.41, 4.32],
        "2022-2024": [2.96, 3.58, 4.14, 3.83, 3.16],
    }
    return pd.DataFrame(data).set_index("Education")


def _static_gender_gap_summary() -> pd.DataFrame:
    data = {
        "Education": [
            "Below Secondary",
            "Degree",
            "Secondary",
            "Post-Secondary (Non-Tertiary)",
            "Diploma & Professional Qualification",
        ],
        "2014-2016": [2.40, 1.07, 0.63, 1.07, 0.47],
        "2020-2021": [2.80, 0.10, 0.55, 0.20, 2.20],
        "2022-2024": [0.77, 0.43, 0.50, 1.30, 0.83],
        "Gap Reduction": [1.63, 0.63, 0.13, -0.23, -0.37],
    }
    return pd.DataFrame(data).set_index("Education")


def page_data_and_schema(engine: sqlalchemy.engine.Engine | None):
    st.header('Module 1 — Data Fundamentals & SQL')
    st.markdown('''
    Executive summary
    ''')

    st.subheader('Why transform wide -> long')
    st.markdown(
        """- Wide tables encode periods (years) as separate columns which makes joins, time-series aggregations and window functions cumbersome.
- Long tables store one observation per row (year, dimension, value), which is easier to query and analyze in SQL."""
    )

    st.markdown('---')
    st.subheader('Resulting long tables (from M1 materials)')
    long_table_examples = [
        'unemployed_by_age_sex_long',
        'unemployed_by_qualification_sex_long',
        'unemployed_by_marital_status_sex_long',
        'unemployment_rate_by_occupation_long',
        'unemployed_by_previous_occupation_sex_long',
        'unemployed_pmets_by_age_long',
        'long_term_unemployed_pmets_by_age_long'
    ]
    st.table(pd.DataFrame({'long_table': long_table_examples}))

    st.markdown('---')
    st.subheader('SQL examples (from M1)')
    sql_create_example = """-- Create DB and example wide table
DROP DATABASE IF EXISTS labourtrendsdb;
CREATE DATABASE labourtrendsDB;
USE labourtrendsDB;

DROP TABLE IF EXISTS unemployed_by_age_sex_wide;
CREATE TABLE unemployed_by_age_sex_wide (
    gender VARCHAR(20),
    age_group VARCHAR(20),
    year_2014 DECIMAL(5,1),
    -- ... year_2015 .. year_2024
    year_2024 DECIMAL(5,1)
);
"""
    sql_unpivot_example = """-- Unpivot / wide -> long pattern
DROP TABLE IF EXISTS unemployed_by_age_sex_long;
CREATE TABLE unemployed_by_age_sex_long AS
    SELECT 2014 AS year, gender, age_group, year_2014 AS unemployed_count
    FROM unemployed_by_age_sex_wide
    UNION ALL
    SELECT 2015, gender, age_group, year_2015
    FROM unemployed_by_age_sex_wide
    -- ... repeat for each year through 2024
    UNION ALL
    SELECT 2024, gender, age_group, year_2024
    FROM unemployed_by_age_sex_wide;
"""

    st.code(sql_create_example, language='sql')
    st.code(sql_unpivot_example, language='sql')

    # Prefer serving separate appendix files from modules/ if present
    create_path = Path(__file__).parent.parent / 'modules' / 'm1_appendix_create.sql'
    transform_path = Path(__file__).parent.parent / 'modules' / 'm1_appendix_transform.sql'

    if create_path.exists() and transform_path.exists():
        st.download_button('Download Appendix 1 — Create SQL', create_path.read_bytes(), file_name='m1_appendix_create.sql', mime='text/sql')
        st.download_button('Download Appendix 2 — Transform SQL', transform_path.read_bytes(), file_name='m1_appendix_transform.sql', mime='text/sql')
    else:
        full_sql_appendix = sql_create_example + "\n" + sql_unpivot_example
        st.download_button('Download example SQL (appendix)', full_sql_appendix.encode('utf-8'), file_name='m1_appendix_example.sql', mime='text/sql')

    st.markdown('---')
    st.subheader('Validation & safe checks')
    st.markdown('Use these non-destructive checks to validate long tables created from the transformations.')

    tables: list[str] = []
    existing_long: list[str] = []

    if engine is None:
        st.warning('No DB connection available. Upload a CSV to inspect schema locally below.')
        uploaded = st.file_uploader('Upload CSV for schema inspection', type=['csv'], key='schema_uploader')
        if uploaded:
            df = pd.read_csv(uploaded)
            st.write('Sample rows')
            st.dataframe(df.head())
            st.write('Columns & dtypes')
            st.table(pd.DataFrame({'column': df.columns, 'dtype': [str(dt) for dt in df.dtypes]}))
    else:
        try:
            tables = data_loader.list_tables(engine)
        except Exception as e:
            st.error(f'Failed to list tables: {e}')
            tables = []

        if tables:
            st.write('Available tables in the database (showing up to 200):')
            st.write(tables[:200])

        existing_long = [t for t in long_table_examples if t in tables]
        missing_long = [t for t in long_table_examples if t not in tables]
        st.write('Expected long tables present:')
        st.write(existing_long or 'None')
        if missing_long:
            st.info('Expected long tables missing (these should be generated by the transformation scripts):')
            st.write(missing_long)

        validate_table = st.selectbox('Select a long table to run validation queries (safe)', options=existing_long or [])
        if validate_table:
            try:
                with engine.connect() as conn:
                    result = conn.execute(sqlalchemy.text(f'SELECT COUNT(*) AS cnt FROM `{validate_table}`'))
                    try:
                        count = result.scalar()
                    except Exception:
                        row = result.fetchone()
                        count = row[0] if row is not None else 0
                st.write(f'Row count for {validate_table}: {int(count or 0):,}')
                preview = pd.read_sql(f'SELECT * FROM `{validate_table}` LIMIT 10', engine)
                st.dataframe(preview)
            except Exception as e:
                st.error(f'Validation query failed: {e}')

        st.markdown('---')
        st.subheader('Preview a wide table (example)')
        wide_choice = st.selectbox('Select a wide table to preview', options=[t for t in tables if t.endswith('wide')] or [])
        if wide_choice:
            try:
                df_wide = data_loader.read_table(engine, wide_choice, limit=5)
                st.write(f'Preview of {wide_choice}')
                st.dataframe(df_wide.head())
                st.markdown('Example unpivot SQL for this table:')
                st.code(sql_unpivot_example.replace('unemployed_by_age_sex_wide', wide_choice).replace('unemployed_by_age_sex_long', wide_choice.replace('_wide', '_long')), language='sql')
            except Exception as e:
                st.error(f'Failed to preview wide table: {e}')

    st.markdown('---')
    st.subheader('Industry & occupation risk lens')
    occupation_df = _get_table_from_session(engine, 'unemployment_rate_by_occupation_long')
    occupation_summary = _occupation_period_summary(occupation_df)
    source_note = 'Calculated from `unemployment_rate_by_occupation_long` (mean unemployment rate % by period)' if occupation_summary is not None else 'Summaries captured from Module 1 analysis notes'
    if occupation_summary is None:
        occupation_summary = _static_occupation_summary()
    st.caption(source_note)
    st.dataframe(occupation_summary.round(2))
    st.markdown(
        """**Insights linked to the problem statement**
        - Customer-facing and support roles (Clerical, Service & Sales) remain the most vulnerable, peaking above 7% during COVID-19 and staying elevated post-2022.
        - Technical trades and managerial tracks recover faster, reinforcing them as resilient anchors that can absorb displaced labour.
        - The period lens shows structural rather than cyclical risk for lower-skilled occupations — automation and demand shifts magnify volatility beyond crisis periods.
        """
    )
    with st.expander('Reskilling angles for high-risk occupations'):
        st.markdown(
            """- Prioritise digital administration and customer analytics pathways for clerical and sales workers to transition towards more resilient associate professional roles.
- Link service workers with trade-up programmes in logistics automation and advanced manufacturing (observed resilience in plant & machine operators).
- Fund targeted safety nets for gig and service workers during shocks to avoid hysteresis in unemployment."""
        )

    st.markdown('---')
    st.subheader('Education exposure & employability resilience')
    education_summary = _static_education_summary()
    st.dataframe(education_summary.round(2))
    st.markdown(
        """- Degree holders stabilise most quickly after shocks (unemployment ~3.1% in 2022-2024), underscoring protective effects of advanced credentials.
- Mid-tier qualifications (Diploma, Post-Secondary) endure the steepest COVID-19 spike and slower recovery — a priority cohort for upskilling support.
- Secondary and below-secondary groups bounce back sharply post-2022 but remain susceptible to structural shifts, requiring blended training plus job-matching services."""
    )

    st.markdown('---')
    st.subheader('Gender gaps within education tracks')
    gender_gap_summary = _static_gender_gap_summary()
    st.dataframe(gender_gap_summary)
    st.bar_chart(gender_gap_summary['Gap Reduction'])
    st.markdown(
        """- Gender parity advances among degree holders and the lowest-educated groups, proving that targeted policy works when funding and placements align.
- Diploma and post-secondary (non-tertiary) programmes show widening gaps, signalling the need for employer commitments and inclusive placement targets.
- Sustained monitoring of PMET vs non-PMET pathways is required so women do not lose ground when the economy tightens."""
    )

    st.markdown('---')
    st.subheader('Next analytical steps')
    st.markdown(
        """1. Blend unemployment counts with labour force denominators (when available) to compute fully normalised rates for education cohorts.
2. Extend volatility metrics (coefficient of variation) to highlight occupations at risk of repeated disruption.
3. Map high-risk occupations to growth sectors (e.g., logistics tech, healthcare operations) to produce the policy-ready reskilling shortlist requested in the problem statement."""
    )
