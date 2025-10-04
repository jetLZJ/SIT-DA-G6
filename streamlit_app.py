import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import sqlalchemy

from app import data_loader, utils, viz


PAGE_TITLE = 'SIT-DA Capstone — Labor Force Trends'
PALETTE = {
    'bg': '#F7F8FA',
    'card': '#FFFFFF',
    'text': '#0F1724',
    'muted': '#4B5563',
    'accent': '#2563EB',
    'positive': '#10B981'
}


def load_problem_statement() -> str:
    p = Path.cwd() / 'modules' / 'Problem statement.md'
    if not p.exists():
        p = Path(__file__).parent / 'modules' / 'Problem statement.md'
    try:
        return p.read_text(encoding='utf-8')
    except Exception:
        return 'Problem statement not found.'


def get_db_engine() -> Optional[sqlalchemy.engine.Engine]:
    conn = st.secrets.get('DB_CONNECTION_STRING')
    if not conn:
        return None
    return data_loader.engine_from_connection_string(conn)


def read_df_from_engine(engine: sqlalchemy.engine.Engine, table_name: str, limit: int = 1000):
    return data_loader.read_table(engine, table_name, limit=limit)


def page_overview():
    st.header('Overview')
    st.markdown('This Streamlit app is a narrative scaffold for the SIT-DA capstone project. Use the sidebar to jump between modules.')
    st.subheader('Problem statement')
    st.markdown(load_problem_statement())
    st.markdown('---')
    st.subheader('Quick demo data')
    uploaded = st.file_uploader('Optional: upload a CSV (if no DB connection)', type=['csv'])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head())
            st.info(f'Columns: {list(df.columns)}')
        except Exception as e:
            st.error(f'Failed to read CSV: {e}')


def page_data_and_schema(engine: Optional[sqlalchemy.engine.Engine]):
    st.header('Module 1 — Data Fundamentals & SQL')
    st.markdown(
        """
        Executive summary

        This module establishes the data foundation for the capstone. It explains where the raw tables came from, how they were ingested into a staging schema, and the tested SQL transforms that convert the original wide-format exports (years as columns) into a canonical long-format schema used by analysis, visualisation and modelling components.

        Provenance & ingest
        - Source: official labour-force CSV extracts (see Appendix files under `modules/`).
        - Ingest method: CSV -> staging tables in the database (LOAD DATA INFILE examples are provided). The app reads from the DB when `st.secrets['DB_CONNECTION_STRING']` is set; otherwise a local CSV upload fallback is offered for demos and verification.

        Transformation approach (wide → long)
        - Why: wide tables (year_2014 ... year_2024) make joins, window functions and time-series aggregations cumbersome. Converting to long format (one row per year/observation) simplifies analysis and ensures a single canonical source of truth for derived fields like unemployment_rate.
        - How: the transformation uses a portable SQL pattern that stacks per-year SELECT statements using UNION ALL to produce one long table. Where supported, DB-native UNPIVOT operations can replace the UNION pattern for brevity.

        Resulting canonical tables (created by the transform scripts)
        Each output table is long-format and contains a `year` column plus the relevant dimensions and a value column (`unemployed_count` or `unemployed_rate`). Examples produced from the M1 materials:
        - `unemployed_by_age_sex_long` — year, age_group, gender, unemployed_count
        - `unemployed_by_qualification_sex_long` — year, education, gender, unemployed_count
        - `unemployed_by_marital_status_sex_long` — year, marital_status, gender, unemployed_count
        - `unemployment_rate_by_occupation_long` — year, occupation, unemployed_rate
        - `unemployed_by_previous_occupation_sex_long` — year, occupation, gender, unemployed_count
        - `unemployed_pmets_by_age_long` — year, pmet_nonpmet, age_group, unemployed_count
        - `long_term_unemployed_pmets_by_age_long` — year, pmet_nonpmet, age_group, unemployed_count

        Key insights (high-level summaries from the M1 analysis)
        - Occupation patterns: Clerical support and service & sales workers recorded the highest and most persistent unemployment through 2014–2024, with a strong COVID-19 spike in 2020–2021 and partial recovery after 2022.
        - Education impact: Degree holders show relatively lower unemployment and greater resilience during shocks; mid-level qualifications (Diploma / Post-Secondary non-tertiary) experienced larger spikes during the pandemic.
        - Gender & education gaps: The dataset enables analysis of changes in male/female unemployment gaps by education level; degree holders show meaningful gap reduction over time whereas some mid-level groups saw widening gaps.

        Data contract (minimum required fields)
        - Required: `year` (INT), one or more dimension columns (occupation_name, industry_name, age_group, education_level, gender), `unemployed_count` (INT), `labor_force_count` (INT) where available.
        - Derived: `unemployment_rate` = unemployed_count / labor_force_count (float). Store derived fields in the canonical tables to avoid repeated computation in dashboards.

        Appendices & reproducibility
        - Appendix 1 (Create DB & staging) and Appendix 2 (Transform wide → long) are available as downloadable SQL files in `modules/` and via the download buttons below. These include CREATE TABLE, LOAD DATA INFILE examples and the UNION ALL transform patterns used to build the long tables.

        Notes & assumptions
        - Missing values: treat missing counts as NULL by default; explicit imputation rules (e.g., 0 when source metadata indicates) are documented in the cleaning log.
        - Label normalization: occupation/industry textual labels are normalised during cleaning to ensure consistent grouping across files.

        What this page provides
        - A provenance summary, schema previews, example SQL (create & unpivot), and safe read-only validation checks (COUNT / LIMIT) so stakeholders can verify results without modifying data.
        """
    )

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
    sql_create_example = '''-- Create DB and example wide table
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
'''
    sql_unpivot_example = '''-- Unpivot / wide -> long pattern
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
'''

    st.code(sql_create_example, language='sql')
    st.code(sql_unpivot_example, language='sql')

    # Prefer serving separate appendix files from modules/ if present
    create_path = Path(__file__).parent / 'modules' / 'm1_appendix_create.sql'
    transform_path = Path(__file__).parent / 'modules' / 'm1_appendix_transform.sql'

    if create_path.exists() and transform_path.exists():
        st.download_button('Download Appendix 1 — Create SQL', create_path.read_bytes(), file_name='m1_appendix_create.sql', mime='text/sql')
        st.download_button('Download Appendix 2 — Transform SQL', transform_path.read_bytes(), file_name='m1_appendix_transform.sql', mime='text/sql')
    else:
        full_sql_appendix = sql_create_example + "\n" + sql_unpivot_example
        st.download_button('Download example SQL (appendix)', full_sql_appendix.encode('utf-8'), file_name='m1_appendix_example.sql', mime='text/sql')

    st.markdown('---')
    st.subheader('Validation & safe checks')
    st.markdown('Use these non-destructive checks to validate long tables created from the transformations.')

    if engine is None:
        st.warning('No DB connection available. Upload a CSV to inspect schema locally below.')
        uploaded = st.file_uploader('Upload CSV for schema inspection', type=['csv'], key='schema_uploader')
        if uploaded:
            df = pd.read_csv(uploaded)
            st.write('Sample rows')
            st.dataframe(df.head())
            st.write('Columns & dtypes')
            st.table(pd.DataFrame({'column': df.columns, 'dtype': [str(dt) for dt in df.dtypes]}))
        return

    # If we have an engine, show available tables and run safe queries
    try:
        tables = data_loader.list_tables(engine)
    except Exception as e:
        st.error(f'Failed to list tables: {e}')
        return

    st.write('Available tables in the database (showing up to 200):')
    st.write(tables[:200])

    # Show which of the expected long tables exist
    existing_long = [t for t in long_table_examples if t in tables]
    missing_long = [t for t in long_table_examples if t not in tables]
    st.write('Expected long tables present:')
    st.write(existing_long or 'None')
    if missing_long:
        st.info('Expected long tables missing (these should be generated by the transformation scripts):')
        st.write(missing_long)

    # Safe validation queries: row counts and a sample
    validate_table = st.selectbox('Select a long table to run validation queries (safe)', options=existing_long or [])
    if validate_table:
        try:
            # Use a connection context and sqlalchemy.text to run a safe count query
            with engine.connect() as conn:
                result = conn.execute(sqlalchemy.text(f'SELECT COUNT(*) AS cnt FROM `{validate_table}`'))
                # SQLAlchemy Result supports scalar() in modern APIs
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

    # Allow users to preview a wide table and see an example unpivot query for it
    st.markdown('---')
    st.subheader('Preview a wide table (example)')
    wide_choice = st.selectbox('Select a wide table to preview', options=[t for t in tables if t.endswith('wide')] or [])
    if wide_choice:
        try:
            df_wide = read_df_from_engine(engine, wide_choice, limit=5)
            st.write(f'Preview of {wide_choice}')
            st.dataframe(df_wide.head())
            st.markdown('Example unpivot SQL for this table:')
            st.code(sql_unpivot_example.replace('unemployed_by_age_sex_wide', wide_choice).replace('unemployed_by_age_sex_long', wide_choice.replace('_wide', '_long')), language='sql')
        except Exception as e:
            st.error(f'Failed to preview wide table: {e}')


def page_cleaning_and_eda(engine: Optional[sqlalchemy.engine.Engine]):
    st.header('Module 2 & 3 — Cleaning, EDA & Feature prep')

    df = None
    if engine is not None:
        try:
            tables = data_loader.list_tables(engine)
            default = tables[0] if tables else None
            choice = st.selectbox('Table to use for EDA', options=tables, index=0 if default else -1)
            if choice:
                df = read_df_from_engine(engine, choice, limit=2000)
        except Exception:
            st.info('No tables found or failed to read from DB.')

    if df is None:
        uploaded = st.file_uploader('Upload CSV for EDA', type=['csv'], key='eda_upload')
        if uploaded:
            df = pd.read_csv(uploaded)

    if df is None:
        st.info('Provide a table from DB or upload a CSV to run EDA examples.')
        return

    st.subheader('Quick checks')
    st.write(f'Rows: {len(df):,} — Columns: {len(df.columns)}')
    st.write('Head:')
    st.dataframe(df.head())

    if set(['unemployed_count', 'labor_force_count']).issubset(df.columns):
        st.write('Computing unemployment_rate using `compute_unemployment_rate` helper...')
        try:
            df2 = utils.compute_unemployment_rate(df)
            st.dataframe(df2.head())
            if 'occupation' in df2.columns and 'year' in df2.columns and 'unemployment_rate' in df2.columns:
                st.subheader('Small multiples: unemployment rate by occupation over time')
                fig = viz.small_multiples_time_series(df2, group_col='occupation', x_col='year', y_col='unemployment_rate')
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f'Failed to compute unemployment_rate: {e}')
    else:
        st.info('The selected dataset does not have the `unemployed_count` and `labor_force_count` columns required for the example computation.')


def page_dashboard_and_deliverables(engine: Optional[sqlalchemy.engine.Engine]):
    st.header('Module 4 — Dashboard & Deliverables')
    st.markdown('This page demonstrates how the cleaned & prepared signals could be presented as a dashboard and packaged for deliverables.')

    if engine is None:
        st.info('Connect a DB or upload a CSV to preview dashboard widgets.')

    df = None
    if engine is not None:
        try:
            tables = data_loader.list_tables(engine)
            choice = st.selectbox('Table for dashboard', options=tables)
            if choice:
                df = read_df_from_engine(engine, choice, limit=2000)
        except Exception:
            pass

    if df is None:
        uploaded = st.file_uploader('Upload CSV for dashboard preview', type=['csv'], key='dash_upload')
        if uploaded:
            df = pd.read_csv(uploaded)

    if df is None:
        st.info('No data available for dashboard widgets.')
        return

    st.subheader('Example KPIs')
    if 'unemployed_count' in df.columns and 'labor_force_count' in df.columns:
        df2 = utils.compute_unemployment_rate(df)
        if 'year' in df2.columns:
            latest = df2.sort_values('year').groupby('occupation', as_index=False).last()
            kpi = latest['unemployment_rate'].mean()
            st.metric('Average unemployment_rate (latest by occupation)', f'{kpi:.2%}')

    st.markdown('---')
    st.markdown('Deliverables checklist:')
    st.checkbox('Cleaned dataset (CSV or DB table)')
    st.checkbox('Exploratory analysis (notebooks / Streamlit pages)')
    st.checkbox('Dashboard wireframes and interactive dashboard')
    st.checkbox('Presentation slides + write-up')


def safe_polyfit(x, y):
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) < 2:
            return None
        coef = np.polyfit(x, y, 1)
        return coef
    except Exception:
        return None


def main():
    st.set_page_config(page_title=PAGE_TITLE, layout='wide')
    st.title(PAGE_TITLE)

    engine = get_db_engine()

    st.sidebar.markdown('## Navigation')
    page = st.sidebar.radio('Go to', ['Overview', 'Module 1 — Data & Schema', 'Module 2 & 3 — Cleaning & EDA', 'Module 4 — Dashboard & Deliverables'])

    if page == 'Overview':
        page_overview()
    elif page == 'Module 1 — Data & Schema':
        page_data_and_schema(engine)
    elif page == 'Module 2 & 3 — Cleaning & EDA':
        page_cleaning_and_eda(engine)
    elif page == 'Module 4 — Dashboard & Deliverables':
        page_dashboard_and_deliverables(engine)


if __name__ == '__main__':
    main()