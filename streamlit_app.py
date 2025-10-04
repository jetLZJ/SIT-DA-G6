import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

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


def get_db_engine():
    conn = st.secrets.get('DB_CONNECTION_STRING')
    if not conn:
        return None
    return data_loader.engine_from_connection_string(conn)


def read_df_from_engine(engine, table_name: str, limit: int = 1000):
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


def page_data_and_schema(engine: Optional[object]):
    st.header('Module 1 — Data & Schema')
    st.markdown('Explore available tables (connects to DB via `st.secrets["DB_CONNECTION_STRING"]`)')

    if engine is None:
        st.warning('No DB connection available. Upload a CSV below to explore a sample table.')
        uploaded = st.file_uploader('Upload CSV for schema inspection', type=['csv'], key='schema_uploader')
        if uploaded:
            df = pd.read_csv(uploaded)
            st.write('Sample rows')
            st.dataframe(df.head())
            st.write('Columns')
            st.table(pd.DataFrame({'column': df.columns, 'dtype': [str(dt) for dt in df.dtypes]}))
        return

    try:
        tables = data_loader.list_tables(engine)
    except Exception as e:
        st.error(f'Failed to list tables: {e}')
        return

    st.write('Available tables (showing up to 200)')
    st.write(tables[:200])
    choice = st.selectbox('Select a table to preview', options=tables)
    if choice:
        try:
            df = read_df_from_engine(engine, choice, limit=1000)
            st.write(f'Preview of {choice} (showing up to 1000 rows)')
            st.dataframe(df.head())
            st.write('Columns & dtypes:')
            st.table(pd.DataFrame({'column': df.columns, 'dtype': [str(dt) for dt in df.dtypes]}))
        except Exception as e:
            st.error(f'Failed to read table: {e}')


def page_cleaning_and_eda(engine: Optional[object]):
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


def page_dashboard_and_deliverables(engine: Optional[object]):
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