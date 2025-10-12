import streamlit as st
import pandas as pd
from typing import Optional
import sqlalchemy

from app import data_loader, utils


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
                df = data_loader.read_table(engine, choice, limit=2000)
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
