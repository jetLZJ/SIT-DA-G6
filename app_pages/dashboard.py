import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from typing import Optional
import sqlalchemy

from app import data_loader, utils


def page_dashboard_and_deliverables(engine: Optional[sqlalchemy.engine.Engine]):
    st.header('Module 4 — Dashboard & Deliverables')
    st.markdown('This page demonstrates how the cleaned & prepared signals could be presented as a dashboard and packaged for deliverables.')

    if engine is None:
        st.info('Connect a DB or upload a CSV to preview dashboard widgets.')

    st.subheader('Embedded Power BI report')
    st.caption('Paste the embed URL from Power BI Service to preview the published dashboard inside Streamlit.')
    default_embed_url = st.secrets.get('POWERBI_EMBED_URL', '')
    embed_url = st.text_input('Power BI embed URL', value=default_embed_url, placeholder='https://app.powerbi.com/...')

    if embed_url:
        components.iframe(embed_url, height=780)
    else:
        st.info('Provide a Power BI embed URL (from Publish to web or a secure embed token flow) to display the report here.')

    with st.expander('How to generate the embed URL'):
        st.markdown(
            '''
            1. In Power BI Service, open the report in your workspace.
            2. Select **File → Embed report** and choose either **Publish to web** (public data only) or **Website or portal** for secure embeds.
            3. Copy the generated embed URL. For secure embeds, ensure the consuming account has permission to view the report.
            4. Paste the URL above or store it in `st.secrets["POWERBI_EMBED_URL"]` for persistence.
            '''
        )

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
