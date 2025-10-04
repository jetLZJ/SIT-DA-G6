import streamlit as st
import pandas as pd
from typing import Optional
import sqlalchemy

from app import data_loader, utils, viz


def page_cleaning_and_eda(engine: Optional[sqlalchemy.engine.Engine]):
    st.header('Module 2 & 3 — Cleaning, EDA & Feature prep')

    df = None
    if engine is not None:
        try:
            tables = data_loader.list_tables(engine)
            default = tables[0] if tables else None
            choice = st.selectbox('Table to use for EDA', options=tables, index=0 if default else -1)
            if choice:
                df = data_loader.read_table(engine, choice, limit=2000)
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
