import streamlit as st
import pandas as pd
from pathlib import Path


def page_overview():
    st.header('Overview')
    st.markdown('This Streamlit app is a narrative scaffold for the SIT-DA capstone project. Use the sidebar to jump between modules.')
    st.subheader('Problem statement')
    try:
        p = Path.cwd() / 'modules' / 'Problem statement.md'
        if not p.exists():
            p = Path(__file__).parent.parent / 'modules' / 'Problem statement.md'
        st.markdown(p.read_text(encoding='utf-8'))
    except Exception:
        st.markdown('Problem statement not found.')

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
