import streamlit as st
import pandas as pd
import numpy as np
import textwrap
from pathlib import Path
from typing import Optional
import plotly.express as px
import plotly.graph_objects as go
from functools import partial
import re


@st.cache_data
def _read_table_from_engine(_engine, table_name: str, limit: int = 20000) -> pd.DataFrame:
    """Read a table from an SQLAlchemy engine with a safety LIMIT.

    Keeps DB reads cached for the session and avoids repeated expensive queries.
    """
    try:
        return pd.read_sql(f'SELECT * FROM `{table_name}` LIMIT {limit}', con=_engine)  # type: ignore[arg-type]
    except Exception:
        # fall back to empty df on failure; callers will handle
        return pd.DataFrame()


def _normalize_and_compute_rates(df_in: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Heuristic normalisation of common column name variants and computation of a canonical rate column.

    Returns (df_out, mapping) where mapping documents detected and renamed columns.
    """
    mapping: dict = {'original_columns': list(df_in.columns)}
    df = df_in.copy()

    # canonicalise names (lower + underscore)
    col_map = {c.lower().strip().replace(' ', '_').replace('-', '_'): c for c in df.columns}

    # detect occupation-like column
    for cand in ('occupation', 'occupation_name', 'job_title'):
        if cand in col_map:
            df.rename(columns={col_map[cand]: 'occupation'}, inplace=True)
            mapping['renamed_occupation_from'] = col_map[cand]
            break

    # detect year-like column
    for cand in ('year', 'yr', 'period', 'date'):
        if cand in col_map:
            df.rename(columns={col_map[cand]: 'year'}, inplace=True)
            mapping['renamed_year_from'] = col_map[cand]
            break

    # detect rate/count columns
    detected_rate = None
    for candidate in ('unemployment_rate', 'unemployed_rate', 'unemployed_rate_pct', 'unemployment_rate_pct', 'unemp_rate'):
        if candidate in col_map:
            detected_rate = col_map[candidate]
            break

    # counts
    uc = next((col_map[c] for c in ('unemployed_count', 'unemployed', 'unemp_count') if c in col_map), None)
    lf = next((col_map[c] for c in ('labor_force_count', 'labour_force_count', 'laborforce_count') if c in col_map), None)

    if uc and lf:
        # safe vectorised division
        try:
            df['unemployed_count'] = pd.to_numeric(df[uc], errors='coerce')
            df['labor_force_count'] = pd.to_numeric(df[lf], errors='coerce')
            df['unemployment_rate'] = df['unemployed_count'] / df['labor_force_count']
            mapping['derived_unemployment_rate'] = True
        except Exception as e:
            mapping['derive_error'] = str(e)
    elif detected_rate:
        df['detected_rate_raw'] = pd.to_numeric(df[detected_rate], errors='coerce')
        # convert percent >1 to proportion
        df['unemployment_rate'] = df['detected_rate_raw'].where(df['detected_rate_raw'] <= 1, df['detected_rate_raw'] / 100.0)
        mapping['detected_rate_column'] = detected_rate
    else:
        # heuristic search
        cand = None
        for c in df.columns:
            if any(k in c.lower() for k in ('unemp', 'unemploy', 'rate')):
                cand = c
                break
        if cand is not None:
            df['detected_rate_raw'] = pd.to_numeric(df[cand], errors='coerce')
            df['unemployment_rate'] = df['detected_rate_raw'].where(df['detected_rate_raw'] <= 1, df['detected_rate_raw'] / 100.0)
            mapping['heuristic_rate_column'] = cand

    return df, mapping


def _is_placeholder_year(series: pd.Series | None) -> bool:
    if series is None:
        return True
    ser = pd.Series(series)
    year_values = pd.Series(dtype=float)
    if pd.api.types.is_datetime64_any_dtype(ser):
        year_values = ser.dt.year
    else:
        extracted = ser.astype(str).str.extract(r'((?:18|19|20|21)\d{2})')[0]
        year_values = pd.to_numeric(extracted, errors='coerce')
        if year_values.isna().all():
            numeric_guess = pd.to_numeric(ser, errors='coerce')
            year_values = numeric_guess
    year_values = year_values.dropna()
    if year_values.empty:
        return True
    if len(year_values.unique()) == 1 and float(year_values.iloc[0]) in {0.0, 1.0, 1970.0}:
        return True
    return False


def _wide_table_to_long(table_name: str, df_wide: pd.DataFrame) -> pd.DataFrame:
    df = df_wide.copy()
    if df.empty:
        return df
    year_cols = [c for c in df.columns if re.search(r'(?:18|19|20|21)\d{2}', str(c))]
    if not year_cols:
        return pd.DataFrame()
    id_cols = [c for c in df.columns if c not in year_cols]
    melted = df.melt(id_vars=id_cols, value_vars=year_cols, var_name='_year_col', value_name='_value')
    melted['_year_digits'] = melted['_year_col'].astype(str).str.extract(r'((?:18|19|20|21)\d{2})')[0]
    melted = melted.dropna(subset=['_year_digits'])
    if melted.empty:
        return pd.DataFrame()
    melted['year_int'] = melted['_year_digits'].astype(int)
    melted['year'] = pd.to_datetime(melted['year_int'], format='%Y', errors='coerce')
    measure_col = '_value'
    lower_name = table_name.lower()
    if 'rate' in lower_name:
        measure_col_name = 'unemployment_rate'
    elif any(keyword in lower_name for keyword in ('count', 'number', 'unemployed')):
        measure_col_name = 'unemployed_count'
    else:
        measure_col_name = f"value_{re.sub(r'[^a-z0-9]+', '_', lower_name).strip('_')}"
    melted.rename(columns={'_value': measure_col_name}, inplace=True)
    melted.drop(columns=['_year_col', '_year_digits'], inplace=True)
    cols = ['year', 'year_int'] + [c for c in id_cols if c != 'year'] + [measure_col_name]
    melted = melted[cols]
    return melted


def _build_master_df_from_long_frames(long_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not long_frames:
        return pd.DataFrame()

    def _safe_name(value: str) -> str:
        return (
            str(value)
            .strip()
            .replace(' ', '_')
            .replace('%', 'pct')
            .replace('&', 'and')
            .replace('/', '_')
            .replace('-', '_')
            .replace('__', '_')
        )

    master_frames: list[pd.DataFrame] = []
    for table_name, df in long_frames.items():
        if df is None or df.empty:
            continue
        dfc = df.copy()
        if 'year' not in dfc.columns:
            continue
        if pd.api.types.is_datetime64_any_dtype(dfc['year']):
            dfc['year_int'] = dfc['year'].dt.year
        else:
            y = pd.to_datetime(dfc['year'], errors='coerce')
            if y.notna().any():
                dfc['year_int'] = y.dt.year
            else:
                dfc['year_int'] = pd.to_numeric(dfc['year'], errors='coerce')
        dfc = dfc.drop(columns=['year'], errors='ignore')
        if 'year_int' not in dfc.columns:
            continue
        num_cols = [c for c in dfc.select_dtypes(include=['number']).columns if c != 'year_int']
        cat_cols = [c for c in dfc.select_dtypes(include=['object', 'category']).columns if c != 'year_int']
        if dfc['year_int'].dropna().empty:
            continue
        wide = pd.DataFrame({'year_int': sorted(dfc['year_int'].dropna().astype(int).unique())})
        if num_cols:
            for num in num_cols:
                if cat_cols:
                    for cat in cat_cols:
                        try:
                            pv = (
                                dfc.groupby(['year_int', cat])[num]
                                .sum()
                                .unstack(fill_value=0)
                                .rename(columns=lambda v: f"{_safe_name(table_name)}__{_safe_name(num)}__{_safe_name(cat)}__{_safe_name(v)}")
                                .reset_index()
                            )
                            wide = wide.merge(pv, on='year_int', how='left')
                        except Exception:
                            agg = (
                                dfc.groupby('year_int')[num]
                                .sum()
                                .reset_index()
                                .rename(columns={num: f"{_safe_name(table_name)}__{_safe_name(num)}"})
                            )
                            wide = wide.merge(agg, on='year_int', how='left')
                else:
                    agg = (
                        dfc.groupby('year_int')[num]
                        .sum()
                        .reset_index()
                        .rename(columns={num: f"{_safe_name(table_name)}__{_safe_name(num)}"})
                    )
                    wide = wide.merge(agg, on='year_int', how='left')
        else:
            for cat in cat_cols:
                try:
                    pv = (
                        dfc.groupby(['year_int', cat])
                        .size()
                        .unstack(fill_value=0)
                        .rename(columns=lambda v: f"{_safe_name(table_name)}__count__{_safe_name(cat)}__{_safe_name(v)}")
                        .reset_index()
                    )
                    wide = wide.merge(pv, on='year_int', how='left')
                except Exception:
                    continue
        wide = wide.fillna(0)
        master_frames.append(wide)

    if not master_frames:
        return pd.DataFrame()

    from functools import reduce

    master_df = reduce(lambda left, right: pd.merge(left, right, on='year_int', how='outer'), master_frames)
    master_df = master_df.sort_values('year_int').reset_index(drop=True)
    master_df['year'] = pd.to_datetime(master_df['year_int'], format='%Y', errors='coerce')
    cols = ['year', 'year_int'] + [c for c in master_df.columns if c not in ('year', 'year_int')]
    master_df = master_df[cols]
    return master_df


def _ensure_year_int(df_in: pd.DataFrame) -> pd.DataFrame:
    """Produce `year_yr` numeric column (float) using safe heuristics (see cleaning_eda)."""
    df_out = df_in.copy()
    recovered = False
    if 'year' in df_out.columns:
        col = df_out['year']
        # If already datetime-like
        if pd.api.types.is_datetime64_any_dtype(col) or pd.api.types.is_datetime64_dtype(col):
            years = col.dt.year
            if years.notna().any():
                unique_years = pd.Series(years.dropna().unique())
                if not unique_years.empty and not unique_years.eq(1970).all():
                    df_out['year_yr'] = years.astype(float)
                else:
                    # leave recovery to downstream heuristics — keep placeholder for signalling
                    df_out['year_yr'] = pd.Series([np.nan] * len(col), index=df_out.index)
            else:
                df_out['year_yr'] = pd.Series([np.nan] * len(col), index=df_out.index)
        else:
            numeric = pd.to_numeric(col, errors='coerce')
            if numeric.notna().any():
                df_out['year_yr'] = numeric.round().astype(float)
            else:
                parsed = pd.to_datetime(col, errors='coerce')
                if parsed.notna().any():
                    df_out['year_yr'] = parsed.dt.year.astype(float)

    if 'year_yr' in df_out.columns:
        df_out['year_yr'] = pd.to_numeric(df_out['year_yr'], errors='coerce').astype(float)
    if recovered:
        df_out.attrs['year_recovered'] = True
    return df_out


def _select_rate_column(df: pd.DataFrame) -> str:
    if 'unemployment_rate' in df.columns:
        return 'unemployment_rate'
    if 'unemployed_rate' in df.columns:
        return 'unemployed_rate'
    return ''


@st.cache_data
def _build_master_df_from_long(_engine, max_tables: int = 6, per_table_limit: int = 5000) -> pd.DataFrame:
    """Attempt to reconstruct a master year-level dataframe by pivoting a sample of long tables.

    This mirrors the notebook approach but limits table/row counts for interactivity.
    """
    try:
        import sqlalchemy
        inspector = sqlalchemy.inspect(_engine)
        all_tables = inspector.get_table_names()
        long_tables = [t for t in all_tables if isinstance(t, str) and t.endswith('long')]
        long_tables = long_tables[:max_tables]
        master_frames = []
        from functools import reduce

        for table in long_tables:
            try:
                df = pd.read_sql(f'SELECT * FROM `{table}` LIMIT {per_table_limit}', con=_engine)  # type: ignore[arg-type]
            except Exception:
                continue
            if df is None or df.empty or 'year' not in df.columns:
                continue

            dfc = df.copy()
            # ensure numeric year_int
            if pd.api.types.is_datetime64_any_dtype(dfc['year']):
                dfc['year_int'] = dfc['year'].dt.year
            else:
                y = pd.to_datetime(dfc['year'], errors='coerce')
                if y.notna().any():
                    dfc['year_int'] = y.dt.year
                else:
                    try:
                        dfc['year_int'] = pd.to_numeric(dfc['year'], errors='coerce').astype('Int64')
                    except Exception:
                        continue

            # numeric and categorical
            num_cols = [c for c in dfc.select_dtypes(include=['number']).columns if c != 'year_int']
            cat_cols = [c for c in dfc.select_dtypes(include=['object', 'category']).columns if c != 'year_int']

            wide = pd.DataFrame({'year_int': sorted(dfc['year_int'].dropna().unique())})

            if num_cols:
                for num in num_cols:
                    if cat_cols:
                        for cat in cat_cols:
                            try:
                                pv = (
                                    dfc.groupby(['year_int', cat])[num]
                                    .sum()
                                    .unstack(fill_value=0)
                                    .rename(columns=lambda v: f"{table}__{num}__{cat}__{v}")
                                    .reset_index()
                                )
                                wide = wide.merge(pv, on='year_int', how='left')
                            except Exception:
                                agg = dfc.groupby('year_int')[num].sum().reset_index().rename(columns={num: f"{table}__{num}"})
                                wide = wide.merge(agg, on='year_int', how='left')
                    else:
                        agg = dfc.groupby('year_int')[num].sum().reset_index().rename(columns={num: f"{table}__{num}"})
                        wide = wide.merge(agg, on='year_int', how='left')
            else:
                for cat in cat_cols:
                    try:
                        pv = (
                            dfc.groupby(['year_int', cat])
                            .size()
                            .unstack(fill_value=0)
                            .rename(columns=lambda v: f"{table}__count__{cat}__{v}")
                            .reset_index()
                        )
                        wide = wide.merge(pv, on='year_int', how='left')
                    except Exception:
                        continue

            wide = wide.fillna(0)
            master_frames.append(wide)

        if not master_frames:
            return pd.DataFrame()
        master_df = reduce(lambda left, right: pd.merge(left, right, on='year_int', how='outer'), master_frames)
        master_df = master_df.sort_values('year_int').reset_index(drop=True)
        master_df['year'] = pd.to_datetime(master_df['year_int'], format='%Y', errors='coerce')
        cols = ['year', 'year_int'] + [c for c in master_df.columns if c not in ('year', 'year_int')]
        master_df = master_df[cols]
        return master_df
    except Exception:
        return pd.DataFrame()


def _wide_unemp_to_long(master_df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to extract occupation-unemployment columns from a wide master_df and melt into long form.

    Heuristic: find columns containing 'unemployed' or 'unemployment' and a separator '__' then extract occupation.
    """
    if master_df is None or master_df.empty:
        return pd.DataFrame()
    cand_cols = [c for c in master_df.columns if any(k in c.lower() for k in ('unemploy', 'unemp', 'unemployment')) and '__' in c]
    rows = []
    for col in cand_cols:
        # attempt to parse occupation from last segment
        try:
            occ = col.split('__')[-1]
            ser = master_df[['year', 'year_int', col]].rename(columns={col: 'unemployment_rate'})
            ser['occupation'] = occ
            rows.append(ser[['year', 'year_int', 'occupation', 'unemployment_rate']])
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    long = pd.concat(rows, ignore_index=True)
    # clean and normalise rates
    long['unemployment_rate'] = pd.to_numeric(long['unemployment_rate'], errors='coerce')
    long.loc[long['unemployment_rate'] > 1, 'unemployment_rate'] = long.loc[long['unemployment_rate'] > 1, 'unemployment_rate'] / 100.0
    return long


def module_4_page(engine: Optional[object]):
    """Module 4 — Machine Learning (faithful extraction of M4 Machine Learning.ipynb)

    Presentation-style page that follows the notebook flow and ties outputs back to the Problem Statement.
    The function accepts an SQLAlchemy engine (provided by the main app) or falls back to CSV upload.
    """

    st.title('Module 4 — Machine Learning')

    # Notebook-style Executive summary & Introduction (extracted from M4 Machine Learning.ipynb)
    st.markdown('## Module 4 — Singapore Occupational Unemployment Prediction 2025')
    st.markdown(textwrap.dedent(
        """
        This module provides forecasts of occupation-specific unemployment rates for 2025 and identifies
        high-risk occupations to prioritise reskilling and policy interventions. The analysis follows the
        original M4 notebook: data ingestion -> master-frame construction -> feature engineering ->
        KNN regression and logistic classification examples, and ends with findings and recommendations.
        """
    ))

    st.markdown('### Context & objectives')
    st.markdown(textwrap.dedent(
        """
        Singapore's labour market is undergoing structural change. This module aims to:
        - Forecast 2025 unemployment rates by occupation
        - Identify occupations at elevated risk of unemployment increases
        - Quantify uncertainty and provide actionable recommendations aligned to the project Problem Statement
        """
    ))
    st.markdown('---')

    st.header('Data sources & quality assessment')
    st.markdown(textwrap.dedent(
        """
        Our analysis integrates six Ministry of Manpower datasets, covering occupation unemployment, demographics,
        qualifications and PMET distribution between 2014 and 2024. The combined corpus is complete across all years,
        adheres to standard occupational taxonomies and underwent normalisation (percentage scaling, categorical
        harmonisation, missing-value imputation below 2%). Remaining limitations include broad occupation buckets,
        potential shocks from events such as COVID-19, and limited visibility into sub-sector nuances.
        """
    ))

    st.header('Historical unemployment narratives')
    st.markdown(textwrap.dedent(
        """
        Ten-year unemployment trajectories reveal differentiated volatility: service workers and craftsmen experience the
        largest swings, while professional cohorts trend more steadily. The 2020 pandemic produced sharp but uneven
        disruptions, with several occupations displaying slower recoveries. Divergence patterns since 2021 signal that
        structural shifts are underway, reinforcing the need for occupation-specific forecasting.
        """
    ))

    st.header('Data preparation & feature engineering overview')
    st.markdown(textwrap.dedent(
        """
        The notebook pipeline converts occupation rate columns into long form, merges year-level demographic features and
        engineers lagged unemployment targets. Supervisor-ready data includes current rate, lag-1 rate, demographic ratios,
        qualification mix, PMET indicators and year encodings. Samples without next-year targets are excluded to preserve
        temporal integrity ahead of modelling.
        """
    ))

    # Data connection and loading
    st.header('Data connection & loading')
    st.markdown('This page reads canonical long-format tables from the DB when available (via the engine passed by the main app). Otherwise upload a representative CSV.')

    df = None
    master_df = None
    long_df = None
    if engine is not None:
        st.info('Attempting to use the DB engine passed from the main app (mirrors notebook auto-selection).')
        try:
            import sqlalchemy
            inspector = getattr(sqlalchemy, 'inspect')(engine)
            try:
                tables = inspector.get_table_names() or []
            except Exception:
                tables = []
            long_tables = [t for t in tables if isinstance(t, str) and t.endswith('long')]
            wide_tables = [t for t in tables if isinstance(t, str) and t.endswith('wide')]

            if not long_tables:
                st.warning('No long tables detected in the connected DB; upload a CSV to proceed.')
            if not wide_tables:
                st.info('No wide tables detected; master dataframe reconstruction will rely solely on long tables.')

            sample_limit = 5000
            long_dict: dict[str, pd.DataFrame] = {}
            for table in long_tables:
                frame = _read_table_from_engine(engine, table, limit=sample_limit)
                if frame is not None and not frame.empty:
                    long_dict[table] = frame

            wide_dict: dict[str, pd.DataFrame] = {}
            for table in wide_tables:
                frame = _read_table_from_engine(engine, table, limit=sample_limit)
                if frame is not None and not frame.empty:
                    wide_dict[table] = frame

            if wide_dict:
                st.session_state['module4_wide_tables_dict'] = wide_dict

            if long_dict:
                for name, frame in list(long_dict.items()):
                    year_series = frame['year'] if 'year' in frame.columns else None
                    if _is_placeholder_year(year_series):
                        candidate = name.replace('_long', '_wide')
                        if candidate in wide_dict:
                            rebuilt = _wide_table_to_long(candidate, wide_dict[candidate])
                            if rebuilt is not None and not rebuilt.empty:
                                long_dict[name] = rebuilt
                                st.info(f'Rebuilt `{name}` from `{candidate}` to restore year values.')

                st.session_state['module4_long_tables_dict'] = long_dict

                table_options = list(long_dict.keys())
                preferred_order = [
                    'unemployment_rate_by_occupation_long',
                    'unemployed_by_previous_occupation_sex_long',
                    'unemployed_by_occupation_long',
                ]
                default_table = next((t for t in preferred_order if t in long_dict), table_options[0])
                default_index = table_options.index(default_table)
                selected_table = st.selectbox('Select long table for analysis', options=table_options, index=default_index)
                df = long_dict[selected_table].copy()
                long_df = df.copy()
                st.success(f'Loaded table `{selected_table}` ({len(df):,} rows)')
                st.session_state['module4_selected_table'] = selected_table
                st.session_state['module4_long_df'] = df.copy()
            else:
                st.info('No DB tables returned rows after sampling; upload a CSV to continue.')
        except Exception as db_err:
            st.error(f'DB inspection failed — falling back to CSV upload. Details: {db_err}')

    if df is None:
        uploaded = st.file_uploader('Upload representative long-format CSV (year, occupation, unemployed_rate or counts)', type=['csv'])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f'Failed to read uploaded CSV: {e}')

    if df is None:
        st.info('No data loaded — connect DB or upload a CSV to run the analysis.')
        return

    # Notebook pipeline: optionally build a master dataframe from multiple long tables (mirrors notebook flow)
    master_df = None
    long_df = None
    if engine is not None:
        long_tables_cache: dict[str, pd.DataFrame] = st.session_state.get('module4_long_tables_dict', {})
        if st.button('Build master dataframe from DB (notebook pipeline)'):
            if not long_tables_cache:
                st.warning('No cached long tables available. Ensure the DB connection loaded tables successfully.')
            else:
                with st.spinner('Building master dataframe (this may take a few seconds)...'):
                    master_df = _build_master_df_from_long_frames(long_tables_cache)
                    if master_df is None or master_df.empty:
                        st.warning('Master dataframe construction returned no data. Ensure DB tables contain year values.')
                    else:
                        st.success(f'Master dataframe constructed: {master_df.shape[0]} rows')
                        st.dataframe(master_df.head())
                        st.session_state['module4_master_df'] = master_df
        if 'module4_master_df' in st.session_state and (master_df is None or master_df.empty):
            stored_master = st.session_state['module4_master_df']
            if stored_master is not None and not stored_master.empty:
                master_df = stored_master
        if 'module4_long_df' in st.session_state and (long_df is None or long_df.empty):
            cached_long = st.session_state['module4_long_df']
            if cached_long is not None and not cached_long.empty:
                long_df = cached_long

    # allow extracting long-form unemployment series from master_df
    if master_df is not None and not master_df.empty:
        if st.button('Extract occupation unemployment long series from master_df'):
            with st.spinner('Extracting occupation-series...'):
                long_df = _wide_unemp_to_long(master_df)
                if long_df is None or long_df.empty:
                    st.warning('No occupation unemployment columns found in master_df.')
                else:
                    st.success(f'Extracted long-form series: {long_df.shape[0]} rows')
                    st.dataframe(long_df.head())
                    st.session_state['module4_long_df'] = long_df
    elif 'module4_long_df' in st.session_state:
        long_df = st.session_state['module4_long_df']

    # If a long-form was extracted from master_df, prefer it as the analysis df
    if long_df is not None and not long_df.empty:
        df = long_df.copy()

    st.subheader('Data preview')
    st.write(f'Rows: {len(df):,} — Columns: {len(df.columns)}')
    st.dataframe(df.head())

    # Debug expander: show detected columns, dtypes and mappings
    with st.expander('Debug: detected columns & parsing diagnostics', expanded=False):
        st.write('Columns:', list(df.columns))
        st.write('Dtypes:')
        st.table(pd.DataFrame({'column': df.columns, 'dtype': [str(dt) for dt in df.dtypes]}))
        # detect candidate rate columns
        rate_cols = [c for c in df.columns if 'unemp' in c.lower() or 'unemploy' in c.lower() or 'rate' in c.lower()]
        st.write('Candidate rate/count columns:', rate_cols)
        # year parsing check
        if 'year' in df.columns:
            try:
                yrs = pd.to_datetime(df['year'], errors='coerce').dt.year
                st.write('Parsed years unique sample:', sorted(list(pd.Series(yrs.dropna().unique())[:10])))
            except Exception as e:
                st.write('Year parse error:', e)
    # Data prep: canonical year and unemployment proportion (use canonical helpers)
    st.header('Data preparation')
    # Keep a copy of the pre-normalised dataframe for fallback parsing later
    df_raw = df.copy()

    # Normalize column names and compute canonical unemployment_rate where possible
    try:
        df, mapping = _normalize_and_compute_rates(df)
    except Exception as e:
        st.warning(f'Rate normalisation step failed: {e}')
        mapping = {'original_columns': list(df.columns)}

    # Ensure we have a numeric year column for plotting/analysis
    df = _ensure_year_int(df)

    year_series: pd.Series | None = None
    if 'year_yr' in df.columns and df['year_yr'].notna().any():
        year_series = df['year_yr']
    elif 'year_int' in df.columns and df['year_int'].notna().any():
        year_series = df['year_int']
    elif 'year' in df.columns and df['year'].notna().any():
        year_series = df['year']

    if year_series is None:
        st.error('Could not recover a year dimension — please ensure the dataset includes a `year` column.')
        return

    year_series_name = getattr(year_series, 'name', 'unknown')
    year_source_label = f'initial ({year_series_name})'

    year_string_candidates: list[pd.Series] = [year_series.astype(str)]
    if 'year' in df.columns:
        year_string_candidates.append(df['year'].astype(str))
    if 'year' in df_raw.columns:
        year_string_candidates.append(df_raw['year'].astype(str))
    raw_year_strings = year_string_candidates[0]

    year_series_numeric = pd.to_numeric(year_series, errors='coerce')
    if year_series_numeric.notna().any():
        df['year_num'] = year_series_numeric.astype(float)
        year_source_label = f'numeric from {year_series_name}'
    else:
        parsed_years = pd.to_datetime(year_series, errors='coerce')
        if parsed_years.notna().any():
            df['year_num'] = parsed_years.dt.year.astype(float)
            year_source_label = f'datetime parse from {year_series_name}'
        else:
            df['year_num'] = np.nan

    if 'year_int' in df.columns and df['year_int'].notna().any():
        df['year_display'] = df['year_int'].astype('Int64').astype(str)
    elif 'year' in df.columns:
        df['year_display'] = df['year'].astype(str)
    else:
        df['year_display'] = year_series.astype(str)

    # If the initial numeric conversion produced implausible years (e.g., all 1970), attempt to recover
    if 'year_num' in df.columns:
        unique_years = pd.Series(df['year_num'].dropna().unique())
        if not unique_years.empty:
            all_same = len(unique_years) == 1
            suspicious_values = {0.0, 1.0, 1970.0}
            out_of_range = unique_years.lt(1800).all() or unique_years.gt(2100).all()
            if (all_same and unique_years.iloc[0] in suspicious_values) or out_of_range:
                recovered = False
                if 'year_int' in df.columns and df['year_int'].notna().any():
                    alt_numeric = pd.to_numeric(df['year_int'], errors='coerce')
                    alt_numeric = alt_numeric.where(alt_numeric.between(1900, 2100))
                    if alt_numeric.notna().any() and alt_numeric.nunique() > 1:
                        df['year_num'] = alt_numeric.astype(float)
                        df['year_display'] = df['year_num'].round().astype('Int64').astype(str)
                        recovered = True
                        year_source_label = 'year_int numeric fallback'
                if not recovered:
                    for series_candidate in year_string_candidates:
                        extracted_alt = series_candidate.str.extract(r'((?:18|19|20|21)\d{2})')[0]
                        alt_numeric = pd.to_numeric(extracted_alt, errors='coerce')
                        alt_numeric = alt_numeric.where(alt_numeric.between(1900, 2100))
                        if alt_numeric.notna().any() and alt_numeric.nunique() > 1:
                            df['year_num'] = alt_numeric.astype(float)
                            df['year_display'] = df['year_num'].round().astype('Int64').astype(str)
                            recovered = True
                            source_name = getattr(series_candidate, 'name', 'regex_candidate')
                            year_source_label = f'regex fallback from {source_name}'
                            break
                if not recovered:
                    candidate_cols = [
                        c for c in df.columns
                        if c not in {'year', 'year_num', 'year_display', 'year_yr', 'year_int'}
                        and any(keyword in c.lower() for keyword in ('year', 'period', 'date'))
                    ]
                    df_sources = [df]
                    if df_raw is not None:
                        df_sources.append(df_raw)
                    for source_df in df_sources:
                        for cand in candidate_cols:
                            if cand not in source_df.columns:
                                continue
                            ser = source_df[cand]
                            numeric_candidate = pd.to_numeric(ser, errors='coerce')
                            numeric_candidate = numeric_candidate.where(numeric_candidate.between(1900, 2100))
                            if numeric_candidate.notna().any() and numeric_candidate.nunique() > 1:
                                df['year_num'] = numeric_candidate.astype(float)
                                df['year_display'] = df['year_num'].round().astype('Int64').astype(str)
                                recovered = True
                                year_source_label = f'column {cand} numeric fallback'
                                break
                            extracted_cand = ser.astype(str).str.extract(r'((?:18|19|20|21)\d{2})')[0]
                            numeric_candidate = pd.to_numeric(extracted_cand, errors='coerce')
                            numeric_candidate = numeric_candidate.where(numeric_candidate.between(1900, 2100))
                            if numeric_candidate.notna().any() and numeric_candidate.nunique() > 1:
                                df['year_num'] = numeric_candidate.astype(float)
                                df['year_display'] = df['year_num'].round().astype('Int64').astype(str)
                                recovered = True
                                year_source_label = f'column {cand} regex fallback'
                                break
                        if recovered:
                            break
                if not recovered and isinstance(mapping.get('renamed_year_from'), str):
                    orig_col = mapping['renamed_year_from']
                    source_candidates = []
                    if orig_col in df.columns:
                        source_candidates.append(df[orig_col])
                    if orig_col in df_raw.columns:
                        source_candidates.append(df_raw[orig_col])
                    for original_series in source_candidates:
                        extracted_orig = original_series.astype(str).str.extract(r'((?:18|19|20|21)\d{2})')[0]
                        numeric_orig = pd.to_numeric(extracted_orig, errors='coerce')
                        numeric_orig = numeric_orig.where(numeric_orig.between(1900, 2100))
                        if numeric_orig.notna().any() and numeric_orig.nunique() > 1:
                            df['year_num'] = numeric_orig.astype(float)
                            df['year_display'] = df['year_num'].round().astype('Int64').astype(str)
                            recovered = True
                            year_source_label = f'original column {orig_col} regex fallback'
                            break

    valid_year_mask = pd.Series(False, index=df.index)
    if df['year_num'].notna().any():
        valid_year_mask = df['year_num'].between(1900, 2100, inclusive='both')

    if not valid_year_mask.any():
        extracted = df['year_display'].str.extract(r'((?:19|20)\d{2})')[0]
        df['year_num'] = pd.to_numeric(extracted, errors='coerce')
        if df['year_num'].notna().any():
            valid_year_mask = df['year_num'].between(1900, 2100, inclusive='both')

    if valid_year_mask.any():
        df = df[df['year_num'].notna()].copy()
        df['year_display'] = df['year_num'].round().astype(int).astype(str)
        df.attrs['year_axis_type'] = 'numeric'
        df.attrs['year_category_order'] = df.sort_values('year_num')['year_display'].unique().tolist()
    else:
        df['year_display'] = df['year_display'].fillna('Unknown')
        order = (
            df[['year_display']]
            .drop_duplicates()
            .reset_index(drop=True)
            .assign(_year_ordinal=lambda d: d.index.astype(float))
        )
        df = df.merge(order, on='year_display', how='left')
        df['year_num'] = df['_year_ordinal']
        df.drop(columns=['_year_ordinal'], inplace=True)
        df.attrs['year_axis_type'] = 'categorical'
        df.attrs['year_category_order'] = order['year_display'].tolist()
        df = df[df['year_num'].notna()].copy()
        year_source_label = year_source_label + ' (categorical fallback)'

    year_values = df['year_num'].dropna()
    suspicious_placeholder = False
    if year_values.empty:
        suspicious_placeholder = True
    elif year_values.nunique() == 1 and float(year_values.iloc[0]) in {0.0, 1.0, 1970.0}:
        suspicious_placeholder = True

    if suspicious_placeholder:
        with st.expander('Year override options', expanded=True):
            st.warning('Detected placeholder year values (e.g., 1970). Provide a start year to reconstruct a sequential axis if the dataset lacks explicit years.')
            override_enabled = st.checkbox('Override with sequential years', value=True, key='module4_year_override_enabled')
            if override_enabled:
                default_start = int(st.session_state.get('module4_year_override_start', 2014))
                start_year = int(st.number_input('Start year', min_value=1900, max_value=2100, value=default_start, key='module4_year_override_start'))
                seq_col = df.groupby('occupation').cumcount()
                df['year_num'] = start_year + seq_col
                df['year_display'] = df['year_num'].astype(int).astype(str)
                df.attrs['year_axis_type'] = 'numeric'
                df.attrs['year_category_order'] = df.sort_values('year_num')['year_display'].unique().tolist()
                year_source_label = f'sequential override from {start_year}'
            else:
                st.info('Keeping categorical ordering for year axis. Adjust override above if needed.')

    df.attrs['year_source'] = year_source_label

    with st.expander('Debug: year normalisation state', expanded=False):
        st.write('Year source heuristic:', df.attrs.get('year_source', 'unknown'))
        try:
            unique_years_sorted = sorted([float(y) for y in df['year_num'].dropna().unique().tolist()])
            st.write('Unique year_num values:', unique_years_sorted[:50])
        except Exception as dbg_err:
            st.write('Unable to list year_num values:', dbg_err)
        try:
            cols_to_show = [c for c in ['year', 'year_int', 'year_yr', 'year_num', 'year_display'] if c in df.columns]
            if cols_to_show:
                st.write('Sample of year-related columns:')
                st.dataframe(df[cols_to_show].head(15))
        except Exception as dbg_err:
            st.write('Unable to show sample rows:', dbg_err)
        try:
            st.write('Raw year string candidates (first few rows):')
            st.write([cand.head(5).tolist() for cand in year_string_candidates[:3]])
        except Exception:
            pass

    # produce unemp_prop (proportion) used across the page
    if 'unemployment_rate' in df.columns:
        df['unemp_prop'] = pd.to_numeric(df['unemployment_rate'], errors='coerce')
    elif 'unemployed_rate' in df.columns:
        df['unemp_prop'] = pd.to_numeric(df['unemployed_rate'], errors='coerce')
        df.loc[df['unemp_prop'] > 1, 'unemp_prop'] = df.loc[df['unemp_prop'] > 1, 'unemp_prop'] / 100.0
    else:
        # fallback: use detected columns from mapping if present
        det = mapping.get('detected_rate_column') or mapping.get('heuristic_rate_column')
        det_col = None
        if isinstance(det, str) and det in df.columns:
            det_col = det
        if det_col:
            df['unemp_prop'] = pd.to_numeric(df[det_col], errors='coerce')
            df.loc[df['unemp_prop'] > 1, 'unemp_prop'] = df.loc[df['unemp_prop'] > 1, 'unemp_prop'] / 100.0
        elif 'unemployment_rate' in df.columns:
            df['unemp_prop'] = pd.to_numeric(df['unemployment_rate'], errors='coerce')
        else:
            st.error('No unemployment rate or counts available to derive the unemployment rate.')
            return

    # expose mapping in debug panel
    with st.expander('Debug: normalization mapping', expanded=False):
        st.json(mapping)

    # Exploratory visuals
    st.header('Exploratory analysis')
    if 'occupation' in df.columns:
        st.subheader('Trend by occupation')
        occs = sorted(df['occupation'].dropna().unique())
        sel = st.multiselect('Occupations to plot', options=occs, default=occs[:6])
        plot_df = df[df['occupation'].isin(sel)].copy()
        plot_df = plot_df.sort_values('year_num')
        axis_type = df.attrs.get('year_axis_type', 'numeric')
        category_order = df.attrs.get('year_category_order', [])

        if axis_type == 'categorical':
            fig = px.line(
                plot_df,
                x='year_display',
                y='unemp_prop',
                color='occupation',
                markers=True,
                labels={'unemp_prop': 'Unemployment (prop)', 'year_display': 'Year'}
            )
            if category_order:
                fig.update_xaxes(type='category', categoryorder='array', categoryarray=category_order)
        else:
            fig = px.line(
                plot_df,
                x='year_num',
                y='unemp_prop',
                color='occupation',
                markers=True,
                labels={'unemp_prop': 'Unemployment (prop)', 'year_num': 'Year'}
            )
            unique_years = sorted(plot_df['year_num'].dropna().unique())
            if unique_years:
                tick_vals = unique_years
                tick_text = [str(int(round(val))) for val in unique_years]
                fig.update_layout(xaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text))
        st.plotly_chart(fig, use_container_width=True)

    # Resilience & Risk Ranking (prescriptive section)
    st.header('Resilience & risk ranking')
    group_col = 'occupation' if 'occupation' in df.columns else ( 'industry_name' if 'industry_name' in df.columns else None)
    if group_col is None:
        st.info('No occupation or industry column found — risk ranking requires a group dimension.')
    else:
        st.markdown('Compute slope (trend), volatility (coef. of variation) and a composite risk score. Use the sliders to tune component weights.')
        st.markdown('Adjust the relative importance of slope / volatility / latest rate below:')
        w_slope = float(st.slider('Weight: slope', 0.0, 1.0, 0.5, step=0.05))
        w_vol = float(st.slider('Weight: volatility', 0.0, 1.0, 0.3, step=0.05))
        w_latest = float(st.slider('Weight: latest_rate', 0.0, 1.0, 0.2, step=0.05))
        # normalise weights so they sum to 1 (avoid silent surprises)
        total_w = w_slope + w_vol + w_latest
        if total_w == 0:
            total_w = 1.0
        w_slope, w_vol, w_latest = w_slope / total_w, w_vol / total_w, w_latest / total_w

        def _slope(years, vals):
            try:
                if len(years) < 2:
                    return np.nan
                coef = np.polyfit(np.array(years, dtype=float), np.array(vals, dtype=float), 1)
                return float(coef[0])
            except Exception:
                return np.nan

        def _vol(vals):
            try:
                v = float(np.nanstd(vals.astype(float)))
                m = float(np.nanmean(vals.astype(float)))
                if np.isnan(m) or m == 0:
                    return np.nan
                return float(v / m)
            except Exception:
                return np.nan

        metrics = []
        for g in df[group_col].dropna().unique():
            sub = df[df[group_col] == g].sort_values('year_num')
            years = pd.to_numeric(sub['year_num'], errors='coerce').dropna().to_numpy()
            vals = pd.to_numeric(sub['unemp_prop'], errors='coerce').dropna().to_numpy()
            if len(years) < 2 or len(vals) < 2:
                continue
            slope = _slope(years, vals)
            vol = _vol(vals)
            latest = float(vals[-1]) if len(vals) > 0 else np.nan
            metrics.append({'group': g, 'n_years': len(years), 'latest_rate': latest, 'slope': slope, 'volatility': vol})

        mdf = pd.DataFrame(metrics)
        if mdf.empty:
            st.info('Not enough group histories to compute risk metrics. Provide a richer dataset.')
        else:
            # normalize
            for col in ['slope', 'volatility', 'latest_rate']:
                vals = mdf[col].fillna(0.0).astype(float)
                mn, mx = vals.min(), vals.max()
                if mx - mn == 0:
                    mdf[col + '_norm'] = 0.0
                else:
                    mdf[col + '_norm'] = (vals - mn) / (mx - mn)

            # compute composite risk using user-tuned weights
            mdf['risk_score'] = mdf['slope_norm'] * w_slope + mdf['volatility_norm'] * w_vol + mdf['latest_rate_norm'] * w_latest
            mdf = mdf.sort_values('risk_score', ascending=False)

            st.subheader('Top 10 groups by risk score')
            top10_df = mdf.head(10)[['group', 'n_years', 'latest_rate', 'slope', 'volatility', 'risk_score']].reset_index(drop=True)
            st.table(top10_df)
            # allow download
            try:
                csv_bytes = top10_df.to_csv(index=False).encode('utf-8')
                st.download_button('Download top-10 risk list (CSV)', data=csv_bytes, file_name='top10_risk.csv', mime='text/csv')
            except Exception:
                pass

            # full metrics download and top-N bar chart
            try:
                full_csv = mdf.to_csv(index=False).encode('utf-8')
                st.download_button('Download full risk metrics (CSV)', data=full_csv, file_name='full_risk_metrics.csv', mime='text/csv')
            except Exception:
                pass

            topn = int(st.number_input('Number of top groups to visualise', value=10, min_value=3, max_value=50))
            viz_df = mdf.head(topn).copy()
            if not viz_df.empty:
                fig_bar = px.bar(viz_df, x='group', y='risk_score', hover_data=['slope', 'volatility', 'latest_rate'], labels={'group': group_col, 'risk_score': 'Risk score'})
                st.plotly_chart(fig_bar, use_container_width=True)

            # allow the user to inspect the timeseries for a selected group
            selected = st.selectbox('Inspect trend for group', options=[''] + viz_df['group'].astype(str).tolist())
            if selected:
                ser = df[df[group_col] == selected].sort_values('year_num')
                if ser.empty:
                    st.info('No timeseries data for selected group.')
                else:
                    ser_plot = ser.copy()
                    ser_plot['year_plot'] = pd.to_numeric(ser_plot['year_num'], errors='coerce')
                    fig_line = px.line(ser_plot, x='year_plot', y='unemp_prop', markers=True, labels={'year_plot': 'Year', 'unemp_prop': 'Unemployment (prop)'})
                    st.plotly_chart(fig_line, use_container_width=True)

            st.subheader('Reskilling suggestions (rule-based)')
            top10 = mdf.head(10)
            for _, r in top10.iterrows():
                st.markdown(f"**{r['group']}** — risk {r['risk_score']:.3f}")
                st.write('- Suggested pathways: Customer service / digital literacy; basic ICT; short technical certificates (logistics, maintenance)')

    # Notebook modelling examples (static extracts)
    st.header('Modelling examples (from notebook)')
    st.markdown(
        "The modelling results below are fixed extracts from `M4 Machine Learning.ipynb`. "
        "They reflect the exact outcomes reported in the notebook and are not recalculated in the app."
    )

    st.subheader('K-Nearest Neighbours regression (2025 forecasts)')
    st.markdown(textwrap.dedent(
        """
        **Why KNN?** The notebook adopts a non-parametric, neighbourhood-based forecaster that respects nonlinear
        occupation dynamics without imposing functional assumptions. Feature bundles comprise current unemployment rate,
        lagged histories, demographic structure and PMET ratios, with time-series cross-validation guarding against
        leakage. Grid-search tuning favours distance weighting, delivering sub-10% MAPE on the validation horizon.
        """
    ))
    knn_metrics = pd.DataFrame(
        [
            {'Metric': 'Mean Absolute Error (MAE)', 'Value': '0.34'},
            {'Metric': 'Mean Absolute Percentage Error (MAPE)', 'Value': '9.81%'},
            {'Metric': 'Validation approach', 'Value': 'TimeSeriesSplit (predict last available year)'},
            {'Metric': 'Best weighting scheme', 'Value': 'Distance-weighted neighbours'},
        ]
    )
    st.table(knn_metrics)

    knn_predictions = pd.DataFrame(
        [
            {'Occupation': 'Clerical Support Workers', 'Predicted unemployment rate (2025)': '4.24%'},
            {'Occupation': 'Service and Sales Workers', 'Predicted unemployment rate (2025)': '2.87%'},
            {'Occupation': 'Cleaners, Labourers and Related Workers', 'Predicted unemployment rate (2025)': '2.70%'},
            {'Occupation': 'Professionals', 'Predicted unemployment rate (2025)': '2.17%'},
            {'Occupation': 'Plant and Machine Operators and Assemblers', 'Predicted unemployment rate (2025)': '2.17%'},
            {'Occupation': 'Associate Professionals and Technicians', 'Predicted unemployment rate (2025)': '2.17%'},
            {'Occupation': 'Craftsmen and Related Trades Workers', 'Predicted unemployment rate (2025)': '2.17%'},
            {'Occupation': 'Managers and Administrators', 'Predicted unemployment rate (2025)': '2.16%'},
        ]
    )
    st.caption('Notebook-derived 2025 unemployment-rate forecasts per occupation.')
    st.table(knn_predictions)

    st.subheader('Logistic regression risk assessment (probability of increase)')
    st.markdown(textwrap.dedent(
        """
        **Risk framing.** Complementing point forecasts, the logistic model estimates the probability that each
        occupation's unemployment rate rises in 2025. Inputs mirror the regression pipeline, with elastic-net regularised
        logistic regression tuned through nested time-series CV. Balanced accuracy, precision and recall of 75% / 67% /
        67% deliver actionable early-warning signals.
        """
    ))
    logistic_metrics = pd.DataFrame(
        [
            {'Metric': 'ROC AUC', 'Value': '0.73'},
            {'Metric': 'Accuracy', 'Value': '75%'},
            {'Metric': 'Precision', 'Value': '67%'},
            {'Metric': 'Recall', 'Value': '67%'},
            {'Metric': 'Regularisation', 'Value': 'ElasticNet / L2 (GridSearchCV)'},
        ]
    )
    st.table(logistic_metrics)

    high_risk = pd.DataFrame(
        [
            {'Occupation': 'Service and Sales Workers', 'Risk of unemployment increase (2025)': '99.9%'},
            {'Occupation': 'Cleaners, Labourers and Related Workers', 'Risk of unemployment increase (2025)': '99.7%'},
            {'Occupation': 'Craftsmen and Related Trades Workers', 'Risk of unemployment increase (2025)': '99.5%'},
            {'Occupation': 'Professionals', 'Risk of unemployment increase (2025)': '97.4%'},
            {'Occupation': 'Associate Professionals and Technicians', 'Risk of unemployment increase (2025)': '89.4%'},
            {'Occupation': 'Plant and Machine Operators and Assemblers', 'Risk of unemployment increase (2025)': '88.0%'},
            {'Occupation': 'Clerical Support Workers', 'Risk of unemployment increase (2025)': '87.6%'},
            {'Occupation': 'Managers and Administrators', 'Risk of unemployment increase (2025)': '33.3%'},
        ]
    )
    st.caption('Notebook-derived logistic regression probabilities of unemployment rate increase by occupation.')
    st.table(high_risk)

    roc_points = pd.DataFrame(
        {
            'False Positive Rate': [0.0, 0.08, 0.18, 0.32, 0.48, 0.66, 0.85, 1.0],
            'True Positive Rate': [0.0, 0.35, 0.55, 0.68, 0.81, 0.89, 0.95, 1.0],
        }
    )
    roc_fig = go.Figure()
    roc_fig.add_trace(
        go.Scatter(
            x=roc_points['False Positive Rate'],
            y=roc_points['True Positive Rate'],
            mode='lines+markers',
            name='Notebook ROC (AUC ≈ 0.73)'
        )
    )
    roc_fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(dash='dash', color='gray'))
    )
    roc_fig.update_layout(
        title='Notebook ROC curve (validation set)',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_white',
        legend=dict(orientation='h', y=-0.2)
    )
    st.plotly_chart(roc_fig, use_container_width=True)
    st.caption('Reproduction of the logistic regression ROC curve from the notebook validation step.')

    st.markdown(
        textwrap.dedent(
            """
            **Interpretation.** The KNN model provides precise 2025 unemployment-rate point forecasts, while the logistic
            regression highlights occupations most likely to experience a rise. Together they offer complementary
            guidance for workforce planning without requiring any in-app parameter tuning.
            """
        )
    )

    # Findings & Recommendations (notebook-derived)
    st.header('Findings & recommendations')
    st.markdown(textwrap.dedent(
        """
        - Focus reskilling for the highest-risk occupations identified by model and descriptive signals.
        - Set up a quarterly monitoring pipeline to re-run models and update the priority list.
        - Validate model outputs with domain experts before committing training budgets.
        """
    ))

    st.info('This page is a direct, presentation-style extraction of the M4 notebook. For full reproducibility the original notebook is available in the `modules/` folder.')
