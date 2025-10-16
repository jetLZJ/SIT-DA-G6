import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, Iterable, cast
import sqlalchemy

from app import data_loader, utils, viz


SESSION_DF_KEY = 'module23_clean_df'
SESSION_TABLE_KEY = 'module23_table_name'
SESSION_LONG_TABLES_KEY = 'module23_long_tables'

PREFERRED_LONG_TABLES = [
    'unemployment_rate_by_occupation_long',
    'unemployed_by_previous_occupation_sex_long',
    'unemployed_by_age_sex_long',
    'unemployed_by_qualification_sex_long',
]


def _normalize_and_compute_rates(df_in: pd.DataFrame):
    """Return a DataFrame with canonical columns where possible and a mapping of detected names."""
    col_map = {}
    for c in df_in.columns:
        n = c.lower().strip()
        n = n.replace(' ', '_').replace('-', '_')
        n = n.replace('%', 'pct').replace('(', '').replace(')', '')
        col_map[n] = c

    # Heuristics for count columns
    unemployed_cands = ['unemployed_count', 'unemployed', 'unemp_count']
    laborforce_cands = ['labor_force_count', 'labour_force_count', 'laborforce_count', 'labor_force', 'labour_force']

    orig_unemployed = next((col_map[k] for k in unemployed_cands if k in col_map), None)
    orig_laborforce = next((col_map[k] for k in laborforce_cands if k in col_map), None)

    df_work = df_in.copy()
    mapping: dict = {'original_columns': list(df_in.columns)}
    mapping['detected_unemployed_col'] = orig_unemployed
    mapping['detected_laborforce_col'] = orig_laborforce
    mapping['derived_unemployment_rate'] = False

    # If we can compute from counts, use the helper (which is robust)
    if orig_unemployed and orig_laborforce:
        try:
            df_work = utils.compute_unemployment_rate(df_work, unemployed_col=orig_unemployed, laborforce_col=orig_laborforce)
            mapping['derived_unemployment_rate'] = True
        except Exception as e:
            mapping['derive_error'] = str(e)

    # If we still don't have a rate column, try to detect any precomputed variants
    detected_rate = ''
    for candidate in ['unemployment_rate', 'unemployed_rate', 'unemployed_rate_pct', 'unemployment_rate_pct', 'unemp_rate']:
        if candidate in col_map:
            detected_rate = col_map[candidate]
            break

    mapping['detected_rate_column'] = detected_rate if detected_rate else None
    # Also try to normalise common dimension columns to canonical names used in the page
    # Occupation variants
    occ_cands = ['occupation', 'occupation_name', 'occupation_title', 'job_title', 'occupation_group']
    for oc in occ_cands:
        if oc in col_map:
            df_work.rename(columns={col_map[oc]: 'occupation'}, inplace=True)
            mapping['renamed_occupation_from'] = col_map[oc]
            break

    # Year variants
    year_cands = ['year', 'yr', 'period', 'date']
    for yc in year_cands:
        if yc in col_map:
            df_work.rename(columns={col_map[yc]: 'year'}, inplace=True)
            mapping['renamed_year_from'] = col_map[yc]
            break

    # If a detected rate column exists, rename to 'unemployed_rate' for page compatibility
    if detected_rate:
        df_work.rename(columns={detected_rate: 'unemployed_rate' if detected_rate != 'unemployment_rate' else 'unemployment_rate'}, inplace=True)
        mapping['renamed_rate_from'] = detected_rate

    # If we derived unemployment_rate via counts, ensure both names are present
    if mapping.get('derived_unemployment_rate'):
        # compute_unemployment_rate creates 'unemployment_rate' and 'unemployed_rate'
        pass

    return df_work, mapping


def _ensure_year_int(df_in: pd.DataFrame):
    """Ensure there's a numeric integer year column called 'year_yr' for plotting."""

    df_out = df_in.copy()
    recovered = False
    if 'year' in df_out.columns:
        col = df_out['year']
        # If already datetime-like
        if pd.api.types.is_datetime64_any_dtype(col) or pd.api.types.is_datetime64_dtype(col):
            years = col.dt.year
            # If parsing produced sensible years (not everything 1970), use them
            if years.notna().any() and not years.dropna().eq(1970).all():
                df_out['year_yr'] = years.astype(float)
            else:
                # Possible nanosecond interpretation: underlying int values are small (e.g., 2014)
                try:
                    ns = col.view('int64')
                except Exception:
                    try:
                        ns = col.astype('int64')
                    except Exception:
                        ns = pd.Series([pd.NA] * len(col))

                if isinstance(ns, (pd.Series,)) and ns.notna().any():
                    # If max raw value is small (< 10 million), it's likely the original year values (e.g., 2014)
                    max_ns = int(ns.max()) if ns.max() is not pd.NaT else None
                    if max_ns is not None and max_ns < 10_000_000:
                        df_out['year_yr'] = ns.astype('Int64').astype(float)
                        recovered = True
                    else:
                        # Fallback to year extraction (will be 1970 if that's what dt.year gave)
                        df_out['year_yr'] = years.astype(float)
                else:
                    df_out['year_yr'] = years.astype(float)
        else:
            # Not datetime-like. Try numeric coercion first (handles ints stored as object)
            numeric = pd.to_numeric(col, errors='coerce')
            if numeric.notna().any():
                df_out['year_yr'] = numeric.round().astype(float)
            else:
                # As a last resort, try parsing strings to datetime then extract year
                parsed = pd.to_datetime(col, errors='coerce')
                if parsed.notna().any():
                    df_out['year_yr'] = parsed.dt.year.astype(float)

    if 'year_yr' in df_out.columns:
        # final safety: ensure numeric dtype
        df_out['year_yr'] = pd.to_numeric(df_out['year_yr'], errors='coerce').astype(float)

    if recovered:
        df_out.attrs['year_recovered'] = True

    return df_out


def _ensure_year_datetime(df_in: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add a `year_dt` datetime column wherever possible and capture conversion metadata."""

    df_out = df_in.copy()
    info: dict[str, object] = {
        'source_column': None,
        'source_dtype': None,
        'conversion_status': 'skipped',
        'non_null_converted': 0,
    }

    year_series = None
    source_col = None
    if 'year' in df_out.columns:
        source_col = 'year'
        year_series = df_out['year']
    elif 'year_yr' in df_out.columns:
        source_col = 'year_yr'
        year_series = df_out['year_yr']

    if year_series is not None:
        info['source_column'] = source_col
        info['source_dtype'] = str(year_series.dtype)
        if pd.api.types.is_datetime64_any_dtype(year_series) or pd.api.types.is_datetime64_dtype(year_series):
            converted = pd.to_datetime(year_series, errors='coerce')
        else:
            year_numeric = pd.to_numeric(year_series, errors='coerce')
            if year_numeric.notna().any():
                converted = pd.to_datetime(year_numeric.round().astype('Int64'), format='%Y', errors='coerce')
            else:
                converted = pd.to_datetime(year_series, errors='coerce')
        df_out['year_dt'] = converted
        info['conversion_status'] = 'success' if converted.notna().any() else 'no_valid_rows'
        info['non_null_converted'] = int(converted.notna().sum())
    else:
        info['conversion_status'] = 'no_source'

    return df_out, info


def _find_column(df: pd.DataFrame, keywords: Iterable[str]) -> Optional[str]:
    """Return the first column whose lowercase name contains any of the keywords."""
    lowered = {col.lower(): col for col in df.columns}
    for key in keywords:
        key_lower = key.lower()
        for col_lower, original in lowered.items():
            if key_lower in col_lower:
                return original
    return None


def prepare_demographic_share(
    df: pd.DataFrame,
    dimension_keywords: Iterable[str],
    *,
    collapse_gender: bool = False,
) -> tuple[pd.DataFrame, str, str, str, str]:
    """Normalise a long-format table so that each row expresses the percentage share per demographic dimension."""

    df_work = df.copy()
    year_col = _find_column(df_work, ['year'])
    occ_col = _find_column(df_work, ['occupation'])
    dim_col = _find_column(df_work, dimension_keywords)
    count_col = _find_column(df_work, ['unemployed_count', 'unemployment_count', 'unemp_count'])

    if not all([year_col, dim_col, count_col]):
        raise KeyError('Missing required columns for share computation.')

    if not occ_col:
        temp_occ_col = '__occupation__'
        df_work[temp_occ_col] = 'Overall'
        occ_col = temp_occ_col

    year_col = cast(str, year_col)
    dim_col = cast(str, dim_col)
    count_col = cast(str, count_col)
    occ_col = cast(str, occ_col)

    if collapse_gender:
        gender_col = _find_column(df_work, ['gender', 'sex'])
        if gender_col:
            group_cols = [col for col in [year_col, occ_col, dim_col] if col]
            df_work = df_work.groupby(group_cols, as_index=False)[count_col].sum()

    if year_col:
        if pd.api.types.is_datetime64_any_dtype(df_work[year_col]):
            df_work[year_col] = df_work[year_col].dt.year
        else:
            df_work[year_col] = pd.to_numeric(df_work[year_col], errors='coerce')

    df_work[count_col] = pd.to_numeric(df_work[count_col], errors='coerce')
    df_work = df_work.dropna(subset=[year_col, dim_col, count_col])  # type: ignore[arg-type]

    group_cols = [col for col in [year_col, occ_col] if col]
    totals = df_work.groupby(group_cols)[count_col].transform('sum')
    df_work = df_work[totals > 0].copy()
    df_work['share_pct'] = (df_work[count_col] / totals.loc[df_work.index]) * 100.0
    df_work = df_work.dropna(subset=['share_pct'])

    return df_work, year_col, occ_col, dim_col, count_col


def load_long_wide_from_db(engine: sqlalchemy.engine.Engine) -> tuple[dict, dict]:
    """Load long and wide tables into dicts (table_name -> DataFrame)."""
    inspector = sqlalchemy.inspect(engine)
    all_tables = inspector.get_table_names()
    long_tables = [t for t in all_tables if t.endswith('long')]
    wide_tables = [t for t in all_tables if t.endswith('wide')]
    df_long_dict = {t: pd.read_sql(f"SELECT * FROM {t}", engine) for t in long_tables}
    df_wide_dict = {t: pd.read_sql(f"SELECT * FROM {t}", engine) for t in wide_tables}
    return df_long_dict, df_wide_dict


def _default_table_index(table_names: list[str], preferred: Optional[str]) -> int:
    if not table_names:
        return 0
    if preferred and preferred in table_names:
        return table_names.index(preferred)
    for candidate in PREFERRED_LONG_TABLES:
        if candidate in table_names:
            return table_names.index(candidate)
    return 0


def _set_active_dataframe(df: pd.DataFrame, table_name: str):
    st.session_state[SESSION_DF_KEY] = df.copy()
    st.session_state[SESSION_TABLE_KEY] = table_name


def _get_long_tables(engine: Optional[sqlalchemy.engine.Engine], *, show_uploader: bool) -> dict[str, pd.DataFrame]:
    tables = st.session_state.get(SESSION_LONG_TABLES_KEY)
    if tables is None or not tables:
        tables = {}
        if engine is not None:
            try:
                tables, _ = load_long_wide_from_db(engine)
                if tables:
                    st.success(f"Loaded {len(tables)} long-format tables from database.")
            except Exception as exc:
                st.error(f'Failed to load tables from database: {exc}')
        st.session_state[SESSION_LONG_TABLES_KEY] = tables

    if show_uploader:
        uploaded_files = st.file_uploader(
            'Upload additional long-format CSVs',
            accept_multiple_files=True,
            key='module23_long_upload'
        )
        if uploaded_files:
            tables = dict(st.session_state.get(SESSION_LONG_TABLES_KEY, {}))
            added = []
            for uploaded in uploaded_files:
                name = Path(uploaded.name).stem
                try:
                    tables[name] = pd.read_csv(uploaded)
                    added.append(name)
                except Exception:
                    st.warning(f'Failed to read {uploaded.name}')
            st.session_state[SESSION_LONG_TABLES_KEY] = tables
            if added:
                st.success(f"Loaded {len(added)} uploaded table(s): {', '.join(added)}")

    return st.session_state.get(SESSION_LONG_TABLES_KEY, {})


def _get_active_dataframe(engine: Optional[sqlalchemy.engine.Engine], *, allow_refresh: bool) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    cached_df = st.session_state.get(SESSION_DF_KEY)
    cached_table = st.session_state.get(SESSION_TABLE_KEY)
    if isinstance(cached_df, pd.DataFrame) and cached_table:
        return cached_df.copy(), cached_table

    tables = _get_long_tables(engine, show_uploader=allow_refresh)
    if not tables:
        return None, None

    table_names = sorted(tables.keys())
    chosen_index = _default_table_index(table_names, cached_table)
    chosen_table = table_names[chosen_index]
    df_raw = tables[chosen_table]
    df_clean, _ = _normalize_and_compute_rates(df_raw)
    df_clean = _ensure_year_int(df_clean)
    _set_active_dataframe(df_clean, chosen_table)
    return df_clean.copy(), chosen_table


def _select_rate_column(df: pd.DataFrame) -> Optional[str]:
    if 'unemployment_rate' in df.columns:
        return 'unemployment_rate'
    if 'unemployed_rate' in df.columns:
        return 'unemployed_rate'
    return None


def page_cleaning_module_two(engine: Optional[sqlalchemy.engine.Engine]):
    st.title('Module 2 — Data cleaning & checking')

    tables = _get_long_tables(engine, show_uploader=False)
    if not tables:
        st.info('No long-format tables available. Connect to the project database or upload CSVs to continue.')
        return

    table_names = sorted(tables.keys())
    default_index = _default_table_index(table_names, st.session_state.get(SESSION_TABLE_KEY))
    selected_table = table_names[default_index]

    df_raw = tables[selected_table]
    df_clean, mapping = _normalize_and_compute_rates(df_raw)
    df_clean = _ensure_year_int(df_clean)
    df_clean, year_conversion_info = _ensure_year_datetime(df_clean)
    _set_active_dataframe(df_clean, selected_table)
    st.session_state['module23_column_mapping'] = mapping
    st.session_state['module23_year_conversion'] = year_conversion_info

    outlier_table_options = sorted(tables.keys())
    default_outlier_table = st.session_state.get('module23_outlier_table', selected_table)
    default_outlier_index = _default_table_index(outlier_table_options, default_outlier_table)
    outlier_table = st.selectbox(
        'Dataset reference for quality checks',
        options=outlier_table_options,
        index=default_outlier_index,
        key='module23_outlier_table'
    )

    with st.expander('Step 1 — Data health checks', expanded=False):
        info_col, missing_col = st.columns(2)
        with info_col:
            st.markdown('**Data types**')
            st.dataframe(df_clean.dtypes.astype(str).rename('dtype'))
        with missing_col:
            st.markdown('**Missing values**')
            st.dataframe(df_clean.isnull().sum().rename('missing_count'))

        numeric_df = df_clean.select_dtypes(include='number')
        if not numeric_df.empty:
            stats_col, dup_col = st.columns(2)
            with stats_col:
                st.markdown('**Descriptive statistics (numeric)**')
                st.dataframe(numeric_df.describe().T)
            with dup_col:
                st.markdown('**Duplicate rows**')
                dup_count = int(df_clean.duplicated().sum())
                st.metric('Total duplicates', dup_count)
                if dup_count:
                    st.dataframe(df_clean[df_clean.duplicated()].head(), use_container_width=True)

    with st.expander('Step 2 — Convert year from float to datetime', expanded=False):
        conversion_meta = st.session_state.get('module23_year_conversion', {})
        source_column = conversion_meta.get('source_column')
        if source_column is None:
            st.info('No year-like column detected yet. Load a dataset containing `year` or `year_yr`.')
        elif 'year_dt' not in df_clean.columns:
            st.warning('Conversion metadata is present but the `year_dt` column is missing. Re-run Module 2 loading step.')
        else:
            st.caption(
                f"Converted `{source_column}` ({conversion_meta.get('source_dtype')}) into `year_dt` (datetime64)."
            )
            preview = pd.DataFrame({
                source_column: df_clean[source_column].head(),
                'year_dt': df_clean['year_dt'].head()
            })
            st.dataframe(preview, use_container_width=True)
            st.metric('Rows with valid datetime', int(conversion_meta.get('non_null_converted', 0)))
            if df_clean['year_dt'].isna().any():
                st.warning('Some rows remain without a valid datetime. Inspect the source data for malformed years.')
            else:
                st.success('All rows now include a datetime representation for year.')

    with st.expander('Step 3 — Outlier discovery across long tables', expanded=False):
        outlier_raw = tables[outlier_table]
        outlier_df, _ = _normalize_and_compute_rates(outlier_raw)
        outlier_df = _ensure_year_int(outlier_df)
        numeric_cols = outlier_df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()
        unemployment_cols = [col for col in numeric_cols if col.lower() in {'unemployed_count', 'unemployment_count', 'unemployed_rate'}]

        if unemployment_cols:
            pick_col = st.selectbox('Numeric column to profile', options=unemployment_cols, key='module23_outlier_column')
            series = pd.to_numeric(outlier_df[pick_col], errors='coerce').dropna()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = series[(series < lower) | (series > upper)]

            left_col, right_col = st.columns(2)
            with left_col:
                hist_fig = px.histogram(outlier_df, x=pick_col, nbins=30, title=f'Histogram — {pick_col}')
                hist_fig.update_layout(margin=dict(t=40, r=20, l=20, b=40))
                utils.render_plotly_chart(hist_fig, key='module2_histogram')
            with right_col:
                box_fig = px.box(outlier_df, y=pick_col, title=f'Box plot — {pick_col}')
                box_fig.update_layout(margin=dict(t=40, r=20, l=20, b=40))
                utils.render_plotly_chart(box_fig, key='module2_boxplot')

            if not outliers.empty:
                st.markdown(f"**Outlier range**: values < {lower:,.2f} or > {upper:,.2f}")
                st.dataframe(outliers.sort_values().to_frame(name=pick_col).head(50), use_container_width=True)
            else:
                st.caption('No IQR-based outliers detected for the selected column.')
        else:
            st.info('Selected table has no unemployment count column available for outlier inspection.')


def render_employed_count_feature(engine: Optional[sqlalchemy.engine.Engine]):
    df_active, table_name = _get_active_dataframe(engine, allow_refresh=False)
    if df_active is None or table_name is None:
        st.info('Load a dataset in Module 2 first to compute the employed_count example.')
        return

    rate_col = _select_rate_column(df_active)
    st.caption(f'Deriving employed_count using **{table_name}**')

    if {'unemployed_count', 'occupation'}.issubset(df_active.columns) and rate_col:
        temp = df_active[['year', 'occupation', 'unemployed_count', rate_col]].dropna().copy()
        if rate_col == 'unemployment_rate':
            temp['unemployed_rate_prop'] = temp[rate_col].astype(float)
            temp['unemployed_rate_display'] = temp[rate_col] * 100.0
        else:
            temp['unemployed_rate_prop'] = temp[rate_col].astype(float) / 100.0
            temp['unemployed_rate_display'] = temp[rate_col].astype(float)

        mask = temp['unemployed_rate_prop'].notna() & (temp['unemployed_rate_prop'] > 0)
        temp.loc[mask, 'employed_count'] = temp.loc[mask, 'unemployed_count'] * (1.0 / temp.loc[mask, 'unemployed_rate_prop'] - 1.0)
        st.dataframe(temp[['year', 'occupation', 'unemployed_count', 'unemployed_rate_display', 'employed_count']].head(), use_container_width=True)
    else:
        st.info('Active dataset is missing the required columns to compute employed_count.')


def page_visualisation_module_three(engine: Optional[sqlalchemy.engine.Engine]):
    st.title('Module 3 — Visual storytelling & diagnostics')

    tables = _get_long_tables(engine, show_uploader=False)
    df_active, table_name = _get_active_dataframe(engine, allow_refresh=False)

    if df_active is None or table_name is None:
        st.info('Load a dataset in Module 2 first, or ensure a database connection is available.')
        return

    df_active = _ensure_year_int(df_active.copy())
    rate_col = _select_rate_column(df_active)
    if not rate_col:
        st.warning('Active dataset has no unemployment rate column after normalisation.')
        return

    if 'occupation' not in df_active.columns:
        st.warning('Active dataset is missing an `occupation` column required for Module 3 visuals.')
        return

    st.markdown(
        """
        ### Trend lens — Unemployment trajectories across occupation groups
        This lens explores decade-long unemployment patterns to identify persistent structural pressures and cyclical shocks across occupations. The visual below tracks trajectories for eight main occupation families from 2014 to 2024. 
        The COVID-19 period (2020-2021) marked a sharp spike across all occupations, with lower-skilled groups such as Clerical Support Workers and Service & Sales Workers experiencing the most severe disruptions. Despite partial recovery post-2021, these roles maintained the highest average unemployment rates, highlighting their structural vulnerability compared to higher-skilled roles such as Professionals and Managers.

        """
    )

    occupations = sorted(df_active['occupation'].dropna().unique().tolist())

    st.markdown('#### Occupation trajectories')
    if occupations:
        pick_mode = st.radio(
            'Occupation selection strategy',
            options=['Top by average unemployment rate', 'Manual selection'],
            key='module23_trend_mode'
        )
        if pick_mode.startswith('Top'):
            max_slider = max(3, min(20, len(occupations)))
            topn = st.slider('Top N occupations', min_value=3, max_value=max_slider, value=min(8, max_slider), key='module23_trend_topn')
            top_occ = (
                df_active.groupby('occupation')[rate_col]
                .mean()
                .nlargest(topn)
                .index.tolist()
            )
            trend_df = df_active[df_active['occupation'].isin(top_occ)]
        else:
            default_selection = occupations[: min(6, len(occupations))]
            selected = st.multiselect('Pick occupations', options=occupations, default=default_selection, key='module23_trend_manual')
            trend_df = df_active[df_active['occupation'].isin(selected)]

        if not trend_df.empty:
            plot_df = _ensure_year_int(trend_df.copy())
            plot_df['plot_unemp_pct'] = plot_df[rate_col] * (100.0 if rate_col == 'unemployment_rate' else 1.0)

            fig = px.line(plot_df, x='year_yr', y='plot_unemp_pct', color='occupation', markers=True, title='Unemployment rate by occupation')
            fig.update_yaxes(title='Unemployment rate (%)')
            try:
                min_year = int(plot_df['year_yr'].min())
                fig.update_xaxes(tickmode='linear', tick0=min_year, dtick=1)
            except Exception:
                pass
            try:
                fig.add_vrect(
                    x0=2019.5,
                    x1=2021.5,
                    fillcolor='rgba(255, 165, 0, 0.15)',
                    line_width=0,
                    annotation_text='COVID shock (2020-2021)',
                    annotation_position='top left',
                )
            except Exception:
                pass
            utils.render_plotly_chart(fig, key='module3_trend_line')
            try:
                latest_year = plot_df['year_yr'].max()
                latest_snapshot = plot_df[plot_df['year_yr'] == latest_year]
                if not latest_snapshot.empty:
                    top_row = latest_snapshot.loc[latest_snapshot['plot_unemp_pct'].idxmax()]
                    bottom_row = latest_snapshot.loc[latest_snapshot['plot_unemp_pct'].idxmin()]
                    st.markdown(
                        f"*{latest_year} snapshot:* **{top_row['occupation']}** led the unemployment rate at {top_row['plot_unemp_pct']:.1f}% while **{bottom_row['occupation']}** was lowest among the selected occupations at {bottom_row['plot_unemp_pct']:.1f}%."
                    )
            except Exception:
                st.caption('Latest-year summary unavailable due to inconsistent occupation data.')
            st.caption("The trajectories reveal that lower-skilled occupations such as Clerical Support Workers and Service & Sales Workers consistently record higher unemployment rates, validating the hypothesis that lower-skilled roles face greater vulnerability.")
        else:
            st.info('No rows available for the selected occupations.')
    else:
        st.info('No occupation values available for trend analysis.')
    

    st.markdown('#### Share of unemployment burden')
    share_df = _ensure_year_int(df_active.copy())
    if 'year_yr' in share_df.columns:
        share_df['plot_unemp_pct'] = share_df[rate_col] * (100.0 if rate_col == 'unemployment_rate' else 1.0)
        pivot_sum = share_df.pivot_table(index='year_yr', columns='occupation', values='plot_unemp_pct', aggfunc='sum').fillna(0)
        row_totals = pivot_sum.sum(axis=1)
        non_zero = row_totals.replace(0, pd.NA)
        prop = pivot_sum.divide(non_zero, axis=0).dropna(how='all')
        if prop.empty:
            st.info('Not enough non-zero data to compute share of unemployment.')
        else:
            area_df = prop.reset_index().rename(columns={'year_yr': 'year'})
            fig_share = px.area(area_df, x='year', y=area_df.columns[1:], title='Share of unemployment by occupation')
            try:
                min_year_share = int(area_df['year'].min())
                fig_share.update_xaxes(tickmode='linear', tick0=min_year_share, dtick=1)
            except Exception:
                pass
            try:
                fig_share.add_vrect(
                    x0=2019.5,
                    x1=2021.5,
                    fillcolor='rgba(255, 165, 0, 0.15)',
                    line_width=0,
                    row='all',
                    col='all',
                )
            except Exception:
                pass
            utils.render_plotly_chart(fig_share, key='module3_share_area')
            try:
                latest_share_year = area_df['year'].max()
                latest_row = area_df[area_df['year'] == latest_share_year].iloc[0, 1:]
                top_occ = latest_row.idxmax()
                top_share = latest_row.max() * 100 if latest_row.max() <= 1 else latest_row.max()
                st.markdown(
                    f"*{latest_share_year} mix:* **{top_occ}** carried the largest unemployment burden, accounting for approximately {top_share:.1f}% of total unemployed workers."
                )
            except Exception:
                st.caption('Share breakdown summary unavailable because the latest year record is incomplete.')
            st.caption('This concentration of unemployment within Clerical Support and Service & Sales occupations underscores the structural persistence of job insecurity among lower-skilled roles.')
    else:
        st.info('Year information is missing; unable to compute share of unemployment.')

    st.markdown('---')
    st.markdown(
        """
        ### Human capital lens — demographic mediators of unemployment risk
        This lens examines how education, gender, and age interact with occupation groups to influence unemployment risk. It highlights demographic profiles driving vulnerability and identifies where targeted policy interventions could deliver the greatest impact.
        """
    )

    education_raw = tables.get('unemployed_by_qualification_sex_long') if tables else None
    st.markdown('#### Education tiers within occupation families')
    if education_raw is not None:
        education_grouped = education_raw.groupby(['year', 'education'])['unemployed_count'].sum().reset_index()
        if pd.api.types.is_datetime64_any_dtype(education_grouped['year']):
            education_grouped['year'] = education_grouped['year'].dt.year
        education_grouped['total_by_year'] = education_grouped.groupby('year')['unemployed_count'].transform('sum')
        education_grouped = education_grouped[education_grouped['total_by_year'] > 0].copy()
        education_grouped['share_pct'] = (education_grouped['unemployed_count'] / education_grouped['total_by_year']) * 100

        edu_order = [
            'Below Secondary',
            'Secondary',
            'Post-Secondary (Non-Tertiary)',
            'Diploma & Professional Qualification',
            'Degree',
        ]
        colors = px.colors.qualitative.Bold
        fig_education = px.area(
            education_grouped,
            x='year',
            y='share_pct',
            color='education',
            color_discrete_map={edu: colors[i % len(colors)] for i, edu in enumerate(edu_order)},
            category_orders={'education': edu_order},
            labels={'year': 'Year', 'share_pct': 'Share of unemployed (%)', 'education': 'Education tier'},
            title='Education tiers driving unemployment share over time',
            height=650,
            hover_data={'unemployed_count': ':.1f', 'share_pct': ':.1f'}
        )
        fig_education.update_yaxes(range=[0, 100])
        fig_education.update_layout(legend_title_text='Education tier', hovermode='x unified', legend=dict(traceorder='reversed'))
        try:
            fig_education.add_vrect(
                x0=2019.5,
                x1=2021.5,
                fillcolor='rgba(255, 165, 0, 0.15)',
                line_width=0,
                row='all',
                col='all',
            )
        except Exception:
            pass

        utils.render_plotly_chart(fig_education, key='module3_education_area')
        try:
            edu_latest_year = education_grouped['year'].max()
            edu_latest = education_grouped[education_grouped['year'] == edu_latest_year]
            if not edu_latest.empty:
                top_tier = edu_latest.loc[edu_latest['share_pct'].idxmax()]
                st.markdown(
                    f"*{edu_latest_year} education profile:* **{top_tier['education']}** contributed the highest share of unemployment at {top_tier['share_pct']:.1f}% across all tiers."
                )
        except Exception:
            st.caption('Education-tier summary unavailable because the dataset lacks a consistent year field.')

        st.caption('The rise in unemployment across higher education tiers across the years reflects structural shifts where education alone no longer insulates workers from job loss, particularly among professional and technical occupations.')
    else:
        st.info('Qualification-level unemployment table not available in the current data connection.')

    st.markdown('#### Gender exposure within occupation families')
    try:
        gender_raw = tables.get('unemployed_by_previous_occupation_sex_long') if tables else None
        if gender_raw is None:
            raise KeyError('Table unavailable')
        gender_df, gen_year, gen_occ, gen_dim, gen_count = prepare_demographic_share(gender_raw, ['gender', 'sex'])
        top_gender_occupations = (
            gender_df.groupby(gen_occ)[gen_count]
            .sum()
            .sort_values(ascending=False)
            .head(6)
            .index
        )
        gender_focus = gender_df[gender_df[gen_occ].isin(top_gender_occupations)].copy()
        if gender_focus.empty:
            st.info('Gender table lacks sufficient occupation-level detail for plotting.')
        else:
            fig_gender = px.area(
                gender_focus,
                x=gen_year,
                y='share_pct',
                color=gen_dim,
                facet_col=gen_occ,
                facet_col_wrap=3,
                category_orders={gen_dim: sorted(gender_focus[gen_dim].unique())},
                color_discrete_map={'Female': 'pink', 'Male': 'blue'},
                labels={gen_year: 'Year', 'share_pct': 'Share of unemployed (%)', gen_dim: 'Gender'},
                title='Gender share of unemployment within top occupations',
                height=650,
            )
            fig_gender.update_yaxes(matches=None, range=[0, 100])
            fig_gender.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
            fig_gender.update_layout(legend_title_text='Gender', hovermode='x unified')
            try:
                fig_gender.add_vrect(
                    x0=2019.5,
                    x1=2021.5,
                    fillcolor='rgba(255, 165, 0, 0.15)',
                    line_width=0,
                    row='all',
                    col='all',
                )
            except Exception:
                pass
            utils.render_plotly_chart(fig_gender, key='module3_gender_facets')
            try:
                latest_gender_year = gender_df[gen_year].max()
                latest_gender = gender_df[gender_df[gen_year] == latest_gender_year]
                totals = latest_gender.groupby(gen_dim)[gen_count].sum()
                total_sum = totals.sum()
                if total_sum > 0:
                    dominant_gender = totals.idxmax()
                    dominant_pct = (totals.max() / total_sum) * 100
                    st.markdown(
                        f"*{latest_gender_year} gender exposure:* **{dominant_gender}** accounted for roughly {dominant_pct:.1f}% of unemployment across these occupation families."
                    )
            except Exception:
                st.caption('Gender summary unavailable because the latest-year counts are incomplete.')
            st.caption('While male unemployment leads in overall volume, gender exposure varies across occupations. Female workers in lower-skilled roles such as clerical and service occupations remain more exposed to cyclical job losses, reinforcing the gendered vulnerability within Singapore’s labour market.')
    except KeyError:
        st.info('Previous occupation by gender table not available in the current data connection.')

    st.markdown('#### Age group differentials (overall unemployment patterns)')
    try:
        age_raw = tables.get('unemployed_by_age_sex_long') if tables else None
        if age_raw is None:
            raise KeyError('Table unavailable')
        
        # Note: This table only has age_group data without occupation breakdown
        # Process data for overall age patterns
        age_col = _find_column(age_raw, ['age_group', 'ageband', 'age bracket', 'age'])
        count_col = _find_column(age_raw, ['unemployed_count', 'unemployment_count', 'unemp_count'])
        year_col = _find_column(age_raw, ['year'])
        gender_col = _find_column(age_raw, ['gender', 'sex'])
        
        if not all([age_col, count_col, year_col]):
            st.info('Age table lacks required columns for analysis.')
        else:
            # Collapse gender to get overall age patterns
            if gender_col:
                age_summary = age_raw.groupby([year_col, age_col])[count_col].sum().reset_index()
            else:
                age_summary = age_raw[[year_col, age_col, count_col]].copy()
            
            # Convert year to numeric
            if pd.api.types.is_datetime64_any_dtype(age_summary[year_col]):
                age_summary[year_col] = age_summary[year_col].dt.year
            
            # Calculate share percentages by year
            age_summary['total_by_year'] = age_summary.groupby(year_col)[count_col].transform('sum')
            age_summary = age_summary[age_summary['total_by_year'] > 0].copy()
            age_summary['share_pct'] = (age_summary[count_col] / age_summary['total_by_year']) * 100
            
            if not age_summary.empty:
                age_summary[year_col] = age_summary[year_col].round().astype(int)
                age_groups = sorted(age_summary[age_col].unique())
                color_map = {age: px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)] for i, age in enumerate(age_groups)}
                
                fig_age = px.bar(
                    age_summary,
                    x=year_col,
                    y='share_pct',
                    color=age_col,
                    category_orders={age_col: age_groups},
                    color_discrete_map=color_map,
                    labels={year_col: 'Year', 'share_pct': 'Share of unemployed (%)', age_col: 'Age group'},
                    title='Age group share of unemployment over time (overall patterns)',
                    height=650,
                )
                fig_age.update_layout(barnorm='percent', hovermode='x unified', legend_title_text='Age group')
                fig_age.update_yaxes(range=[0, 100], title='Share of unemployed (%)')
                try:
                    fig_age.add_vrect(
                        x0=2019.5,
                        x1=2021.5,
                        fillcolor='rgba(255, 165, 0, 0.15)',
                        line_width=0,
                    )
                except Exception:
                    pass
                utils.render_plotly_chart(fig_age, key='module3_age_overall')
                
                # Summary for latest year
                try:
                    latest_age_year = age_summary[year_col].max()
                    latest_age = age_summary[age_summary[year_col] == latest_age_year]
                    if not latest_age.empty:
                        dominant_age = latest_age.loc[latest_age['share_pct'].idxmax(), age_col]
                        dominant_age_pct = latest_age['share_pct'].max()
                        st.markdown(
                            f"*{int(latest_age_year)} age focus:* **{dominant_age}** made up about {dominant_age_pct:.1f}% of unemployed jobseekers."
                        )
                except Exception:
                    st.caption('Age profile summary unavailable due to data processing issues.')
                
                st.caption('Note: Age data is available at the overall level only. Occupation-specific age breakdowns are not available in the current dataset structure.')
                st.caption('Persistent youth unemployment reinforces the need for early career upskilling and transition initiatives that support integration into stable occupations.')
            else:
                st.info('Age table lacks sufficient data for plotting.')
    except KeyError:
        st.info('Age-based unemployment table not available in the current data connection.')

    st.markdown('---')
    st.markdown(
        """
        ### Comparative lens — Resilience of high- vs low-skill occupations
        This section benchmarks high-skill (PMET) occupations against low-skill groups to assess long-term resilience. This persistent gap confirms the hypothesis that lower-skilled occupations are more vulnerable to technological and industry transformations. 
        """
    )

    default_high_skill = [
        'Professionals',
        'Managers & Administrators (Including Working Proprietors)',
        'Associate Professionals & Technicians',
    ]
    default_low_skill = [
        'Cleaners, Labourers & Related Workers',
        'Service & Sales Workers',
        'Clerical Support Workers',
        'Craftsmen & Related Trades Workers',
        'Plant & Machine Operators & Assemblers',
    ]
    use_defaults = st.checkbox('Use default skill mapping from notebook analysis', value=True, key='module23_comp_skill_defaults')
    if use_defaults:
        high_skill = default_high_skill
        low_skill = default_low_skill
    else:
        high_skill = [
            s.strip()
            for s in st.text_area(
                'High skill occupations (comma separated)',
                value=','.join(default_high_skill),
                key='module23_comp_high_skill',
            ).split(',')
            if s.strip()
        ]
        low_skill = [
            s.strip()
            for s in st.text_area(
                'Low skill occupations (comma separated)',
                value=','.join(default_low_skill),
                key='module23_comp_low_skill',
            ).split(',')
            if s.strip()
        ]

    skill_rate_raw = tables.get('unemployment_rate_by_occupation_long') if tables else df_active
    if skill_rate_raw is None:
        st.info('No unemployment rate table available to compute the comparative lens.')
        return

    skill_rate_raw = skill_rate_raw.copy()
    skill_rate_raw['skill_level'] = skill_rate_raw['occupation'].apply(
        lambda occ: 'High Skill' if occ in high_skill else ('Low Skill' if occ in low_skill else 'Other')
    )
    skill_rate_raw = _ensure_year_int(skill_rate_raw)
    rate_column = _select_rate_column(skill_rate_raw)
    if rate_column is None:
        st.info('Selected unemployment table does not contain a rate column.')
        return

    skill_rate = (
        skill_rate_raw[skill_rate_raw['skill_level'].isin(['High Skill', 'Low Skill'])]
        .groupby(['year_yr', 'skill_level'])[rate_column]
        .mean()
        .unstack('skill_level')
        .dropna()
        .sort_index()
    )

    if rate_column == 'unemployment_rate':
        skill_rate *= 100.0

    if skill_rate.empty:
        st.info('Unable to compute comparative lens because the rate table is empty after filtering.')
        return

    skill_rate['gap_pct_point'] = skill_rate['Low Skill'] - skill_rate['High Skill']
    skill_rate['ratio'] = skill_rate['Low Skill'] / skill_rate['High Skill']
    skill_rate['rolling_ratio'] = skill_rate['ratio'].rolling(window=3, min_periods=1).mean()

    fig_comp = make_subplots(
        rows=2,
        cols=2,
        specs=[[{'colspan': 2}, None], [{}, {}]],
        subplot_titles=(
            'Average unemployment rate by skill tier',
            'Low - High unemployment rate gap',
            'Low-to-high unemployment rate ratio',
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
    )

    fig_comp.add_trace(
        go.Scatter(
            x=skill_rate.index,
            y=skill_rate['High Skill'],
            mode='lines+markers',
            name='High Skill',
            line=dict(color='#1f77b4'),
        ),
        row=1,
        col=1,
    )
    fig_comp.add_trace(
        go.Scatter(
            x=skill_rate.index,
            y=skill_rate['Low Skill'],
            mode='lines+markers',
            name='Low Skill',
            line=dict(color='#ff7f0e'),
        ),
        row=1,
        col=1,
    )
    fig_comp.add_trace(
        go.Bar(
            x=skill_rate.index,
            y=skill_rate['gap_pct_point'],
            name='Gap (pct pts)',
            marker_color='#ff7f0e',
        ),
        row=2,
        col=1,
    )
    fig_comp.add_hline(y=0, line=dict(color='#444', dash='dash'), row=2, col=1)  # type: ignore[call-arg]
    fig_comp.add_trace(
        go.Scatter(
            x=skill_rate.index,
            y=skill_rate['ratio'],
            mode='lines+markers',
            name='Ratio',
            line=dict(color='#2ca02c'),
        ),
        row=2,
        col=2,
    )
    fig_comp.add_trace(
        go.Scatter(
            x=skill_rate.index,
            y=skill_rate['rolling_ratio'],
            mode='lines',
            name='3-year rolling ratio',
            line=dict(color='#17becf', dash='dash'),
        ),
        row=2,
        col=2,
    )
    fig_comp.add_hline(y=1, line=dict(color='#444', dash='dash'), row=2, col=2)  # type: ignore[call-arg]

    fig_comp.update_xaxes(title_text='Year', row=1, col=1)
    fig_comp.update_xaxes(title_text='Year', row=2, col=1)
    fig_comp.update_xaxes(title_text='Year', row=2, col=2)
    fig_comp.update_yaxes(title_text='Unemployment rate (%)', row=1, col=1)
    fig_comp.update_yaxes(title_text='Gap (percentage points)', row=2, col=1)
    fig_comp.update_yaxes(title_text='Low / High ratio', row=2, col=2)
    fig_comp.update_layout(
        height=720,
        legend_title_text='Series',
        hovermode='x unified',
        title_text='Structural resilience comparison: high vs low skill occupations',
        title_x=0.5,
    )
    try:
        fig_comp.add_vrect(
            x0=2019.5,
            x1=2021.5,
            fillcolor='rgba(255, 165, 0, 0.12)',
            line_width=0,
            row='all',
            col='all',
        )
    except Exception:
        pass
    utils.render_plotly_chart(fig_comp, key='module3_comparative_subplots')

    latest_comp_year = skill_rate.index.max()
    latest_row = skill_rate.loc[latest_comp_year]
    st.markdown(
        f"*{int(latest_comp_year)} comparative snapshot:* Low-skill unemployment averaged {latest_row['Low Skill']:.1f}% versus {latest_row['High Skill']:.1f}% for high-skill roles, leaving a gap of {latest_row['gap_pct_point']:.1f} percentage points (ratio {latest_row['ratio']:.2f}x)."
    )
    st.caption('Although recovery reduced the disparity post-COVID, the resilience of high-skill occupations continues to outperform, illustrating the ongoing need for reskilling in lower-skilled job segments.')

    comp_period_bins = pd.cut(
        skill_rate.index,
        bins=[2013, 2019, 2021, 2025],
        labels=['Pre-pandemic (2014-2019)', 'COVID shock (2020-2021)', 'Recovery (2022-2024)'],
        include_lowest=True,
    )
    comp_summary = (
        pd.DataFrame({
            'period': comp_period_bins,
            'gap_pct_point': skill_rate['gap_pct_point'].values,
            'ratio': skill_rate['ratio'].values,
        })
        .groupby('period', observed=False)
        .agg(
            avg_gap_pct_point=('gap_pct_point', 'mean'),
            max_gap_pct_point=('gap_pct_point', 'max'),
            avg_ratio=('ratio', 'mean'),
        )
        .reset_index()
        .round({'avg_gap_pct_point': 2, 'max_gap_pct_point': 2, 'avg_ratio': 2})
    )
    st.dataframe(comp_summary)


def page_cleaning_and_eda(engine: Optional[sqlalchemy.engine.Engine]):
    st.info('Modules 2 and 3 now have dedicated sections. The combined view below mirrors that structure.')
    page_cleaning_module_two(engine)
    st.markdown('---')
    page_visualisation_module_three(engine)


