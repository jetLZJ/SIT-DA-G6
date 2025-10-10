import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Optional
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
    st.title('Module 2 — Data cleaning & feature preparation')

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
    _set_active_dataframe(df_clean, selected_table)

    outlier_table_options = sorted(tables.keys())
    default_outlier_table = st.session_state.get('module23_outlier_table', selected_table)
    default_outlier_index = _default_table_index(outlier_table_options, default_outlier_table)
    outlier_table = st.selectbox(
        'Dataset reference for quality checks',
        options=outlier_table_options,
        index=default_outlier_index,
        key='module23_outlier_table'
    )

    with st.expander('Step 1 — Column normalisation summary', expanded=False):
        st.json(mapping)

    with st.expander('Step 2 — Data health checks', expanded=False):
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
                st.plotly_chart(hist_fig, use_container_width=True)
            with right_col:
                box_fig = px.box(outlier_df, y=pick_col, title=f'Box plot — {pick_col}')
                box_fig.update_layout(margin=dict(t=40, r=20, l=20, b=40))
                st.plotly_chart(box_fig, use_container_width=True)

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

    df_active, table_name = _get_active_dataframe(engine, allow_refresh=False)
    if df_active is None or table_name is None:
        st.info('Load a dataset in Module 2 first, or ensure a database connection is available.')
        return

    st.caption(f'Using **{table_name}** — {df_active.shape[0]} rows × {df_active.shape[1]} columns')

    df_active = _ensure_year_int(df_active.copy())
    rate_col = _select_rate_column(df_active)
    if not rate_col:
        st.warning('Active dataset has no unemployment rate column after normalisation.')
        return

    if 'occupation' not in df_active.columns:
        st.warning('Active dataset is missing an `occupation` column required for Module 3 visuals.')
        return

    occupations = sorted(df_active['occupation'].dropna().unique().tolist())

    st.subheader('Trend: unemployment rate by occupation')
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
            if rate_col == 'unemployment_rate':
                plot_df['plot_unemp_pct'] = plot_df[rate_col] * 100.0
            else:
                plot_df['plot_unemp_pct'] = plot_df[rate_col]

            fig = px.line(plot_df, x='year_yr', y='plot_unemp_pct', color='occupation', markers=True, title='Unemployment rate by occupation')
            fig.update_yaxes(title='Unemployment rate (%)')
            try:
                min_year = int(plot_df['year_yr'].min())
                fig.update_xaxes(tickmode='linear', tick0=min_year, dtick=1)
            except Exception:
                pass
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No rows available for the selected occupations.')
    else:
        st.info('No occupation values available for trend analysis.')

    st.markdown('---')
    st.subheader('Skill-level comparison (High vs Low)')
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
    use_defaults = st.checkbox('Use notebook default skill mapping', value=True, key='module23_skill_defaults')
    if use_defaults:
        high_skill = default_high_skill
        low_skill = default_low_skill
    else:
        high_skill = [s.strip() for s in st.text_area('High skill occupations (comma separated)', value=','.join(default_high_skill), key='module23_high_skill').split(',') if s.strip()]
        low_skill = [s.strip() for s in st.text_area('Low skill occupations (comma separated)', value=','.join(default_low_skill), key='module23_low_skill').split(',') if s.strip()]

    skill_df = df_active.copy()
    skill_df['skill_level'] = skill_df['occupation'].apply(
        lambda occ: 'High Skill' if occ in high_skill else ('Low Skill' if occ in low_skill else 'Other')
    )
    skill_df = _ensure_year_int(skill_df)
    if rate_col == 'unemployment_rate':
        skill_df['plot_unemp_pct'] = skill_df[rate_col] * 100.0
    else:
        skill_df['plot_unemp_pct'] = skill_df[rate_col]

    fig2 = px.line(
        skill_df,
        x='year_yr',
        y='plot_unemp_pct',
        color='skill_level',
        line_group='occupation',
        markers=True,
        title='Unemployment rate — High vs Low skill occupations'
    )
    fig2.update_yaxes(title='Unemployment rate (%)')
    try:
        min_year2 = int(skill_df['year_yr'].min())
        fig2.update_xaxes(tickmode='linear', tick0=min_year2, dtick=1)
    except Exception:
        pass
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('---')
    st.subheader('Occupation small multiples')
    default_multiples = occupations[: min(8, len(occupations))]
    chosen_multiples = st.multiselect('Pick occupations (max 12)', options=occupations, default=default_multiples, key='module23_small_multiples')
    if chosen_multiples:
        occ_df = df_active[df_active['occupation'].isin(chosen_multiples)].copy()
        occ_df = _ensure_year_int(occ_df)
        if rate_col == 'unemployment_rate':
            occ_df['plot_unemp_pct'] = occ_df[rate_col] * 100.0
        else:
            occ_df['plot_unemp_pct'] = occ_df[rate_col]

        fig3 = px.line(
            occ_df,
            x='year_yr',
            y='plot_unemp_pct',
            color='occupation',
            facet_col='occupation',
            facet_col_wrap=4,
            markers=True,
            title='Occupation small multiples'
        )
        fig3.update_yaxes(title='Unemployment rate (%)')
        try:
            min_year3 = int(occ_df['year_yr'].min())
            fig3.update_xaxes(tickmode='linear', tick0=min_year3, dtick=1)
        except Exception:
            pass
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info('Select at least one occupation to show the small multiples.')

    st.markdown('---')
    st.subheader('Heatmap: unemployment rate by occupation and year')
    heat_df = _ensure_year_int(df_active.copy())
    if 'year_yr' in heat_df.columns:
        if rate_col == 'unemployment_rate':
            heat_df['plot_unemp_pct'] = heat_df[rate_col] * 100.0
        else:
            heat_df['plot_unemp_pct'] = heat_df[rate_col]

        pivot = heat_df.pivot_table(index='occupation', columns='year_yr', values='plot_unemp_pct', aggfunc='mean')
        if pivot.empty:
            st.info('Heatmap has no data to display after pivoting.')
        else:
            fig4 = px.imshow(
                pivot.fillna(0),
                labels=dict(x='Year', y='Occupation', color='Unemployment rate (%)'),
                aspect='auto',
                color_continuous_scale='YlGnBu'
            )
            fig4.update_layout(title='Unemployment rate by occupation and year')
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info('Year information is missing; unable to generate heatmap.')

    st.markdown('---')
    st.subheader('Share of unemployment by occupation (stacked area)')
    share_df = _ensure_year_int(df_active.copy())
    if 'year_yr' in share_df.columns:
        if rate_col == 'unemployment_rate':
            share_df['plot_unemp_pct'] = share_df[rate_col] * 100.0
        else:
            share_df['plot_unemp_pct'] = share_df[rate_col]

        pivot_sum = share_df.pivot_table(index='year_yr', columns='occupation', values='plot_unemp_pct', aggfunc='sum').fillna(0)
        row_totals = pivot_sum.sum(axis=1)
        non_zero = row_totals.replace(0, pd.NA)
        prop = pivot_sum.divide(non_zero, axis=0).dropna(how='all')
        if prop.empty:
            st.info('Not enough non-zero data to compute share of unemployment.')
        else:
            area_df = prop.reset_index().rename(columns={'year_yr': 'year'})
            fig5 = px.area(area_df, x='year', y=area_df.columns[1:], title='Share of unemployment by occupation')
            try:
                min_year5 = int(area_df['year'].min())
                fig5.update_xaxes(tickmode='linear', tick0=min_year5, dtick=1)
            except Exception:
                pass
            st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info('Year information is missing; unable to compute share of unemployment.')


def page_cleaning_and_eda(engine: Optional[sqlalchemy.engine.Engine]):
    st.info('Modules 2 and 3 now have dedicated sections. The combined view below mirrors that structure.')
    page_cleaning_module_two(engine)
    st.markdown('---')
    page_visualisation_module_three(engine)


