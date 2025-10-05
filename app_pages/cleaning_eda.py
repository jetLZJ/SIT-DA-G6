import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Optional, List
import sqlalchemy

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app import data_loader, utils, viz


def recommend_actions_to_fully_answer_problem_statement() -> List[str]:
    return [
        "1) Confirm canonical tables and their schema (long/wide) and document code-lists.",
        "2) Recompute and validate base rates (unemployment_rate from counts) and apply weights if available.",
        "3) Create canonical occupation/industry mappings and attach external metadata (skill level, automation exposure).",
        "4) Compute resilience metrics (trend slope, volatility, recovery speed) and rank occupations.",
        "5) Produce stratified analyses by education, gender, age (small multiples & heatmaps).",
        "6) Apply robustness checks (outlier handling, sensitivity to imputation) and re-evaluate metrics.",
        "7) Build forecasting baselines per occupation with rolling-origin CV; report RMSE and intervals.",
        "8) Combine trend/resilience with automation exposure into a composite risk score and rank occupations.",
        "9) Map high-risk occupations to related growing occupations/skills for reskilling targets.",
        "10) Package deliverables: reproducible notebooks, interactive Streamlit dashboard, and a concise policy brief.",
    ]


def load_long_wide_from_db(engine: sqlalchemy.engine.Engine) -> tuple[dict, dict]:
    """Load long and wide tables into dicts (table_name -> DataFrame)."""
    inspector = sqlalchemy.inspect(engine)
    all_tables = inspector.get_table_names()
    long_tables = [t for t in all_tables if t.endswith('long')]
    wide_tables = [t for t in all_tables if t.endswith('wide')]
    df_long_dict = {t: pd.read_sql(f"SELECT * FROM {t}", engine) for t in long_tables}
    df_wide_dict = {t: pd.read_sql(f"SELECT * FROM {t}", engine) for t in wide_tables}
    return df_long_dict, df_wide_dict


def page_cleaning_and_eda(engine: Optional[sqlalchemy.engine.Engine]):
    """Render a presentation-ready version of the M2/M3 notebook with interactive visualisations and narrative.

    The content follows the notebook's flow but is rewritten to be presentation-friendly and to explicitly address
    the project Problem Statement (trend analysis, stratified analysis, comparative insights, automation risk gap).
    """
    st.title('Module 2 & 3 — Cleaning, EDA & Visualisation (Presentation)')

    st.markdown('## Executive summary')
    st.markdown(
        """
        This page converts the `M2 M3 EDA and Visualisation` notebook into a presentation narrative. It loads the
        project's canonical long/wide tables (from the DB when available, otherwise via CSV upload), runs the same
        cleaning and derived-table computations (for example, computing employed_count from unemployment_rate),
        performs data checks and outlier inspection, and reproduces the notebook visualisations interactively.

        The goal: answer the main research question — how industries and occupations contributed to changes in the
        unemployment rate — and provide a clear set of next steps to complete prescriptive outputs.
        """
    )

    st.markdown('---')

    # Data loading controls
    st.header('Data Loading')
    st.write('Choose a data source: connect to the project DB (via Streamlit secrets) or upload CSVs for quick checks.')

    df_long_dict: dict = {}
    df_wide_dict: dict = {}
    if engine is not None:
        st.success('DB connection available — listing tables...')
        try:
            df_long_dict, df_wide_dict = load_long_wide_from_db(engine)
        except Exception as e:
            st.error(f'Failed to load tables from DB: {e}')

    if not df_long_dict:
        uploaded = st.file_uploader('Upload a representative long-format CSV (if no DB). You can upload multiple files sequentially.', accept_multiple_files=True)
        if uploaded:
            for f in uploaded:
                name = Path(f.name).stem
                try:
                    df_long_dict[name] = pd.read_csv(f)
                except Exception:
                    st.warning(f'Failed to read {f.name}')

    if not df_long_dict:
        st.info('No long-format tables available. Provide DB connection or upload CSVs to proceed.')
        return

    st.markdown('Loaded long-format tables: ' + ', '.join(list(df_long_dict.keys())[:10]))

    # Choose which long table to use for examples
    options = list(df_long_dict.keys())
    # Prefer the canonical notebook tables if present
    preferred = [
        'unemployment_rate_by_occupation_long',
        'unemployed_by_previous_occupation_sex_long',
        'unemployed_by_age_sex_long',
        'unemployed_by_qualification_sex_long'
    ]
    default_index = 0
    for p in preferred:
        if p in options:
            default_index = options.index(p)
            break

    table_choice = st.selectbox('Pick a long table to work with (example visualisations)', options=options, index=default_index)
    df = df_long_dict[table_choice].copy()

    # Basic cleaning from notebook: year -> datetime (safer heuristic)
    if 'year' in df.columns:
        # Inspect a small sample to decide whether to parse as date or keep numeric years
        sample_vals = df['year'].dropna().head(20).astype(str)
        looks_like_date = sample_vals.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}") | sample_vals.str.contains(r"[A-Za-z]{3,}")
        # If the sample clearly contains date-like strings, parse to datetime; otherwise keep as-is
        try:
            if looks_like_date.any():
                df['year'] = pd.to_datetime(df['year'], errors='coerce')
            else:
                # Keep numeric years as numeric (do not coerce to datetime which treats numbers as nanoseconds)
                # We still leave strings untouched so downstream logic can attempt safer parsing.
                pass
        except Exception:
            pass

    st.header('Data checks and descriptive summary')
    st.write('Rows: ', len(df), ' — Columns: ', len(df.columns))
    st.subheader('Sample')
    st.dataframe(df.head())

    st.subheader('Data types & missingness')
    st.write(df.dtypes.astype(str))
    st.write(df.isnull().sum())

    # Attempt to normalise common column name variants and compute/standardise rate columns
    def _normalize_and_compute_rates(df_in: pd.DataFrame):
        """Return a DataFrame with canonical columns where possible and a mapping of detected names.

        This will try to:
         - map common variants (spaces, capitals) to canonical column names
         - compute unemployment_rate/unemployed_rate from counts if possible
        """
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

    df, _mapping = _normalize_and_compute_rates(df)

    def _ensure_year_int(df_in: pd.DataFrame):
        """Ensure there's a numeric integer year column called 'year_yr' for plotting.

        Heuristics applied (in order):
        - If 'year' is datetime-like and produces sensible years (not all 1970) use .dt.year
        - If 'year' is datetime-like but all years are 1970 (common when numeric years were parsed as
          nanoseconds), attempt to recover by reading the underlying integer nanosecond values and
          treating small integers as raw year values.
        - If 'year' is numeric-like, coerce to integer year.
        - If 'year' is string-like, attempt pd.to_datetime and extract year.

        The function returns a copy of the dataframe. If a recovery from the epoch/nanosecond issue
        was performed, it sets df.attrs['year_recovered'] = True so the UI can inform the user.
        """
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

    # create a canonical integer-year column used across plots
    df = _ensure_year_int(df)

    # Debug panel: show what column names we detected and sample values for the rate column
    with st.expander('Debug: detected column mapping and sample values', expanded=False):
        st.json(_mapping)
        det = _mapping.get('detected_rate_column')
        if det and det in df.columns:
            st.write('Sample detected rate column values (first 10):')
            st.write(df[det].head(10))
            st.write('Dtype:', str(df[det].dtype))
        else:
            st.write('No pre-existing rate column detected; derived columns may have been created if counts were present.')
        # Show year diagnostics
        if 'year' in df.columns:
            st.write('\nSample raw `year` values (first 10):')
            try:
                st.write(df['year'].head(10))
            except Exception:
                st.write('Could not display raw year values')
            st.write('raw year dtype:', str(df['year'].dtype))
        if 'year_yr' in df.columns:
            st.write('\nComputed `year_yr` values (first 10):')
            try:
                st.write(df['year_yr'].head(10))
            except Exception:
                st.write('Could not display `year_yr` values')
            st.write('year_yr dtype:', str(df['year_yr'].dtype))
        # If year_yr collapsed to 1970 for all rows, warn the user — this is usually caused by parsing
        # numeric year values with pd.to_datetime which interprets integers as nanoseconds since epoch.
        try:
            if 'year_yr' in df.columns and df['year_yr'].notna().any():
                uniq = df['year_yr'].dropna().unique()
                if len(uniq) == 1 and int(uniq[0]) == 1970:
                    st.warning('All computed `year_yr` values are 1970. This often happens when numeric years were parsed as nanoseconds (pd.to_datetime on ints).\nRecommend: re-upload CSV or provide DB table with a proper numeric year column, or contact the maintainer to force numeric year extraction.')
        except Exception:
            pass

    st.subheader('Descriptive statistics (numeric)')
    st.dataframe(df.select_dtypes(include=['number']).describe().T)

    st.markdown('---')

    def _select_rate_column(df: pd.DataFrame) -> str:
        """Return the column name to use for plotting unemployment rate and a normalized series (percent).

        Preference order:
         - 'unemployment_rate' (proportion) -> converted to percent for plotting
         - 'unemployed_rate' (already percent)
        """
        if 'unemployment_rate' in df.columns:
            return 'unemployment_rate'
        if 'unemployed_rate' in df.columns:
            return 'unemployed_rate'
        return ''

    # Employed_count derivation (notebook cell)
    st.header('Derived table: employed_count (notebook example)')
    # Determine rate column early
    rate_col = _select_rate_column(df)
    # Use detected rate column (could be 'unemployment_rate' (proportion) or 'unemployed_rate' (percent))
    if 'unemployed_count' in df.columns and rate_col and 'occupation' in df.columns:
        st.write('Computing employed_count from unemployed_count and detected rate column as shown in the notebook...')
        cols = ['year', 'occupation', 'unemployed_count', rate_col]
        temp = df[cols].dropna()
        temp = temp.copy()
        # If rate_col is a proportion (unemployment_rate) use it directly; if percent convert to proportion
        if rate_col == 'unemployment_rate':
            temp['unemployed_rate_prop'] = temp[rate_col].astype(float)
            temp['unemployed_rate_display'] = temp[rate_col] * 100.0
        else:
            temp['unemployed_rate_prop'] = temp[rate_col].astype(float) / 100.0
            temp['unemployed_rate_display'] = temp[rate_col].astype(float)

        # Avoid division by zero using vectorized operations
        mask = temp['unemployed_rate_prop'].notna() & (temp['unemployed_rate_prop'] > 0)
        temp['employed_count'] = None
        temp.loc[mask, 'employed_count'] = temp.loc[mask, 'unemployed_count'] * (1.0 / temp.loc[mask, 'unemployed_rate_prop'] - 1.0)
        st.dataframe(temp[['year', 'occupation', 'unemployed_count', 'unemployed_rate_display', 'employed_count']].head())
    else:
        st.info('This table does not contain the columns required for the employed_count example (unemployed_count & unemployed_rate & occupation).')

    st.markdown('---')

    # Outlier inspection (notebook flow)
    st.header('Outlier inspection')
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if numeric_cols:
        pick_col = st.selectbox('Pick numeric column to inspect', options=numeric_cols)
        fig = px.histogram(df, x=pick_col, nbins=30, title=f'Distribution of {pick_col}', marginal='box')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('No numeric columns found for outlier inspection.')

    st.markdown('---')

    # Module 3 visualisations from notebook
    st.header('Module 3 — Visualisations (recreated from notebook)')

    # Trend analysis: unemployment_rate by occupation
    rate_col = _select_rate_column(df)
    if 'occupation' in df.columns and rate_col and 'year' in df.columns:
        st.subheader('Trend: unemployment rate by occupation')
        # let user pick occupations or use top N by average rate
        occupations = sorted(df['occupation'].dropna().unique())
        pick_mode = st.radio('Select occupations by', options=['Top by average unemployment_rate', 'Manual select'])
        if pick_mode == 'Top by average unemployment_rate':
            topn = st.slider('Top N occupations', min_value=3, max_value=min(20, len(occupations)), value=8)
            # Use the selected rate column for ranking
            top_occ = df.groupby('occupation')[rate_col].mean().nlargest(topn).index.tolist()
            trend_df = df[df['occupation'].isin(top_occ)]
        else:
            sel = st.multiselect('Pick occupations', options=occupations, default=occupations[:6])
            trend_df = df[df['occupation'].isin(sel)]

        # use canonical integer year column
        trend_df = _ensure_year_int(trend_df)
        x_col = 'year_yr'

        # Normalize y to percent for display
        tdf = trend_df.copy()
        if rate_col == 'unemployment_rate':
            tdf['plot_unemp_pct'] = tdf[rate_col] * 100.0
            ycol = 'plot_unemp_pct'
            ytitle = 'Unemployment rate (%)'
        else:
            tdf['plot_unemp_pct'] = tdf[rate_col]
            ycol = 'plot_unemp_pct'
            ytitle = 'Unemployment rate (%)'

        fig = px.line(tdf, x=x_col, y=ycol, color='occupation', markers=True, title='Unemployment rate by occupation')
        fig.update_yaxes(title=ytitle)
        # Force integer year ticks (one tick per year) when x is numeric year
        try:
            minx = int(tdf[x_col].min())
            fig.update_xaxes(tickmode='linear', tick0=minx, dtick=1)
        except Exception:
            pass
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info('Required columns for trend plot not found (occupation, unemployed_rate, year).')

    # Skill-level comparison
    st.markdown('---')
    st.subheader('Skill-level comparison (High vs Low)')
    # Use notebook's example mapping; allow override
    default_high_skill = ["Professionals", "Managers & Administrators (Including Working Proprietors)", "Associate Professionals & Technicians"]
    default_low_skill = ["Cleaners, Labourers & Related Workers", "Service & Sales Workers", "Clerical Support Workers", "Craftsmen & Related Trades Workers", "Plant & Machine Operators & Assemblers"]
    use_defaults = st.checkbox('Use notebook default skill mapping', value=True)
    if use_defaults:
        high_skill = default_high_skill
        low_skill = default_low_skill
    else:
        high_skill = st.text_area('High skill comma-separated', value=','.join(default_high_skill)).split(',')
        low_skill = st.text_area('Low skill comma-separated', value=','.join(default_low_skill)).split(',')

    if 'occupation' in df.columns and rate_col:
        df2 = df.copy()
        df2['skill_level'] = df2['occupation'].apply(lambda x: 'High Skill' if x in high_skill else ('Low Skill' if x in low_skill else 'Other'))
        df2 = _ensure_year_int(df2)
        xcol = 'year_yr'
        # Prepare percent series
        if rate_col == 'unemployment_rate':
            df2['plot_unemp_pct'] = df2[rate_col] * 100.0
        else:
            df2['plot_unemp_pct'] = df2[rate_col]

        fig2 = px.line(df2, x=xcol, y='plot_unemp_pct', color='skill_level', line_group='occupation', markers=True,
                       title='Unemployment Rate: High vs Low Skill Occupations')
        fig2.update_yaxes(title='Unemployment rate (%)')
        try:
            minx2 = int(df2[xcol].min())
            fig2.update_xaxes(tickmode='linear', tick0=minx2, dtick=1)
        except Exception:
            pass
        st.plotly_chart(fig2, use_container_width=True)

    # Period segmented small multiples (notebook used matplotlib subplots)
    st.markdown('---')
    st.subheader('Period-segmented small multiples (occupation-level)')
    if 'year' in df.columns and 'occupation' in df.columns and 'unemployed_rate' in df.columns:
        periods = {
            '2014-2016': ('2014-01-01', '2016-12-31'),
            '2017-2019': ('2017-01-01', '2019-12-31'),
            '2020-2021': ('2020-01-01', '2021-12-31'),
            '2022-2024': ('2022-01-01', '2024-12-31'),
        }
        # convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['year']):
            try:
                df['year'] = pd.to_datetime(df['year'])
            except Exception:
                pass

        # Use plotly facet for small multiples for chosen occupations
        occs = st.multiselect('Pick occupations to show small multiples (max 12)', options=sorted(df['occupation'].unique()), default=sorted(df['occupation'].unique())[:8])
        if occs:
            df_occ = _ensure_year_int(df[df['occupation'].isin(occs)].copy())
            # Prepare percent series
            if rate_col == 'unemployment_rate':
                df_occ['plot_unemp_pct'] = df_occ[rate_col] * 100.0
            else:
                df_occ['plot_unemp_pct'] = df_occ[rate_col]

            fig3 = px.line(df_occ, x='year_yr', y='plot_unemp_pct', color='occupation', facet_col='occupation', facet_col_wrap=4, markers=True,
                           title='Occupation small multiples (segmented periods visualised by lines)')
            fig3.update_yaxes(title='Unemployment rate (%)')
            try:
                minx3 = int(df_occ['year_yr'].min())
                fig3.update_xaxes(tickmode='linear', tick0=minx3, dtick=1)
            except Exception:
                pass
            st.plotly_chart(fig3, use_container_width=True)

    # Heatmap
    st.markdown('---')
    st.subheader('Heatmap: unemployment rate by occupation and year')
    if {'occupation', 'year'}.issubset(df.columns) and rate_col:
        # Pivot using the normalized percent series
        tmp = _ensure_year_int(df.copy())
        if rate_col == 'unemployment_rate':
            tmp['plot_unemp_pct'] = tmp[rate_col] * 100.0
        else:
            tmp['plot_unemp_pct'] = tmp[rate_col]

        # Pivot on integer year
        pivot = tmp.pivot_table(index='occupation', columns='year_yr', values='plot_unemp_pct', aggfunc='mean')
        fig4 = px.imshow(pivot.fillna(0), labels=dict(x='Year', y='Occupation', color='Unemployment Rate (%)'), aspect='auto', color_continuous_scale='YlGnBu')
        fig4.update_layout(title='Unemployment Rate by Occupation and Year (heatmap)')
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info('Heatmap requires occupation, year and unemployed_rate columns.')

    # Share of unemployment (stacked area)
    st.markdown('---')
    st.subheader('Share of unemployment by occupation (stacked area)')
    if {'occupation', 'year'}.issubset(df.columns) and rate_col:
        tmp = _ensure_year_int(df.copy())
        if rate_col == 'unemployment_rate':
            tmp['plot_unemp_pct'] = tmp[rate_col] * 100.0
        else:
            tmp['plot_unemp_pct'] = tmp[rate_col]

        pivot_sum_df = tmp.pivot_table(index='year_yr', columns='occupation', values='plot_unemp_pct', aggfunc='sum')
        # Convert to proportions per year
        prop = pivot_sum_df.divide(pivot_sum_df.sum(axis=1), axis=0).reset_index()
        # Rename the first column (year_yr) to 'year' for plotting consistency
        first_col = prop.columns[0]
        prop = prop.rename(columns={first_col: 'year'})

        fig5 = px.area(prop, x='year', y=prop.columns[1:], title='Share of unemployment by occupation (proportion)')
        try:
            minx5 = int(prop['year'].min())
            fig5.update_xaxes(tickmode='linear', tick0=minx5, dtick=1)
        except Exception:
            pass
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info('Stacked share requires occupation, year, unemployed_rate columns.')

    st.markdown('---')
    st.header('How this answers the Problem Statement')
    st.markdown(
        """
        - Trend analysis: the time-series and small-multiples show which occupations consistently have higher unemployment and when spikes occurred (notably 2020).
        - Education & occupation dynamics: the notebook contains animated and stratified examples; the page provides the same and allows further stratified groupings to be added.
        - Comparative insights: the skill-level comparison and share charts demonstrate the resilience gap between high- and low-skill occupations.
        - Role of technology: TODO — join an external automation/AI exposure dataset to test correlations with vulnerability.
        """
    )

    st.subheader('Recommended next steps (ordered)')
    for step in recommend_actions_to_fully_answer_problem_statement():
        st.write(step)

    st.markdown('---')
    nb_path = Path(__file__).parent.parent / 'modules' / 'M2 M3 EDA and Visualisation.ipynb'
    if nb_path.exists():
        st.download_button('Download the notebook (original)', nb_path.read_bytes(), file_name='M2_M3_EDA_and_Visualisation.ipynb')

