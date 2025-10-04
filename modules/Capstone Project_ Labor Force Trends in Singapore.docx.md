
## Capstone Project — Labor Force Trends in Singapore (streamlit presentation spec)

This document is the presentation specification for the Streamlit app. It maps the project requirements (modules M1..M5) to a 4‑module Streamlit app (M2 and M3 are combined into a single module page) and ties the interactive narrative to the problem statement in `modules/Problem statement.md`.

The goal: produce a light‑themed, professional data narrative that walks stakeholders through the problem, exploratory findings, statistical insights, predictive analysis, and an interactive BI-style summary.

Design principles
- Keep the visual style light and professional (soft neutrals + a single accent color). Prefer Plotly for interactive charts and ensure accessibility (sufficient contrast).
- Narrative-first: each module starts with a short narrative excerpt drawn from `modules/Problem statement.md` and links back to the main research question.
- Minimal dependencies: rely on Streamlit, pandas, Plotly, and SQLAlchemy for DB access. Provide upload fallback when `st.secrets['DB_CONNECTION_STRING']` is unavailable.

Color palette (light, professional)
- Background: #F7F8FA (very light gray)
- Surface / cards: #FFFFFF (white)
- Text primary: #0F1724 (dark slate)
- Text secondary: #4B5563 (muted gray)
- Accent: #2563EB (professional blue)
- Accent 2 (optional): #10B981 (soft green for positive trends)

App structure — mapping modules

Overview (landing)
- Landing pane: app title, short narrative (pulled from `modules/Problem statement.md` executive summary), and 1–2 headline KPIs (current national unemployment rate, most vulnerable industry).
- Navigation: left sidebar with Module links: Module 1 → Module 2+3 → Module 4 → Module 5 (BI summary).

Module 1 — Data Fundamentals & SQL (M1)
- Purpose: show data provenance and schema. Demonstrate SQL exploratory queries and how raw tables map to the analysis tables used in later modules.
- Components:
  - Table list from DB (via `st.secrets['DB_CONNECTION_STRING']`) and small preview.
  - Schema viewer: columns, types, sample rows.
  - Example SQL queries (display only) and short explanation of what they show.

Module 2 — Data Cleaning, EDA & Statistics (M2 + M3 combined)
- Purpose: clean, transform and explore. Present descriptive statistics and robust visuals.
- Components:
  - Data cleaning log (what was done: missing values, filters, renames).
  - Time-series charts: unemployment_rate by industry and occupation (Plotly small‑multiples).
  - Comparative charts: education vs unemployment, gender splits.
  - Statistical summaries: mean/median/trend slopes and a correlation heatmap.

Module 3 — Machine Learning (M4)
- Purpose: provide predictive insights and risk ranking for occupations/industries.
- Components:
  - Simple forecasting example (per‑industry ARIMA/Prophet or simple exponential smoothing) with train/test evaluation and visualization of forecasts.
  - Risk classifier or scoring: a small model that flags occupations at high unemployment risk (logistic/regression feature importance table).
  - Short explanation of model limitations and recommended next steps.

Module 4 — Business Intelligence & Presentation (M5)
- Purpose: synthesize insights into an interactive report suitable for stakeholders.
- Components:
  - Executive dashboard: headline indicators, trend sparklines, top 5 vulnerable occupations, recommended reskilling targets.
  - Filters for year, industry, occupation, education level.
  - Download/export buttons for charts and data (CSV/PNG).
  - Embedded instructions for exporting to Power BI (link or brief steps) — do not require Power BI to be installed in the app.

Data contract (pull from `modules/Problem statement.md`)
- Required fields (minimum): year, industry_name, occupation_name, education_level, gender, unemployed_count, labor_force_count
- Derived fields: unemployment_rate = unemployed_count / labor_force_count, trend_slope, volatility

Interactive behavior & fallbacks
- Primary data source: MySQL DB via `st.secrets['DB_CONNECTION_STRING']` (SQLAlchemy engine). The app must attempt the DB connection on load and show a friendly banner if missing.
- Fallback: CSV upload widget that maps columns to the required schema.
- Performance: limit SQL reads to reasonable chunk sizes for previews and use streaming/limits for large tables.

Visual design notes
- Use Plotly themes with the accent color (#2563EB). Keep fonts legible (system UI fonts). Use card layouts for narrative + charts.
- Keep interactions responsive: use sidebar controls, avoid heavy synchronous model retraining on every widget change (provide an explicit "Run analysis" button).

Prompt-friendly helper examples (for LLM or Copilot)
- "Generate a Streamlit snippet that connects to the DB using `st.secrets['DB_CONNECTION_STRING']`, lists tables, and previews a selected table (limit 1000 rows). Include error handling."
- "Write a `compute_unemployment_rate(df)` utility that validates columns and returns a DataFrame with `unemployment_rate` as float." 
- "Provide Plotly code to render small-multiples of unemployment_rate vs year for up to 12 occupations, wrapped into 4 columns, suitable for use in Streamlit." 
- "Create a Streamlit UI snippet that exposes filters (year, industry, education) and updates a Plotly chart when an explicit 'Apply filters' button is clicked."

Deliverables (app & artifacts)
- Streamlit app codebase (entry `streamlit_app.py` + small `app/` helpers).
- Notebooks showing EDA and model experiments.
- SQL query snippets used for data selection/aggregation.
- Short presentation (PDF or slides) with the 4‑module narrative and top recommendations.

Acceptance criteria
- The app runs with the DB connection in `st.secrets` or shows the CSV upload fallback.
- The landing page uses the Problem Statement narrative and presents at least one interactive Plotly chart.
- Each module contains the described components and matches the problem statement's analytical goals.

Notes on implementation
- Combine M2 and M3 into one module page (Data Cleaning + EDA + Statistics) to keep the app to 4 pages while still covering the course modules (M1..M5).
- Keep the UI light-themed and professional as specified.
- Document any assumptions about data (missing fields, units) in the app's "Data Notes" section.

——

If you'd like, I can now:
1) Update the repository `README.md` and `modules/Problem statement.md` to reflect this new presentation spec, or
2) Scaffold the Streamlit pages and wire up the navigation (I will reuse the DB secret and add CSV fallback), or
3) Generate the Plotly templates and example notebook cells for the main visuals.

Tell me which of (1/2/3) you want me to do next and I'll proceed.


