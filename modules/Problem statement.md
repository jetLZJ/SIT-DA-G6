# Industry-specific unemployment impact — problem statement

## Executive summary

This project analyzes how unemployment rates vary by industry, occupation, education level and demographic groups in Singapore. The goal is to identify which sectors and roles are most vulnerable to job losses (automation, economic shocks, structural change) and to derive evidence-based recommendations for reskilling and policy.

## Main research question

How have different industries and occupations contributed to changes in the unemployment rate in Singapore?

## Objectives

- Identify industries/occupations with consistently high or rising unemployment.
- Measure how education level, gender and occupation skill level affect unemployment.
- Detect resilient industries (stable or falling unemployment) and vulnerable ones.
- Produce predictive and prescriptive outputs (risk ranking, projected trends, recommended reskilling targets).

## Supporting questions

- Trend analysis: Which industries or occupations show persistently high unemployment? How have rates changed over time?
- Education & occupation dynamics: How does education level affect unemployment within occupations? Have gaps widened or narrowed?
- Comparative insights: How do high-skill occupations compare to lower-skill occupations? Are technical skill industries more insulated?
- Role of technology: Are occupations exposed to automation/AI experiencing larger unemployment increases?

## Hypothesis

Lower-skilled occupations (service, sales, clerical, manual labor) have higher and more volatile unemployment rates than higher-skilled occupations (managers, professionals).

> Why it matters: confirmation supports targeted upskilling/reskilling programs and helps prioritize sectors for workforce development.

## Data required (suggested schema)

- year (int)
- quarter (optional)
- industry_code / industry_name
- occupation_code / occupation_name
- education_level (e.g., none, secondary, diploma, degree)
- gender
- age_group
- unemployed_count (int)
- labor_force_count (int)
- unemployment_rate (float)  # unemployed_count / labor_force_count
- avg_wage (optional)
- automation_risk_score (optional)  # external mapping per occupation

If you have household survey or administrative microdata, include identifiers and sampling weights where applicable.

## Suggested analysis plan

1. Data cleaning & validation: missingness, consistent industry/occupation mappings, compute unemployment_rate from counts.
2. Descriptive analysis: time-series plots by industry and occupation; heatmaps of unemployment rates.
3. Stratified analysis: compare rates by education level, gender and age groups within occupations.
4. Resilience / volatility metrics: compute variability (std, CV) and trend slopes per industry/occupation.
5. Predictive model (optional): forecast unemployment rates using time-series or regression models; or build a classifier that flags occupations at high risk.
6. Prescriptive output: rank occupations by risk and map to potential reskilling targets (identify growing industries and overlapping skill sets).

## Possible metrics & evaluation

- Mean and median unemployment rate by group
- Trend slope (change per year)
- Volatility (standard deviation / coefficient of variation)
- Prediction accuracy (RMSE for continuous forecasts; precision/recall for binary high-risk classification)

## Deliverables

- Notebook(s) with reproducible analysis (Python/pandas or R)
- Visual dashboard or Streamlit app showing interactive filters by industry/occupation/education
- Short written summary (1–2 pages) with policy recommendations and top 10 occupations for targeted reskilling
- Optional: model artifacts (trained models, prediction scripts)

## Prompt-friendly example tasks (use with an LLM / Copilot)

1. Quick descriptive prompt

"Given a CSV with columns year, industry_name, occupation_name, education_level, unemployed_count, labor_force_count — compute unemployment_rate, then return a ranked table of industries by average unemployment_rate for the most recent year. Provide example pandas code and a short explanation of the steps."

2. Visualization prompt

"Create Python/pandas + matplotlib (or plotly) code that plots a small multiples chart of unemployment_rate vs year for the top 6 industries by average unemployment rate. The code should be copy-paste runnable given a dataframe named df."

3. Streamlit UI prompt

"Generate a Streamlit app snippet that loads `data.csv`, lets the user select an industry and education level, and displays a time series chart of unemployment_rate and a table of the top 10 occupations by unemployment_rate for the selected filters."

4. Predictive prompt

"Suggest a simple time-series forecasting approach (and example code) to project unemployment_rate for the next 4 quarters for each industry using ARIMA or Prophet. Include how to evaluate forecast accuracy with a train/test split."

5. Policy recommendation prompt

"Summarize the top 5 policy recommendations if analysis finds lower-skilled occupations are at significantly higher unemployment risk. Keep recommendations short and actionable."

## Next steps / notes

- Confirm available datasets and their schema; share a sample CSV (anonymized) for faster iteration.
- Start with exploratory notebooks and one Streamlit page for interactive exploration.
- When ready, add predictive modelling and the reskilling mapping.
