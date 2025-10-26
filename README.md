# 📊 Singapore Labour Force Analysis
## Unemployment Insights for Workforce Planning (2014-2024)

A comprehensive data analytics project examining unemployment trends across occupations, demographics, and education levels in Singapore to inform evidence-based policy recommendations and workforce development strategies.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sit-da-g6.streamlit.app/)

## 🎯 Project Overview

This capstone project analyzes 11 years of Singapore's labour force data to identify unemployment patterns, predict future trends, and provide actionable insights for workforce planning. The analysis focuses on occupation-specific vulnerabilities and proposes targeted interventions for high-risk worker groups.

### Key Research Questions
- **Primary**: Which occupations & industries drive unemployment swings?
- How do education level, gender, and age affect unemployment within occupations?
- Which sectors are most resilient vs. vulnerable to economic shocks?
- What are the predicted unemployment trends for 2025?

### Working Hypothesis
> Lower-skilled occupations (service, sales, clerical, manual labour) exhibit higher and more volatile unemployment than professional and managerial cohorts.

## �️ Project Structure

```
SIT-DA-G6/
├── 📱 streamlit_app.py          # Main application entry point
├── 🔧 app/                     # Core utilities and data processing
│   ├── data_loader.py          # Database connection and data loading
│   ├── utils.py                # Utility functions and calculations
│   └── viz.py                  # Visualization helpers
├── 📊 app_pages/               # Multi-page application modules
│   ├── overview.py             # Project overview and Power BI dashboard
│   ├── data_schema.py          # Data structure and SQL schemas
│   ├── cleaning_eda.py         # Data cleaning and exploratory analysis
│   ├── dashboard.py            # Interactive dashboards
│   ├── module_4_machine_learning.py  # ML models and predictions
│   └── presentation_slides.py  # Presentation mode interface
├── 📚 modules/                 # Project documentation and notebooks
│   ├── M1 Data Fundamentals and SQL G6 v4.docx
│   ├── M2 M3 EDA and Visualisation.ipynb
│   ├── M4 Machine Learning.ipynb
│   ├── m1_appendix_create.sql
│   ├── m1_appendix_transform.sql
│   └── Problem statement.md
├── 🎨 assets/                  # Images and static resources
├── 🧪 tests/                   # Unit tests
└── 📋 requirements.txt         # Python dependencies
```

## 🔬 Analysis Framework

### Module 1: Data Fundamentals & SQL
- **Objective**: Data sourcing, transformation, and quality validation
- **Key Outputs**: Wide→Long format transformation, SQL schemas
- **Tools**: SQL, Database design, ETL pipelines

### Module 2 & 3: EDA and Visualization
- **Objective**: Exploratory data analysis and pattern discovery
- **Key Outputs**: Trend analysis, demographic insights, outlier detection
- **Tools**: Pandas, Plotly, Statistical analysis

### Module 4: Machine Learning & Forecasting
- **Objective**: Predict 2025 unemployment trends and identify high-risk groups
- **Key Outputs**: KNN regression forecasts, Logistic regression risk scoring
- **Tools**: Scikit-learn, Time-series analysis, Classification models

### Module 5: Business Intelligence
- **Objective**: Interactive dashboards and executive reporting
- **Key Outputs**: Power BI dashboards, Strategic recommendations
- **Tools**: Power BI, Streamlit dashboards

## 🎯 Key Findings & Results

### 📈 Predictive Models Performance
- **KNN Regression**: 9.81% MAPE, 0.34 MAE for 2025 unemployment forecasts
- **Logistic Regression**: 75% accuracy, 0.73 ROC-AUC for risk classification

### ⚠️ High-Risk Occupations for 2025
1. **Service & Sales Workers** (99.9% risk probability)
2. **Cleaners, Labourers & Related Workers** (99.7% risk probability)
3. **Craftsmen & Related Trades Workers** (99.5% risk probability)

### 💡 Strategic Insights
- **COVID Impact**: 2020-2021 marked peak unemployment across all sectors
- **Recovery Paradox**: 2025 forecasts show recovery, but high-risk groups remain vulnerable
- **Demographic Patterns**: Mature workers + low education = 5-7x higher unemployment risk
- **Policy Target**: 800,000+ workers in high-risk categories need resilience programs

## 🚀 Technology Stack

### Frontend & Visualization
- **Streamlit**: Interactive web application framework
- **Plotly**: Interactive charts and visualizations
- **Power BI**: Executive dashboards and reporting

### Data Processing & Analysis
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **SQLAlchemy**: Database connectivity and ORM
- **MySQL**: Data storage and management

### Machine Learning & Statistics
- **Scikit-learn**: Machine learning models (KNN, Logistic Regression)
- **Matplotlib**: Statistical visualizations
- **Time-series Analysis**: Trend forecasting and pattern recognition

### Development & Testing
- **pytest**: Unit testing framework
- **Git**: Version control and collaboration
- **Virtual Environment**: Dependency management

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- MySQL database (optional, for full functionality)
- Git for cloning the repository

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/jetLZJ/SIT-DA-G6.git
   cd SIT-DA-G6
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure database connection (optional)**
   ```bash
   # Create .streamlit/secrets.toml file
   mkdir .streamlit
   echo 'DB_CONNECTION_STRING = "mysql://user:password@host:port/database"' > .streamlit/secrets.toml
   ```

5. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

6. **Access the application**
   - Open your browser to `http://localhost:8501`
   - Navigate through the different modules using the sidebar

## 📊 Data Sources & Schema

### Primary Data Sources
- **Ministry of Manpower (MOM)**: Labour Force Singapore datasets (2014-2024)
- **Resident Unemployment by Qualification**: Education-employment relationships
- **Age and Gender Distribution**: Demographic unemployment patterns
- **PMET vs Non-PMET Classification**: Professional vs non-professional analysis

### Key Data Tables
- `unemployment_rate_by_occupation_long`: Time-series occupation data
- `unemployed_by_qualification_long`: Education level analysis
- `unemployed_by_age_sex_long`: Demographic breakdowns
- `unemployed_by_previous_occupation_sex_long`: Career transition insights

## 🎯 Usage Guide

### For Analysts & Researchers
1. **Module 2 (EDA)**: Explore unemployment trends and patterns
2. **Module 4 (ML)**: Run predictive models and risk assessments
3. **Dashboard**: Create custom visualizations and filters

### For Policymakers & Executives
1. **Overview**: Review key findings and executive summary
2. **Presentation Mode**: Navigate structured slide presentation
3. **Power BI Dashboard**: Interactive policy-focused dashboards

### For Developers
1. **Data Schema**: Understand database structure and transformations
2. **API Documentation**: Explore data loading and processing functions
3. **Test Suite**: Run unit tests and validation checks

## 🎯 Strategic Recommendations

Based on our analysis, we recommend four strategic priorities:

### 🎯 Priority 1: Resilience-Building (Q1 2025)
- **Target**: 800,000+ high-risk workers
- **Budget**: S$50M investment
- **Actions**: Digital literacy, service sector adaptation, trades modernization

### 📡 Priority 2: Early Warning System (Q2-Q3 2025)
- **Goal**: Real-time unemployment monitoring
- **Budget**: S$5M investment
- **Impact**: 12+ month policy foresight

### 🤝 Priority 3: Placement Support (Q4 2025+)
- **Target**: 120,000 displaced workers
- **Budget**: S$30M investment
- **Outcome**: 75% placement rate

### 🎓 Priority 4: Curriculum Redesign (2025-2026)
- **Goal**: Future-ready workforce development
- **Budget**: Existing education budget reallocation
- **Impact**: Structural unemployment prevention


## 📚 Documentation & Resources

### Project Artifacts
- **Module 1 Documentation**: Data fundamentals and SQL transformation guide
- **Jupyter Notebooks**: Detailed analysis workflows (M2-M4)
- **SQL Scripts**: Database creation and transformation queries
- **Presentation Slides**: Executive summary and key findings

## 🤝 Contributing

### SIT Data Analytics Group 6

This project was developed as part of the Singapore Institute of Technology (SIT) Data Analytics capstone program by **Group 6**. Our team brought together diverse expertise in data science, business analytics, and workforce policy to deliver comprehensive insights for Singapore's labour market planning.

#### Team Contributions
- **Data Engineering & ETL**: Database design, SQL transformations, data quality validation
- **Statistical Analysis**: Exploratory data analysis, trend identification, hypothesis testing  
- **Machine Learning**: Predictive modeling, risk assessment, forecast validation
- **Business Intelligence**: Dashboard development, executive reporting, policy recommendations

