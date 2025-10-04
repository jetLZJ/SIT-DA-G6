# Data Fundamentals & SQL

We imported the tables (see Appendix 1: SQL Code for Database Creation), which were initially structured in a “wide” format (e.g., with years represented as columns). Such a format is suboptimal for relational database operations. Consequently, we transformed the data from this wide format into a “long” format, where each row corresponds to a single data point identified by year, occupation, or category. This restructuring facilitates easier data manipulation and analysis by enhancing the flexibility of querying and joining operations.

The SQL script for this transformation is provided in Appendix 2: SQL Code for Data Transformation. The approach involves creating a new table by stacking multiple SELECT statements via UNION ALL. Each statement extracts data from one of the original wide columns (e.g., year\_2014, year\_2015), renames the value as a generic measure (unemployed\_count), and appends a fixed indicator for the corresponding category (e.g., 2014, 2015). By aggregating all these queries, the output is a single long-format table in which each row represents a unique category–value pair. This method is generalizable and can be employed for pivoting other tables with columns representing various time periods, occupations, or categories.

The below are the tables after transformation:

1. **UNEMPLOYED RESIDENTS AGED FIFTEEN YEARS AND OVER BY AGE AND SEX, 2014 \- 2024**	

**Table Name :	unemployed\_by\_age\_sex\_long**						

| Fields | Content |
| :---- | :---- |
| Year | 2014 to 2024 |
| age\_group | 15 \- 29 30 \- 39 40 \- 49 50 \- 59 60 \- 69 70 & Over |
| unemployed\_count | count |

2. **UNEMPLOYED RESIDENTS AGED FIFTEEN YEARS AND OVER BY HIGHEST QUALIFICATION ATTAINED AND SEX, 2014 \- 2024**	

**Table Name :	unemployed\_by\_qualification\_sex\_long**					

| Fields | Content |
| :---- | :---- |
| year | 2014 to 2024 |
| gender | Male Female |
| education | Below Secondary Degree Diploma & Professional Qualification Post-Secondary (Non-Tertiary) Secondary |
| unemployed\_count | count |

3. **UNEMPLOYED RESIDENTS AGED FIFTEEN YEARS AND OVER BY MARITAL STATUS AND SEX, 2014 \- 2024**	

**Table Name :	unemployed\_by\_marital\_status\_sex\_long**

| Fields | Content |
| :---- | :---- |
| year | 2014 to 2024 |
| gender | Male Female |
| marital\_status | Married Single Widowed / Divorced |
| Unemployed\_count | count |

												

4. **RESIDENT UNEMPLOYMENT RATE BY OCCUPATION, 2014 \- 2024**	

**Table Name : unemployment\_rate\_by\_occupation\_long**

| Fields | Content |
| :---- | :---- |
| year | 2014 to 2024 |
| occupation | Managers & Administrators (Including Working Proprietors) Professionals Associate Professionals & Technicians Clerical Support Workers Service & Sales Workers Craftsmen & Related Trades Workers Plant & Machine Operators & Assemblers Cleaners, Labourers & Related Workers |
| unemployed\_rate | percentage |

5. **UNEMPLOYED RESIDENTS AGED FIFTEEN YEARS AND OVER WHO HAVE WORKED BEFORE BY PREVIOUS OCCUPATION AND SEX, 2014 – 2024**

**Table Name : unemployed\_by\_previous\_occupation\_sex\_long**

| Fields | Content |
| :---- | :---- |
| year | 2014 to 2024 |
| occupation | Managers & Administrators (Including Working Proprietors) Professionals Associate Professionals & Technicians Clerical Support Workers Service & Sales Workers Craftsmen & Related Trades Workers Plant & Machine Operators & Assemblers Cleaners, Labourers & Related Workers Others |
| unemployed\_count | count |

6. **UNEMPLOYED RESIDENT PMETs AND NON-PMETs AGED FIFTEEN YEARS AND OVER WHO HAVE WORKED BEFORE BY AGE, 2014 – 2024**

**Table Name : unemployed\_pmets\_by\_age\_long**

| Fields | Content |
| :---- | :---- |
| year | 2014 to 2024 |
| pmet\_nonpmet | Male Female |
| age\_group | 15 \- 29 30 \- 39 40 \- 49 50 \- 59 60 \- 69 70 & Over |
| unemployed\_count | count |

7. **LONG-TERM UNEMPLOYED RESIDENT PMETs AND NON-PMETs AGED FIFTEEN YEARS AND OVER WHO HAVE WORKED BEFORE BY AGE, 2014 – 2024**

**Table Name : long\_term\_unemployed\_pmets\_by\_age\_long**

| Fields | Content |
| :---- | :---- |
| year | 2014 to 2024 |
| pmet\_nonpmet | PMET Non-PMET |
| age\_group | 15 \- 29 30 \- 39 40 \- 49 50 \- 59 60 \- 69 70 & Over |
| unemployed\_count | count |

# Occupation Patterns Over Time

Using the table “Resident unemployment rate By Occupation, 2014 – 2024 ” we examined how the relationship between occupation and unemployment evolved across three periods. Unemployment rates varied markedly across occupations from 2014 to 2024, with clerical and service roles facing the highest and most persistent unemployment. The COVID-19 pandemic caused a pronounced spike in joblessness across all groups, especially in customer-facing and office-support occupations. 

| Occupation | 2014-2016 | 2017-2019 | 2020-2021 | 2022-2024 |
| :---- | :---: | :---: | :---: | :---: |
| **Associate Professionals & Technicians** | 3.23 | 3.30 | 4.00 | 2.77 |
| **Cleaners, Labourers & Related Workers** | 4.00 | 3.97 | 5.60 | 3.57 |
| **Clerical Support Workers** | 5.33 | 5.67 | 7.15 | 5.47 |
| **Craftsmen & Related Trades Workers** | 3.00 | 3.43 | 3.95 | 2.50 |
| **Managers & Administrators (Incl. Prop.)** | 2.60 | 2.63 | 2.80 | 2.23 |
| **Plant & Machine Operators & Assemblers** | 3.20 | 3.13 | 3.85 | 2.73 |
| **Professionals** | 2.77 | 2.90 | 3.45 | 2.57 |
| **Service & Sales Workers** | 5.17 | 5.40 | 7.05 | 4.10 |

* **Clerical Support Workers** and **Service & Sales Workers** consistently recorded the highest unemployment rates through all periods, signaling ongoing vulnerability.

* The **2020–2021 COVID-19** period saw unemployment peak sharply in these groups (over 7%), reflecting heavy disruption due to social distancing and economic shutdowns.

* More skilled roles such as **Professionals and Managers & Administrators** had comparatively lower unemployment rates and smaller increases during the pandemic.

* **Post-2022**, unemployment fell across all occupations but remained relatively high for clerical and service roles, suggesting persistent labor market pressures.

* **Trades and technical occupations** experienced faster recovery and lower sustained unemployment rates, indicating greater resilience against automation and economic cycles.

These insights reveal both pandemic effects and longer-term structural shifts shaping employment across occupations.

# Education Impact Over Time

We examined how the relationship between educational attainment and unemployment evolved across three periods. Overall, degree holders consistently show lower unemployment after crises, suggesting resilience and better employment outcomes. For mid-level education, Diploma and post-secondary non-tertiary holders, they are more exposed to economic shocks like COVID-19.

| Education | 2014-2016  | 2017-2019  | 2020-2021  | 2022-2024  |
| :---- | :---: | :---: | :---: | :---: |
| **Below Secondary** | 3.55 | 3.59 | 4.87 | 2.96 |
| **Secondary** | 3.97 | 4.08 | 5.44 | 3.58 |
| **Post-Secondary (Non-Tertiary)** | 4.04 | 4.83 | 5.68 | 4.14 |
| **Diploma & Professional Qualification** | 3.92 | 4.34 | 5.41 | 3.83 |
| **Degree** | 3.90 | 3.94 | 4.32 | 3.16 |

* **Higher-educated workers:** Higher education generally provides stronger employment outcomes, especially in recovery periods. The COVID-19 shock disproportionately affected mid-level qualifications (diploma & post-secondary non-tertiary) more than degree holders.

* **Lower-educated workers:** Lower-educated workers experienced the most volatility during COVID-19 but also showed quick post-pandemic recovery. Over the long term, higher qualifications still correlate with stronger employment stability, but lower-educated groups can benefit from recovery phases, possibly due to labor-intensive sectors bouncing back.

* **Impact of COVID-19:** Mid-level education (Diploma, Post-Secondary Non-Tertiary, Secondary) experienced the highest spikes, indicating that roles requiring intermediate skills were most affected. Degree holders were relatively more insulated (4.32%) compared to other higher-educated groups, while below-secondary workers (4.87%) saw a large relative increase, reflecting exposure in lower-skill service and labor-intensive sectors

* **Post\-2022 trends:** Rapid recovery across all groups; unemployment fell below pre-COVID levels for below-secondary and secondary workers. Degree holders continue to enjoy the lowest unemployment (3.16%), reinforcing the long-term advantage of higher education in employment stability.

# Combined Gender & Education Trends

A dedicated query was constructed to analyze changes in unemployment gaps between men and women across different education groups. The procedure involved summing total male and female unemployed counts by year and education level, then calculating the absolute gender gap for each period (first, recent, supplementary). Comparing the average absolute gaps across these periods provides a measure of Gap Reduction, indicating whether the disparity has narrowed (positive values) or widened (negative values).

| Education | Average Gap 2014-2016 | Average Gap 2020-2021 | Average Gap 2022-2024 | Gap Reduction? |
| :---- | :---: | :---: | :---: | :---: |
| **Below Secondary** | 2.40 | 2.80 | 0.77 | 1.63 |
| **Degree** | 1.07 | 0.10 | 0.43 | 0.63 |
| **Secondary** | 0.63 | 0.55 | 0.50 | 0.13 |
| **Post-Secondary (Non-Tertiary)** | 1.07 | 0.20 | 1.30 | \-0.23 |
| **Diploma & Professional Qualification** | 0.47 | 2.20 | 0.83 | \-0.37 |

Results indicate that degree-level education has been most effective in closing gender unemployment gaps, while gains among individuals with below-secondary education are fragile and susceptible to reversal during economic shocks. Other qualification levels, particularly diploma and professional certifications, have witnessed widening gaps, reflecting uneven progress and persistent structural inequalities in the labor market.

Specifically, for degree holders, the gender gap steadily declined from 1.07 in 2014–2016 to 0.43 in 2022–2024, reaching a low of 0.1 during the COVID-19 period—representing the most consistent long-term improvement. A similar trend was noted for those with below secondary education, whose gap shrank significantly from 2.4 to 0.77, albeit interrupted by a spike to 2.8 amid the pandemic.

At the secondary education level, the gender gap remained small and relatively stable, varying slightly between 0.63 and 0.55, suggesting a segment of the labor market approaching balance. In contrast, individuals with post-secondary (non-tertiary) qualifications experienced deteriorating outcomes, with the gap increasing from 1.07 in the initial period to 1.3 recently, despite a brief narrowing to 0.2 during the pandemic.

The most pronounced reversal was observed among diploma and professional qualification holders, whose gender gap widened consistently, from 0.47 in 2014–2016 to 2.2 in 2022–2024.

# Appendices	

## Appendix 1 : Create Database SQL

**DROP** DATABASE **IF** **EXISTS** labourtrendsdb;  
**CREATE** DATABASE labourtrendsDB;  
**USE** labourtrendsDB;

**DROP** **TABLE** **IF** **EXISTS** unemployed\_by\_age\_sex\_wide;  
**CREATE** **TABLE** unemployed\_by\_age\_sex\_wide (  
    gender VARCHAR(20),  
    age\_group VARCHAR(20),  
    year\_2014 DECIMAL(5,1),  
    year\_2015 DECIMAL(5,1),  
    year\_2016 DECIMAL(5,1),  
    year\_2017 DECIMAL(5,1),  
    year\_2018 DECIMAL(5,1),  
    year\_2019 DECIMAL(5,1),  
    year\_2020 DECIMAL(5,1),  
    year\_2021 DECIMAL(5,1),  
    year\_2022 DECIMAL(5,1),  
    year\_2023 DECIMAL(5,1),  
    year\_2024 DECIMAL(5,1)  
);

**DROP** **TABLE** **IF** **EXISTS** unemployed\_by\_qualification\_sex\_wide;  
**CREATE** **TABLE** unemployed\_by\_qualification\_sex\_wide (  
    gender VARCHAR(20),  
    education VARCHAR(50),  
    year\_2014 DECIMAL(5,1),  
    year\_2015 DECIMAL(5,1),  
    year\_2016 DECIMAL(5,1),  
    year\_2017 DECIMAL(5,1),  
    year\_2018 DECIMAL(5,1),  
    year\_2019 DECIMAL(5,1),  
    year\_2020 DECIMAL(5,1),  
    year\_2021 DECIMAL(5,1),  
    year\_2022 DECIMAL(5,1),  
    year\_2023 DECIMAL(5,1),  
    year\_2024 DECIMAL(5,1)  
);

**DROP** **TABLE** **IF** **EXISTS** unemployed\_by\_marital\_status\_sex\_wide;  
**CREATE** **TABLE** unemployed\_by\_marital\_status\_sex\_wide (  
    gender VARCHAR(20),  
    marital\_status VARCHAR(20),  
    year\_2014 DECIMAL(5,1),  
    year\_2015 DECIMAL(5,1),  
    year\_2016 DECIMAL(5,1),  
    year\_2017 DECIMAL(5,1),  
    year\_2018 DECIMAL(5,1),  
    year\_2019 DECIMAL(5,1),  
    year\_2020 DECIMAL(5,1),  
    year\_2021 DECIMAL(5,1),  
    year\_2022 DECIMAL(5,1),  
    year\_2023 DECIMAL(5,1),  
    year\_2024 DECIMAL(5,1)  
);

**DROP** **TABLE** **IF** **EXISTS** unemployment\_rate\_by\_occupation\_wide;  
**CREATE** **TABLE** unemployment\_rate\_by\_occupation\_wide (  
year INT,  
Managers\_N\_Administrators\_Including\_Working\_Proprietors DECIMAL(5,1),  
Professionals DECIMAL(5,1),  
Associate\_Professionals\_N\_Technicians DECIMAL(5,1),  
Clerical\_Support\_Workers DECIMAL(5,1),  
Service\_N\_Sales\_Workers DECIMAL(5,1),  
Craftsmen\_N\_Related\_Trades\_Workers DECIMAL(5,1),  
Plant\_N\_Machine\_Operators\_N\_Assemblers DECIMAL(5,1),  
Cleaners\_Labourers\_N\_Related\_Workers DECIMAL(5,1)  
);

**DROP** **TABLE** **IF** **EXISTS** unemployed\_by\_previous\_occupation\_sex\_wide;  
**CREATE** **TABLE** unemployed\_by\_previous\_occupation\_sex\_wide (  
year INT,  
gender VARCHAR(20),  
Managers\_N\_Administrators\_Including\_Working\_Proprietors DECIMAL(5,1),  
Professionals DECIMAL(5,1),  
Associate\_Professionals\_N\_Technicians DECIMAL(5,1),  
Clerical\_Support\_Workers DECIMAL(5,1),  
Service\_N\_Sales\_Workers DECIMAL(5,1),  
Craftsmen\_N\_Related\_Trades\_Workers DECIMAL(5,1),  
Plant\_N\_Machine\_Operators\_N\_Assemblers DECIMAL(5,1),  
Cleaners\_Labourers\_N\_Related\_Workers DECIMAL(5,1),  
\`Others\` DECIMAL(5,1)  
);

**DROP** **TABLE** **IF** **EXISTS** unemployed\_pmets\_by\_age\_wide;  
**CREATE** **TABLE** unemployed\_pmets\_by\_age\_wide (  
    pmets\_status VARCHAR(20),  
    age\_group VARCHAR(20),  
    year\_2014 DECIMAL(5,1),  
    year\_2015 DECIMAL(5,1),  
    year\_2016 DECIMAL(5,1),  
    year\_2017 DECIMAL(5,1),  
    year\_2018 DECIMAL(5,1),  
    year\_2019 DECIMAL(5,1),  
    year\_2020 DECIMAL(5,1),  
    year\_2021 DECIMAL(5,1),  
    year\_2022 DECIMAL(5,1),  
    year\_2023 DECIMAL(5,1),  
    year\_2024 DECIMAL(5,1)  
);

**DROP** **TABLE** **IF** **EXISTS** long\_term\_unemployed\_pmets\_by\_age\_wide;  
**CREATE** **TABLE** long\_term\_unemployed\_pmets\_by\_age\_wide (  
    pmets\_status VARCHAR(20),  
    age\_group VARCHAR(20),  
    year\_2014 DECIMAL(5,1),  
    year\_2015 DECIMAL(5,1),  
    year\_2016 DECIMAL(5,1),  
    year\_2017 DECIMAL(5,1),  
    year\_2018 DECIMAL(5,1),  
    year\_2019 DECIMAL(5,1),  
    year\_2020 DECIMAL(5,1),  
    year\_2021 DECIMAL(5,1),  
    year\_2022 DECIMAL(5,1),  
    year\_2023 DECIMAL(5,1),  
    year\_2024 DECIMAL(5,1)  
);

**LOAD** **DATA** **INFILE** 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/k2unemployed\_by\_age\_sex\_wide.csv'  
**INTO** **TABLE** unemployed\_by\_age\_sex\_wide  
**FIELDS** **TERMINATED** **BY** ','  
**LINES** **TERMINATED** **BY** '\\r\\n'  
**IGNORE** 1 **ROWS**;

**LOAD** **DATA** **INFILE** 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/k3unemployed\_by\_qualification\_sex\_wide.csv'  
**INTO** **TABLE** unemployed\_by\_qualification\_sex\_wide  
**FIELDS** **TERMINATED** **BY** ','  
**LINES** **TERMINATED** **BY** '\\r\\n'  
**IGNORE** 1 **ROWS**;

**LOAD** **DATA** **INFILE** 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/k6unemployed\_by\_marital\_status\_sex\_wide.csv'  
**INTO** **TABLE** unemployed\_by\_marital\_status\_sex\_wide  
**FIELDS** **TERMINATED** **BY** ','  
**LINES** **TERMINATED** **BY** '\\r\\n'  
**IGNORE** 1 **ROWS**;

**LOAD** **DATA** **INFILE** 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/k14unemployment\_rate\_by\_occupation\_wide.csv'  
**INTO** **TABLE** unemployment\_rate\_by\_occupation\_wide  
**FIELDS** **TERMINATED** **BY** ','  
**LINES** **TERMINATED** **BY** '\\r\\n'  
**IGNORE** 1 **ROWS**;

**LOAD** **DATA** **INFILE** 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/k15unemployed\_by\_previous\_occupation\_sex\_wide.csv'  
**INTO** **TABLE** unemployed\_by\_previous\_occupation\_sex\_wide  
**FIELDS** **TERMINATED** **BY** ','  
**LINES** **TERMINATED** **BY** '\\r\\n'  
**IGNORE** 1 **ROWS**;

**LOAD** **DATA** **INFILE** 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/k16unemployed\_pmets\_by\_age\_wide.csv'  
**INTO** **TABLE** unemployed\_pmets\_by\_age\_wide  
**FIELDS** **TERMINATED** **BY** ','  
**LINES** **TERMINATED** **BY** '\\r\\n'  
**IGNORE** 1 **ROWS**;

**LOAD** **DATA** **INFILE** 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/k17long\_term\_unemployed\_pmets\_by\_age\_wide.csv'  
**INTO** **TABLE** long\_term\_unemployed\_pmets\_by\_age\_wide  
**FIELDS** **TERMINATED** **BY** ','  
**LINES** **TERMINATED** **BY** '\\r\\n'  
**IGNORE** 1 **ROWS**;

## Appendix 2 : SQL Code for Data Transformation

**USE** labourtrendsDB;

\-- Transform unemployed\_by\_age\_sex\_wide  
**DROP** **TABLE** **IF** **EXISTS** unemployed\_by\_age\_sex\_long;  
**CREATE** **TABLE** unemployed\_by\_age\_sex\_long **AS**  
    **select** 2014 **AS** year, gender, age\_group, year\_2014 **As** unemployed\_count  
    **from** unemployed\_by\_age\_sex\_wide  
    **UNION** **ALL**   
    **select** 2015, gender, age\_group, year\_2015  
    **from** unemployed\_by\_age\_sex\_wide   
    **UNION** **ALL**    
    **select** 2016, gender, age\_group, year\_2016   
    **from** unemployed\_by\_age\_sex\_wide   
    **UNION** **ALL**    
    **select** 2017, gender, age\_group, year\_2017  
    **from** unemployed\_by\_age\_sex\_wide   
    **UNION** **ALL**    
    **select** 2018, gender, age\_group, year\_2018  
    **from** unemployed\_by\_age\_sex\_wide   
    **UNION** **ALL**    
    **select** 2019, gender, age\_group, year\_2019  
    **from** unemployed\_by\_age\_sex\_wide   
    **UNION** **ALL**    
    **select** 2020, gender, age\_group, year\_2020  
    **from** unemployed\_by\_age\_sex\_wide   
    **UNION** **ALL**    
    **select** 2021, gender, age\_group, year\_2021  
    **from** unemployed\_by\_age\_sex\_wide   
    **UNION** **ALL**   
    **select** 2022, gender, age\_group, year\_2022  
    **from** unemployed\_by\_age\_sex\_wide   
    **UNION** **ALL**   
    **select** 2023, gender, age\_group, year\_2023  
    **from** unemployed\_by\_age\_sex\_wide   
    **UNION** **ALL**   
    **select** 2024, gender, age\_group, year\_2024  
    **from** unemployed\_by\_age\_sex\_wide;

\-- Transform unemployed\_by\_qualification\_sex\_wide  
**DROP** **TABLE** **IF** **EXISTS** unemployed\_by\_qualification\_sex\_long;  
**CREATE** **TABLE** unemployed\_by\_qualification\_sex\_long **AS**  
    **select** 2014 **AS** year, gender, education, year\_2014 **As** unemployed\_count  
    **from** unemployed\_by\_qualification\_sex\_wide  
    **UNION** **ALL**   
    **select** 2015, gender, education, year\_2015  
    **from** unemployed\_by\_qualification\_sex\_wide   
    **UNION** **ALL**    
    **select** 2016, gender, education, year\_2016   
    **from** unemployed\_by\_qualification\_sex\_wide   
    **UNION** **ALL**    
    **select** 2017, gender, education, year\_2017  
    **from** unemployed\_by\_qualification\_sex\_wide   
    **UNION** **ALL**    
    **select** 2018, gender, education, year\_2018  
    **from** unemployed\_by\_qualification\_sex\_wide   
    **UNION** **ALL**    
    **select** 2019, gender, education, year\_2019  
    **from** unemployed\_by\_qualification\_sex\_wide   
    **UNION** **ALL**    
    **select** 2020, gender, education, year\_2020  
    **from** unemployed\_by\_qualification\_sex\_wide   
    **UNION** **ALL**    
    **select** 2021, gender, education, year\_2021  
    **from** unemployed\_by\_qualification\_sex\_wide   
    **UNION** **ALL**   
    **select** 2022, gender, education, year\_2022  
    **from** unemployed\_by\_qualification\_sex\_wide   
    **UNION** **ALL**   
    **select** 2023, gender, education, year\_2023  
    **from** unemployed\_by\_qualification\_sex\_wide   
    **UNION** **ALL**   
    **select** 2024, gender, education, year\_2024  
    **from** unemployed\_by\_qualification\_sex\_wide;

\-- Transform unemployed\_by\_marital\_status\_sex\_wide  
**DROP** **TABLE** **IF** **EXISTS** unemployed\_by\_marital\_status\_sex\_long;  
    **CREATE** **TABLE** unemployed\_by\_marital\_status\_sex\_long **AS**  
    **select** 2014 **AS** year, gender, marital\_status, year\_2014 **As** unemployed\_count  
    **from** unemployed\_by\_marital\_status\_sex\_wide  
    **UNION** **ALL**   
    **select** 2015, gender, marital\_status, year\_2015  
    **from** unemployed\_by\_marital\_status\_sex\_wide   
    **UNION** **ALL**    
    **select** 2016, gender, marital\_status, year\_2016   
    **from** unemployed\_by\_marital\_status\_sex\_wide   
    **UNION** **ALL**    
    **select** 2017, gender, marital\_status, year\_2017  
    **from** unemployed\_by\_marital\_status\_sex\_wide   
    **UNION** **ALL**    
    **select** 2018, gender, marital\_status, year\_2018  
    **from** unemployed\_by\_marital\_status\_sex\_wide   
    **UNION** **ALL**    
    **select** 2019, gender, marital\_status, year\_2019  
    **from** unemployed\_by\_marital\_status\_sex\_wide   
    **UNION** **ALL**    
    **select** 2020, gender, marital\_status, year\_2020  
    **from** unemployed\_by\_marital\_status\_sex\_wide   
    **UNION** **ALL**    
    **select** 2021, gender, marital\_status, year\_2021  
    **from** unemployed\_by\_marital\_status\_sex\_wide   
    **UNION** **ALL**   
    **select** 2022, gender, marital\_status, year\_2022  
    **from** unemployed\_by\_marital\_status\_sex\_wide   
    **UNION** **ALL**   
    **select** 2023, gender, marital\_status, year\_2023  
    **from** unemployed\_by\_marital\_status\_sex\_wide   
    **UNION** **ALL**   
    **select** 2024, gender, marital\_status, year\_2024  
    **from** unemployed\_by\_marital\_status\_sex\_wide;

\-- Transform unemployment\_rate\_by\_occupation\_wide  
**DROP** **TABLE** **IF** **EXISTS** unemployment\_rate\_by\_occupation\_long;  
**CREATE** **TABLE** unemployment\_rate\_by\_occupation\_long **AS**  
    **SELECT** year, "Managers & Administrators (Including Working Proprietors)" **As** occupation, Managers\_N\_Administrators\_Including\_Working\_Proprietors **As** unemployed\_rate  
    **FROM** unemployment\_rate\_by\_occupation\_wide  
    **UNION** **ALL**   
    **SELECT** year, "Professionals" **As** occupation, Professionals  
    **FROM** unemployment\_rate\_by\_occupation\_wide  
    **UNION** **ALL**   
    **SELECT** year, "Associate Professionals & Technicians" **As** occupation, Associate\_Professionals\_N\_Technicians  
    **FROM** unemployment\_rate\_by\_occupation\_wide  
    **UNION** **ALL**   
    **SELECT** year, "Clerical Support Workers" **As** occupation, Clerical\_Support\_Workers  
    **FROM** unemployment\_rate\_by\_occupation\_wide  
    **UNION** **ALL**   
    **SELECT** year, "Service & Sales Workers" **As** occupation, Service\_N\_Sales\_Workers  
    **FROM** unemployment\_rate\_by\_occupation\_wide  
    **UNION** **ALL**   
    **SELECT** year, "Craftsmen & Related Trades Workers" **As** occupation, Craftsmen\_N\_Related\_Trades\_Workers  
    **FROM** unemployment\_rate\_by\_occupation\_wide  
    **UNION** **ALL**   
    **SELECT** year, "Plant & Machine Operators & Assemblers" **As** occupation, Plant\_N\_Machine\_Operators\_N\_Assemblers  
    **FROM** unemployment\_rate\_by\_occupation\_wide  
    **UNION** **ALL**   
    **SELECT** year, "Cleaners Labourers & Related Workers" **As** occupation, Cleaners\_Labourers\_N\_Related\_Workers  
    **FROM** unemployment\_rate\_by\_occupation\_wide;  
      
\-- Transform unemployed\_by\_previous\_occupation\_sex\_wide      
**DROP** **TABLE** **IF** **EXISTS** unemployed\_by\_previous\_occupation\_sex\_long;  
**CREATE** **TABLE** unemployed\_by\_previous\_occupation\_sex\_long **AS**  
    **SELECT** year, gender, "Managers & Administrators (Including Working Proprietors)" **As** occupation, Managers\_N\_Administrators\_Including\_Working\_Proprietors **As** unemployed\_count  
    **FROM** unemployed\_by\_previous\_occupation\_sex\_wide  
    **UNION** **ALL**   
    **SELECT** year, gender, "Professionals" **As** occupation, Professionals  
    **FROM** unemployed\_by\_previous\_occupation\_sex\_wide  
    **UNION** **ALL**   
    **SELECT** year, gender, "Associate Professionals & Technicians" **As** occupation, Associate\_Professionals\_N\_Technicians  
    **FROM** unemployed\_by\_previous\_occupation\_sex\_wide  
    **UNION** **ALL**   
    **SELECT** year, gender, "Clerical Support Workers" **As** occupation, Clerical\_Support\_Workers  
    **FROM** unemployed\_by\_previous\_occupation\_sex\_wide  
    **UNION** **ALL**   
    **SELECT** year, gender, "Service & Sales Workers" **As** occupation, Service\_N\_Sales\_Workers  
    **FROM** unemployed\_by\_previous\_occupation\_sex\_wide  
    **UNION** **ALL**   
    **SELECT** year, gender, "Craftsmen & Related Trades Workers" **As** occupation, Craftsmen\_N\_Related\_Trades\_Workers  
    **FROM** unemployed\_by\_previous\_occupation\_sex\_wide  
    **UNION** **ALL**   
    **SELECT** year, gender, "Plant & Machine Operators & Assemblers" **As** occupation, Plant\_N\_Machine\_Operators\_N\_Assemblers  
    **FROM** unemployed\_by\_previous\_occupation\_sex\_wide  
    **UNION** **ALL**   
    **SELECT** year, gender, "Cleaners Labourers & Related Workers" **As** occupation, Cleaners\_Labourers\_N\_Related\_Workers  
    **FROM** unemployed\_by\_previous\_occupation\_sex\_wide  
    **UNION** **ALL**   
    **SELECT** year, gender, "Others" **As** occupation, Others  
    **FROM** unemployed\_by\_previous\_occupation\_sex\_wide;  
      
\-- Transform unemployed\_pmets\_by\_age\_wide     
**DROP** **TABLE** **IF** **EXISTS** unemployed\_pmets\_by\_age\_long;  
**CREATE** **TABLE** unemployed\_pmets\_by\_age\_long **AS**  
    **select** 2014 **AS** year, pmets\_status, age\_group, year\_2014 **As** unemployed\_count  
    **from** unemployed\_pmets\_by\_age\_wide  
    **UNION** **ALL**   
    **select** 2015, pmets\_status, age\_group, year\_2015  
    **from** unemployed\_pmets\_by\_age\_wide   
    **UNION** **ALL**    
    **select** 2016, pmets\_status, age\_group, year\_2016   
    **from** unemployed\_pmets\_by\_age\_wide  
    **UNION** **ALL**    
    **select** 2017, pmets\_status, age\_group, year\_2017  
    **from** unemployed\_pmets\_by\_age\_wide   
    **UNION** **ALL**    
    **select** 2018, pmets\_status, age\_group, year\_2018  
    **from** unemployed\_pmets\_by\_age\_wide   
    **UNION** **ALL**    
    **select** 2019, pmets\_status, age\_group, year\_2019  
    **from** unemployed\_pmets\_by\_age\_wide   
    **UNION** **ALL**    
    **select** 2020, pmets\_status, age\_group, year\_2020  
    **from** unemployed\_pmets\_by\_age\_wide   
    **UNION** **ALL**    
    **select** 2021, pmets\_status, age\_group, year\_2021  
    **from** unemployed\_pmets\_by\_age\_wide  
    **UNION** **ALL**   
    **select** 2022, pmets\_status, age\_group, year\_2022  
    **from** unemployed\_pmets\_by\_age\_wide   
    **UNION** **ALL**   
    **select** 2023, pmets\_status, age\_group, year\_2023  
    **from** unemployed\_pmets\_by\_age\_wide   
    **UNION** **ALL**   
    **select** 2024, pmets\_status, age\_group, year\_2024  
    **from** unemployed\_pmets\_by\_age\_wide;

\-- Transform long\_term\_unemployed\_pmets\_by\_age\_wide      
**DROP** **TABLE** **IF** **EXISTS** long\_term\_unemployed\_pmets\_by\_age\_long;  
**CREATE** **TABLE** long\_term\_unemployed\_pmets\_by\_age\_long **AS**  
    **select** 2014 **AS** year, pmets\_status, age\_group, year\_2014 **As** unemployed\_count  
    **from** long\_term\_unemployed\_pmets\_by\_age\_wide  
    **UNION** **ALL**   
    **select** 2015, pmets\_status, age\_group, year\_2015  
    **from** long\_term\_unemployed\_pmets\_by\_age\_wide   
    **UNION** **ALL**    
    **select** 2016, pmets\_status, age\_group, year\_2016   
    **from** long\_term\_unemployed\_pmets\_by\_age\_wide  
    **UNION** **ALL**    
    **select** 2017, pmets\_status, age\_group, year\_2017  
    **from** long\_term\_unemployed\_pmets\_by\_age\_wide   
    **UNION** **ALL**    
    **select** 2018, pmets\_status, age\_group, year\_2018  
    **from** long\_term\_unemployed\_pmets\_by\_age\_wide   
    **UNION** **ALL**    
    **select** 2019, pmets\_status, age\_group, year\_2019  
    **from** long\_term\_unemployed\_pmets\_by\_age\_wide   
    **UNION** **ALL**    
    **select** 2020, pmets\_status, age\_group, year\_2020  
    **from** long\_term\_unemployed\_pmets\_by\_age\_wide   
    **UNION** **ALL**    
    **select** 2021, pmets\_status, age\_group, year\_2021  
    **from** long\_term\_unemployed\_pmets\_by\_age\_wide  
    **UNION** **ALL**   
    **select** 2022, pmets\_status, age\_group, year\_2022  
    **from** long\_term\_unemployed\_pmets\_by\_age\_wide   
    **UNION** **ALL**   
    **select** 2023, pmets\_status, age\_group, year\_2023  
    **from** long\_term\_unemployed\_pmets\_by\_age\_wide   
    **UNION** **ALL**   
    **select** 2024, pmets\_status, age\_group, year\_2024  
    **from** long\_term\_unemployed\_pmets\_by\_age\_wide ;

## Appendix 3 : SQL for Occupation Patterns Over Time

**SET** @first\_period\_start \= 2014;  
**SET** @first\_period\_end \= 2016;

**SET** @middle\_period\_start \= 2017;  
**SET** @middle\_period\_end \= 2019;

**SET** @supplementary\_period\_start \= 2020;  
**SET** @supplementary\_period\_end \= 2021;

**SET** @recent\_period\_start \= 2022;  
**SET** @recent\_period\_end \= 2024;

**WITH** first\_period **AS** (  
  **SELECT**   
    occupation,   
    AVG(unemployed\_rate) **AS** first\_period\_avg   
  **FROM**   
    unemployment\_rate\_by\_occupation\_long   
  **WHERE**   
    year **BETWEEN** @first\_period\_start   
    AND @first\_period\_end   
  **GROUP** **BY**   
    occupation  
),   
middle\_period **AS** (  
  **SELECT**   
    occupation,   
    AVG(unemployed\_rate) **AS** middle\_period\_avg   
  **FROM**   
    unemployment\_rate\_by\_occupation\_long   
  **WHERE**   
    year **BETWEEN** @middle\_period\_start   
    AND @middle\_period\_end   
  **GROUP** **BY**   
    occupation  
),   
supplementary\_period **AS** (  
  **SELECT**   
    occupation,   
    AVG(unemployed\_rate) **AS** supplementary\_period\_avg   
  **FROM**   
    unemployment\_rate\_by\_occupation\_long   
  **WHERE**   
    year **BETWEEN** @supplementary\_period\_start   
    AND @supplementary\_period\_end   
  **GROUP** **BY**   
    occupation  
),   
recent\_period **AS** (  
  **SELECT**   
    occupation,   
    AVG(unemployed\_rate) **AS** recent\_period\_avg   
  **FROM**   
    unemployment\_rate\_by\_occupation\_long   
  **WHERE**   
    year **BETWEEN** @recent\_period\_start   
    AND @recent\_period\_end   
  **GROUP** **BY**   
    occupation  
)   
**SELECT**   
  f.occupation,   
  ROUND(f.first\_period\_avg, 2) **AS** \`2014\-2016\`,   
  ROUND(m.middle\_period\_avg, 2) **AS** \`2017\-2019\`,   
  ROUND(s.supplementary\_period\_avg, 2) **AS** \`2020\-2021\`,   
  ROUND(r.recent\_period\_avg, 2) **AS** \`2022\-2024\`   
**FROM**   
  first\_period f   
  LEFT **JOIN** middle\_period m **ON** f.occupation \= m.occupation   
  LEFT **JOIN** supplementary\_period s **ON** f.occupation \= s.occupation   
  LEFT **JOIN** recent\_period r **ON** f.occupation \= r.occupation   
**ORDER** **BY**   
  f.occupation;

## Appendix 4 : SQL for Education Impact Over Time

**SET** @first\_period\_start \= 2014;  
**SET** @first\_period\_end \= 2016;

**SET** @middle\_period\_start \= 2017;  
**SET** @middle\_period\_end \= 2019;

**SET** @recent\_period\_start \= 2022;  
**SET** @recent\_period\_end \= 2024;

**SET** @supplementary\_period\_start \= 2020;  
**SET** @supplementary\_period\_end \= 2021;   
      
    **WITH** first\_period **AS** (  
    **SELECT**  
        education,  
        AVG(unemployed\_count) **AS** first\_period\_avg  
    **FROM** unemployed\_by\_qualification\_sex\_long  
    **WHERE** year **BETWEEN** @first\_period\_start AND @first\_period\_end  
    **GROUP** **BY** education  
),  
middle\_period **AS** (  
    **SELECT**  
        education,  
        AVG(unemployed\_count) **AS** middle\_period\_avg  
    **FROM** unemployed\_by\_qualification\_sex\_long  
    **WHERE** year **BETWEEN** @middle\_period\_start AND @middle\_period\_end  
    **GROUP** **BY** education  
),  
supplementary\_period **AS** (  
    **SELECT**  
        education,  
        AVG(unemployed\_count) **AS** supplementary\_period\_avg  
    **FROM** unemployed\_by\_qualification\_sex\_long  
    **WHERE** year **BETWEEN** @supplementary\_period\_start AND @supplementary\_period\_end  
    **GROUP** **BY** education  
),  
recent\_period **AS** (  
    **SELECT**  
        education,  
        AVG(unemployed\_count) **AS** recent\_period\_avg  
    **FROM** unemployed\_by\_qualification\_sex\_long  
    **WHERE** year **BETWEEN** @recent\_period\_start AND @recent\_period\_end  
    **GROUP** **BY** education  
)

**SELECT**  
    f.education,  
    ROUND(f.first\_period\_avg,2) **AS** \`**First** Period\`,  
    ROUND(m.middle\_period\_avg,2) **AS** \`Middle Period\`,  
    ROUND(s.supplementary\_period\_avg,2) **AS** \`Supplementary Period\`,  
    ROUND(r.recent\_period\_avg,2) **AS** \`Recent Period\`  
**FROM** first\_period f  
LEFT **JOIN** middle\_period m   
    **ON** f.education \= m.education  
LEFT **JOIN** supplementary\_period s   
    **ON** f.education \= s.education  
LEFT **JOIN** recent\_period r   
    **ON** f.education \= r.education  
**ORDER** **BY**   
    **CASE** f.education  
        **WHEN** 'Below Secondary' **THEN** 1  
        **WHEN** 'Secondary' **THEN** 2  
        **WHEN** 'Post-Secondary (Non-Tertiary)' **THEN** 3  
        **WHEN** 'Diploma & Professional Qualification' **THEN** 4  
        **WHEN** 'Degree' **THEN** 5  
    **END**;

## Appendix 5 : SQL for Combined Gender & Education Trends

**SET** @first\_period\_start \= 2014;  
**SET** @first\_period\_end \= 2016;

**SET** @middle\_period\_start \= 2017;  
**SET** @middle\_period\_end \= 2019;

**SET** @recent\_period\_start \= 2022;  
**SET** @recent\_period\_end \= 2024;

**SET** @supplementary\_period\_start \= 2020;  
**SET** @supplementary\_period\_end \= 2021; 

**WITH** GenderEducationUnemployment **AS** (  
    **SELECT**  
        year,  
        education,  
        SUM(**CASE** **WHEN** gender \= 'Male' **THEN** unemployed\_count **ELSE** 0 **END**) **AS** male\_unemployed,  
        SUM(**CASE** **WHEN** gender \= 'Female' **THEN** unemployed\_count **ELSE** 0 **END**) **AS** female\_unemployed  
    **FROM** unemployed\_by\_qualification\_sex\_long  
    **WHERE** year **BETWEEN** @first\_period\_start AND @recent\_period\_end  
       OR year **BETWEEN** @supplementary\_period\_start AND @supplementary\_period\_end  
    **GROUP** **BY** year, education  
)  
**SELECT**  
    education,  
    AVG(**CASE** **WHEN** year **BETWEEN** @first\_period\_start AND @first\_period\_end **THEN** ABS(male\_unemployed \- female\_unemployed) **ELSE** NULL **END**) **AS** avg\_gap\_first\_period,  
    AVG(**CASE** **WHEN** year **BETWEEN** @recent\_period\_start AND @recent\_period\_end **THEN** ABS(male\_unemployed \- female\_unemployed) **ELSE** NULL **END**) **AS** avg\_gap\_recent\_period,  
    AVG(**CASE** **WHEN** year **BETWEEN** @supplementary\_period\_start AND @supplementary\_period\_end **THEN** ABS(male\_unemployed \- female\_unemployed) **ELSE** NULL **END**) **AS** avg\_gap\_supplementary\_period,  
    (AVG(**CASE** **WHEN** year **BETWEEN** @first\_period\_start AND @first\_period\_end **THEN** ABS(male\_unemployed \- female\_unemployed) **ELSE** NULL **END**) \- AVG(**CASE** **WHEN** year **BETWEEN** @recent\_period\_start AND @recent\_period\_end **THEN** ABS(male\_unemployed \- female\_unemployed) **ELSE** NULL **END**)) **AS** gap\_reduction  
**FROM** GenderEducationUnemployment  
**GROUP** **BY** education  
**ORDER** **BY** gap\_reduction **DESC**;  
