-- Appendix 2: Transform wide -> long (example unpivot patterns)
USE labourtrendsDB;

-- Transform unemployed_by_age_sex_wide -> unemployed_by_age_sex_long
DROP TABLE IF EXISTS unemployed_by_age_sex_long;
CREATE TABLE unemployed_by_age_sex_long AS
SELECT 2014 AS year, gender, age_group, year_2014 AS unemployed_count FROM unemployed_by_age_sex_wide
UNION ALL
SELECT 2015, gender, age_group, year_2015 FROM unemployed_by_age_sex_wide
UNION ALL
SELECT 2016, gender, age_group, year_2016 FROM unemployed_by_age_sex_wide
UNION ALL
SELECT 2017, gender, age_group, year_2017 FROM unemployed_by_age_sex_wide
UNION ALL
SELECT 2018, gender, age_group, year_2018 FROM unemployed_by_age_sex_wide
UNION ALL
SELECT 2019, gender, age_group, year_2019 FROM unemployed_by_age_sex_wide
UNION ALL
SELECT 2020, gender, age_group, year_2020 FROM unemployed_by_age_sex_wide
UNION ALL
SELECT 2021, gender, age_group, year_2021 FROM unemployed_by_age_sex_wide
UNION ALL
SELECT 2022, gender, age_group, year_2022 FROM unemployed_by_age_sex_wide
UNION ALL
SELECT 2023, gender, age_group, year_2023 FROM unemployed_by_age_sex_wide
UNION ALL
SELECT 2024, gender, age_group, year_2024 FROM unemployed_by_age_sex_wide;

-- Transform pattern for occupation rate wide -> long (example)
DROP TABLE IF EXISTS unemployment_rate_by_occupation_long;
CREATE TABLE unemployment_rate_by_occupation_long AS
SELECT year, 'Managers & Administrators (Including Working Proprietors)' AS occupation, Managers_N_Administrators_Including_Working_Proprietors AS unemployed_rate FROM unemployment_rate_by_occupation_wide
UNION ALL
SELECT year, 'Professionals', Professionals FROM unemployment_rate_by_occupation_wide
UNION ALL
SELECT year, 'Associate Professionals & Technicians', Associate_Professionals_N_Technicians FROM unemployment_rate_by_occupation_wide
UNION ALL
SELECT year, 'Clerical Support Workers', Clerical_Support_Workers FROM unemployment_rate_by_occupation_wide
UNION ALL
SELECT year, 'Service & Sales Workers', Service_N_Sales_Workers FROM unemployment_rate_by_occupation_wide
UNION ALL
SELECT year, 'Craftsmen & Related Trades Workers', Craftsmen_N_Related_Trades_Workers FROM unemployment_rate_by_occupation_wide
UNION ALL
SELECT year, 'Plant & Machine Operators & Assemblers', Plant_N_Machine_Operators_N_Assemblers FROM unemployment_rate_by_occupation_wide
UNION ALL
SELECT year, 'Cleaners, Labourers & Related Workers', Cleaners_Labourers_N_Related_Workers FROM unemployment_rate_by_occupation_wide;

-- Repeat the pattern for other wide tables as required
