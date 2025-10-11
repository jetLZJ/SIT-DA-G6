/*Revision : 
Data Transformation
	Data Transformation Techniques
	Transformation(Pivoting & Unpivoting)
		● What It Is:
		Pivoting involves rearranging or summarizing data to make it easier to analyze. Unpivoting is the reverse process,
		where data is converted from a wide format to a long format for easier manipulation and analysis.
		● Why It Matters:
		Pivoting and unpivoting can help restructure data to meet the needs of different analytical tools or reporting formats,
		especially when dealing with large and complex datasets.

	To best utilize a relational database, it's recommended to convert data from the "wide" format to a "long" format. 
    This means each row will represent a single data point.
 
SQL Syntax Order
  SELECT 
  FROM & JOINs - determine & filter rows
  WHERE - more filters on the rows (The WHERE clause is used to filter coloumns in FROM statement)
  GROUP BY - combines those rows into groups
  HAVING - filters groups (The HAVING clause is used to filter aggregate functions & coloumns of the SELECT statement.)
  ORDER BY - arranges the remaining rows/groups
  LIMIT - filters on the remaining rows/groups
 */
SET SESSION sql_require_primary_key = 0;
USE labourtrendsDB;

-- Transform unemployed_by_age_sex_wide
DROP TABLE IF EXISTS unemployed_by_age_sex_long;
CREATE TABLE unemployed_by_age_sex_long AS
	select 2014 AS year, gender, age_group, year_2014 As unemployed_count
	from unemployed_by_age_sex_wide
	UNION ALL 
	select 2015, gender, age_group, year_2015
	from unemployed_by_age_sex_wide 
	UNION ALL  
	select 2016, gender, age_group, year_2016 
	from unemployed_by_age_sex_wide 
	UNION ALL  
	select 2017, gender, age_group, year_2017
	from unemployed_by_age_sex_wide 
	UNION ALL  
	select 2018, gender, age_group, year_2018
	from unemployed_by_age_sex_wide 
	UNION ALL  
	select 2019, gender, age_group, year_2019
	from unemployed_by_age_sex_wide 
	UNION ALL  
	select 2020, gender, age_group, year_2020
	from unemployed_by_age_sex_wide 
	UNION ALL  
	select 2021, gender, age_group, year_2021
	from unemployed_by_age_sex_wide 
	UNION ALL 
	select 2022, gender, age_group, year_2022
	from unemployed_by_age_sex_wide 
	UNION ALL 
	select 2023, gender, age_group, year_2023
	from unemployed_by_age_sex_wide 
	UNION ALL 
	select 2024, gender, age_group, year_2024
	from unemployed_by_age_sex_wide;

-- Transform unemployed_by_qualification_sex_wide
DROP TABLE IF EXISTS unemployed_by_qualification_sex_long;
CREATE TABLE unemployed_by_qualification_sex_long AS
	select 2014 AS year, gender, education, year_2014 As unemployed_count
	from unemployed_by_qualification_sex_wide
	UNION ALL 
	select 2015, gender, education, year_2015
	from unemployed_by_qualification_sex_wide 
	UNION ALL  
	select 2016, gender, education, year_2016 
	from unemployed_by_qualification_sex_wide 
	UNION ALL  
	select 2017, gender, education, year_2017
	from unemployed_by_qualification_sex_wide 
	UNION ALL  
	select 2018, gender, education, year_2018
	from unemployed_by_qualification_sex_wide 
	UNION ALL  
	select 2019, gender, education, year_2019
	from unemployed_by_qualification_sex_wide 
	UNION ALL  
	select 2020, gender, education, year_2020
	from unemployed_by_qualification_sex_wide 
	UNION ALL  
	select 2021, gender, education, year_2021
	from unemployed_by_qualification_sex_wide 
	UNION ALL 
	select 2022, gender, education, year_2022
	from unemployed_by_qualification_sex_wide 
	UNION ALL 
	select 2023, gender, education, year_2023
	from unemployed_by_qualification_sex_wide 
	UNION ALL 
	select 2024, gender, education, year_2024
	from unemployed_by_qualification_sex_wide;

-- Transform unemployed_by_marital_status_sex_wide
DROP TABLE IF EXISTS unemployed_by_marital_status_sex_long;
	CREATE TABLE unemployed_by_marital_status_sex_long AS
	select 2014 AS year, gender, marital_status, year_2014 As unemployed_count
	from unemployed_by_marital_status_sex_wide
	UNION ALL 
	select 2015, gender, marital_status, year_2015
	from unemployed_by_marital_status_sex_wide 
	UNION ALL  
	select 2016, gender, marital_status, year_2016 
	from unemployed_by_marital_status_sex_wide 
	UNION ALL  
	select 2017, gender, marital_status, year_2017
	from unemployed_by_marital_status_sex_wide 
	UNION ALL  
	select 2018, gender, marital_status, year_2018
	from unemployed_by_marital_status_sex_wide 
	UNION ALL  
	select 2019, gender, marital_status, year_2019
	from unemployed_by_marital_status_sex_wide 
	UNION ALL  
	select 2020, gender, marital_status, year_2020
	from unemployed_by_marital_status_sex_wide 
	UNION ALL  
	select 2021, gender, marital_status, year_2021
	from unemployed_by_marital_status_sex_wide 
	UNION ALL 
	select 2022, gender, marital_status, year_2022
	from unemployed_by_marital_status_sex_wide 
	UNION ALL 
	select 2023, gender, marital_status, year_2023
	from unemployed_by_marital_status_sex_wide 
	UNION ALL 
	select 2024, gender, marital_status, year_2024
	from unemployed_by_marital_status_sex_wide;

-- Transform unemployment_rate_by_occupation_wide
DROP TABLE IF EXISTS unemployment_rate_by_occupation_long;
CREATE TABLE unemployment_rate_by_occupation_long AS
	SELECT year, "Managers_N_Administrators_Including_Working_Proprietors" As occupation, Managers_N_Administrators_Including_Working_Proprietors As unemployed_rate
	FROM unemployment_rate_by_occupation_wide
	UNION ALL 
	SELECT year, "Professionals" As occupation, Professionals
	FROM unemployment_rate_by_occupation_wide
	UNION ALL 
	SELECT year, "Associate_Professionals_N_Technicians" As occupation, Associate_Professionals_N_Technicians
	FROM unemployment_rate_by_occupation_wide
	UNION ALL 
	SELECT year, "Clerical_Support_Workers" As occupation, Clerical_Support_Workers
	FROM unemployment_rate_by_occupation_wide
	UNION ALL 
	SELECT year, "Service_N_Sales_Workers" As occupation, Service_N_Sales_Workers
	FROM unemployment_rate_by_occupation_wide
	UNION ALL 
	SELECT year, "Craftsmen_N_Related_Trades_Workers" As occupation, Craftsmen_N_Related_Trades_Workers
	FROM unemployment_rate_by_occupation_wide
	UNION ALL 
	SELECT year, "Plant_N_Machine_Operators_N_Assemblers" As occupation, Plant_N_Machine_Operators_N_Assemblers
	FROM unemployment_rate_by_occupation_wide
	UNION ALL 
	SELECT year, "Cleaners_Labourers_N_Related_Workers" As occupation, Cleaners_Labourers_N_Related_Workers
	FROM unemployment_rate_by_occupation_wide;
    
-- Transform unemployed_by_previous_occupation_sex_wide    
DROP TABLE IF EXISTS unemployed_by_previous_occupation_sex_long;
CREATE TABLE unemployed_by_previous_occupation_sex_long AS
	SELECT year, gender, "Managers_N_Administrators_Including_Working_Proprietors" As occupation, Managers_N_Administrators_Including_Working_Proprietors As unemployed_count
	FROM unemployed_by_previous_occupation_sex_wide
	UNION ALL 
	SELECT year, gender, "Professionals" As occupation, Professionals
	FROM unemployed_by_previous_occupation_sex_wide
	UNION ALL 
	SELECT year, gender, "Associate_Professionals_N_Technicians" As occupation, Associate_Professionals_N_Technicians
	FROM unemployed_by_previous_occupation_sex_wide
	UNION ALL 
	SELECT year, gender, "Clerical_Support_Workers" As occupation, Clerical_Support_Workers
	FROM unemployed_by_previous_occupation_sex_wide
	UNION ALL 
	SELECT year, gender, "Service_N_Sales_Workers" As occupation, Service_N_Sales_Workers
	FROM unemployed_by_previous_occupation_sex_wide
	UNION ALL 
	SELECT year, gender, "Craftsmen_N_Related_Trades_Workers" As occupation, Craftsmen_N_Related_Trades_Workers
	FROM unemployed_by_previous_occupation_sex_wide
	UNION ALL 
	SELECT year, gender, "Plant_N_Machine_Operators_N_Assemblers" As occupation, Plant_N_Machine_Operators_N_Assemblers
	FROM unemployed_by_previous_occupation_sex_wide
	UNION ALL 
	SELECT year, gender, "Cleaners_Labourers_N_Related_Workers" As occupation, Cleaners_Labourers_N_Related_Workers
	FROM unemployed_by_previous_occupation_sex_wide
    UNION ALL 
	SELECT year, gender, "Others" As occupation, Others
	FROM unemployed_by_previous_occupation_sex_wide;
    
-- Transform unemployed_pmets_by_age_wide   
DROP TABLE IF EXISTS unemployed_pmets_by_age_long;
CREATE TABLE unemployed_pmets_by_age_long AS
	select 2014 AS year, pmets_status, age_group, year_2014 As unemployed_count
	from unemployed_pmets_by_age_wide
	UNION ALL 
	select 2015, pmets_status, age_group, year_2015
	from unemployed_pmets_by_age_wide 
	UNION ALL  
	select 2016, pmets_status, age_group, year_2016 
	from unemployed_pmets_by_age_wide
	UNION ALL  
	select 2017, pmets_status, age_group, year_2017
	from unemployed_pmets_by_age_wide 
	UNION ALL  
	select 2018, pmets_status, age_group, year_2018
	from unemployed_pmets_by_age_wide 
	UNION ALL  
	select 2019, pmets_status, age_group, year_2019
	from unemployed_pmets_by_age_wide 
	UNION ALL  
	select 2020, pmets_status, age_group, year_2020
	from unemployed_pmets_by_age_wide 
	UNION ALL  
	select 2021, pmets_status, age_group, year_2021
	from unemployed_pmets_by_age_wide
	UNION ALL 
	select 2022, pmets_status, age_group, year_2022
	from unemployed_pmets_by_age_wide 
	UNION ALL 
	select 2023, pmets_status, age_group, year_2023
	from unemployed_pmets_by_age_wide 
	UNION ALL 
	select 2024, pmets_status, age_group, year_2024
	from unemployed_pmets_by_age_wide;

-- Transform long_term_unemployed_pmets_by_age_wide    
DROP TABLE IF EXISTS long_term_unemployed_pmets_by_age_long;
CREATE TABLE long_term_unemployed_pmets_by_age_long AS
	select 2014 AS year, pmets_status, age_group, year_2014 As unemployed_count
	from long_term_unemployed_pmets_by_age_wide
	UNION ALL 
	select 2015, pmets_status, age_group, year_2015
	from long_term_unemployed_pmets_by_age_wide 
	UNION ALL  
	select 2016, pmets_status, age_group, year_2016 
	from long_term_unemployed_pmets_by_age_wide
	UNION ALL  
	select 2017, pmets_status, age_group, year_2017
	from long_term_unemployed_pmets_by_age_wide 
	UNION ALL  
	select 2018, pmets_status, age_group, year_2018
	from long_term_unemployed_pmets_by_age_wide 
	UNION ALL  
	select 2019, pmets_status, age_group, year_2019
	from long_term_unemployed_pmets_by_age_wide 
	UNION ALL  
	select 2020, pmets_status, age_group, year_2020
	from long_term_unemployed_pmets_by_age_wide 
	UNION ALL  
	select 2021, pmets_status, age_group, year_2021
	from long_term_unemployed_pmets_by_age_wide
	UNION ALL 
	select 2022, pmets_status, age_group, year_2022
	from long_term_unemployed_pmets_by_age_wide 
	UNION ALL 
	select 2023, pmets_status, age_group, year_2023
	from long_term_unemployed_pmets_by_age_wide 
	UNION ALL 
	select 2024, pmets_status, age_group, year_2024
	from long_term_unemployed_pmets_by_age_wide ;