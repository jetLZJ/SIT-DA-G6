/*Revision :
Creating Databases and Tables
Database creation is the foundation of any SQL project. 
The CREATE statements allow developers to establish the data structure and define relationships.

SQL Data Types
1. Numeric Types
These data types are used to store numeric values in a database.long_term_unemployed_pmets_by_age_long
	● INT (Integer):
	● DECIMAL (Precise numbers):
	● FLOAT (Approximate numbers):
2. String Types
These data types are used for storing text or alphanumeric data in a database.
	● VARCHAR (Variable-length string):
	● CHAR (Fixed-length string):
	● TEXT (Long strings):
3. Date/Time Types
	● DATE (Calendar date)
	● TIME (Time of day):
	● TIMESTAMP (Date and time):
4.	Other Types
	BOOLEAN (True/False):
	BLOB (Binary Large Object):
	JSON (Structured data):
  */

/*Upload and Pre-checks :
1. Before runing the below codes, save the upload files into the secure folder 
   C:/ProgramData/MySQL/MySQL Server 8.0/Uploads for MySQL to access and load data from. 

2. To check where is the secure folder, run the below code.
   SHOW VARIABLES LIKE 'secure_file_priv'; 

3. If the secure folder is not as per step 1 provided, then save the files in the secure folder in step 2 
   and update the folder path for all the commands
   LOAD DATA INFILE
*/

DROP DATABASE IF EXISTS labourtrendsdb;
CREATE DATABASE labourtrendsDB;
USE labourtrendsDB;

DROP TABLE IF EXISTS unemployed_by_age_sex_wide;
CREATE TABLE unemployed_by_age_sex_wide (
    gender VARCHAR(20),
    age_group VARCHAR(20),
    year_2014 DECIMAL(5,1),
    year_2015 DECIMAL(5,1),
    year_2016 DECIMAL(5,1),
    year_2017 DECIMAL(5,1),
    year_2018 DECIMAL(5,1),
    year_2019 DECIMAL(5,1),
    year_2020 DECIMAL(5,1),
    year_2021 DECIMAL(5,1),
    year_2022 DECIMAL(5,1),
    year_2023 DECIMAL(5,1),
    year_2024 DECIMAL(5,1)
);

DROP TABLE IF EXISTS unemployed_by_qualification_sex_wide;
CREATE TABLE unemployed_by_qualification_sex_wide (
    gender VARCHAR(20),
    education VARCHAR(50),
    year_2014 DECIMAL(5,1),
    year_2015 DECIMAL(5,1),
    year_2016 DECIMAL(5,1),
    year_2017 DECIMAL(5,1),
    year_2018 DECIMAL(5,1),
    year_2019 DECIMAL(5,1),
    year_2020 DECIMAL(5,1),
    year_2021 DECIMAL(5,1),
    year_2022 DECIMAL(5,1),
    year_2023 DECIMAL(5,1),
    year_2024 DECIMAL(5,1)
);

DROP TABLE IF EXISTS unemployed_by_marital_status_sex_wide;
CREATE TABLE unemployed_by_marital_status_sex_wide (
    gender VARCHAR(20),
    marital_status VARCHAR(20),
    year_2014 DECIMAL(5,1),
    year_2015 DECIMAL(5,1),
    year_2016 DECIMAL(5,1),
    year_2017 DECIMAL(5,1),
    year_2018 DECIMAL(5,1),
    year_2019 DECIMAL(5,1),
    year_2020 DECIMAL(5,1),
    year_2021 DECIMAL(5,1),
    year_2022 DECIMAL(5,1),
    year_2023 DECIMAL(5,1),
    year_2024 DECIMAL(5,1)
);


DROP TABLE IF EXISTS unemployment_rate_by_occupation_wide;
CREATE TABLE unemployment_rate_by_occupation_wide (
year INT,
Managers_N_Administrators_Including_Working_Proprietors DECIMAL(5,1),
Professionals DECIMAL(5,1),
Associate_Professionals_N_Technicians DECIMAL(5,1),
Clerical_Support_Workers DECIMAL(5,1),
Service_N_Sales_Workers DECIMAL(5,1),
Craftsmen_N_Related_Trades_Workers DECIMAL(5,1),
Plant_N_Machine_Operators_N_Assemblers DECIMAL(5,1),
Cleaners_Labourers_N_Related_Workers DECIMAL(5,1)
);


DROP TABLE IF EXISTS unemployed_by_previous_occupation_sex_wide;
CREATE TABLE unemployed_by_previous_occupation_sex_wide (
year INT,
gender VARCHAR(20),
Managers_N_Administrators_Including_Working_Proprietors DECIMAL(5,1),
Professionals DECIMAL(5,1),
Associate_Professionals_N_Technicians DECIMAL(5,1),
Clerical_Support_Workers DECIMAL(5,1),
Service_N_Sales_Workers DECIMAL(5,1),
Craftsmen_N_Related_Trades_Workers DECIMAL(5,1),
Plant_N_Machine_Operators_N_Assemblers DECIMAL(5,1),
Cleaners_Labourers_N_Related_Workers DECIMAL(5,1),
`Others` DECIMAL(5,1)
);

DROP TABLE IF EXISTS unemployed_pmets_by_age_wide;
CREATE TABLE unemployed_pmets_by_age_wide (
    pmets_status VARCHAR(20),
    age_group VARCHAR(20),
    year_2014 DECIMAL(5,1),
    year_2015 DECIMAL(5,1),
    year_2016 DECIMAL(5,1),
    year_2017 DECIMAL(5,1),
    year_2018 DECIMAL(5,1),
    year_2019 DECIMAL(5,1),
    year_2020 DECIMAL(5,1),
    year_2021 DECIMAL(5,1),
    year_2022 DECIMAL(5,1),
    year_2023 DECIMAL(5,1),
    year_2024 DECIMAL(5,1)
);

DROP TABLE IF EXISTS long_term_unemployed_pmets_by_age_wide;
CREATE TABLE long_term_unemployed_pmets_by_age_wide (
    pmets_status VARCHAR(20),
    age_group VARCHAR(20),
    year_2014 DECIMAL(5,1),
    year_2015 DECIMAL(5,1),
    year_2016 DECIMAL(5,1),
    year_2017 DECIMAL(5,1),
    year_2018 DECIMAL(5,1),
    year_2019 DECIMAL(5,1),
    year_2020 DECIMAL(5,1),
    year_2021 DECIMAL(5,1),
    year_2022 DECIMAL(5,1),
    year_2023 DECIMAL(5,1),
    year_2024 DECIMAL(5,1)
);

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/k2unemployed_by_age_sex_wide.csv'
INTO TABLE unemployed_by_age_sex_wide
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS;

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/k3unemployed_by_qualification_sex_wide.csv'
INTO TABLE unemployed_by_qualification_sex_wide
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS;

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/k6unemployed_by_marital_status_sex_wide.csv'
INTO TABLE unemployed_by_marital_status_sex_wide
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS;

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/k14unemployment_rate_by_occupation_wide.csv'
INTO TABLE unemployment_rate_by_occupation_wide
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS;

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/k15unemployed_by_previous_occupation_sex_wide.csv'
INTO TABLE unemployed_by_previous_occupation_sex_wide
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS;

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/k16unemployed_pmets_by_age_wide.csv'
INTO TABLE unemployed_pmets_by_age_wide
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS;

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/k17long_term_unemployed_pmets_by_age_wide.csv'
INTO TABLE long_term_unemployed_pmets_by_age_wide
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS;