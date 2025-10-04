-- Appendix 1: Create database and staging tables (example)
-- DROP / CREATE database
DROP DATABASE IF EXISTS labourtrendsdb;
CREATE DATABASE labourtrendsDB;
USE labourtrendsDB;

-- Example: unemployed_by_age_sex_wide (wide-format staging table)
DROP TABLE IF EXISTS unemployed_by_age_sex_wide;
CREATE TABLE unemployed_by_age_sex_wide (
    gender VARCHAR(20),
    age_group VARCHAR(50),
    year_2014 INT,
    year_2015 INT,
    year_2016 INT,
    year_2017 INT,
    year_2018 INT,
    year_2019 INT,
    year_2020 INT,
    year_2021 INT,
    year_2022 INT,
    year_2023 INT,
    year_2024 INT
);

-- Example: unemployment_rate_by_occupation_wide
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

-- Example load commands (adjust paths per environment)
-- LOAD DATA INFILE 'C:/path/to/k2unemployed_by_age_sex_wide.csv' 
-- INTO TABLE unemployed_by_age_sex_wide
-- FIELDS TERMINATED BY ','
-- LINES TERMINATED BY '\r\n'
-- IGNORE 1 ROWS;

-- Repeat LOAD DATA INFILE for other wide CSVs as needed
