# breakfast-DASH
## Introduction
Trying out the Dash for Python to create dashboards. Utilizing a small data set from Reid's, a fictional "small breakfast stand that sells drinks (coffee, tea, sodas) and food (egg & sausage, oatmeal) in a commercial downtown area, Monday through Friday, from 8a until 11am. Although their menu is small, they do try to cater to a wide variety of diets and thus provide both vegan and keto options for most of their meals. They started using Ordr as their Point of Sale (POS) system about two months ago and are on the Basic Plan." Under the basic Ordr Plan, Reid's is able to use the Ordr API to access orders. "This order information comes in the form of a denormalized JSON document"

The data set was provided to me during a Johns Hopkins' data science course as part of a lab assignment focusing on ETL. The goal of the lab assigment was to was read the JSON data, extract the values, and create a database using SQL DDL and python code, and insert the values into the database. SQL queries can then be performed on the data. Due to time constraints, EDA was more informally done simply by querying the data to answer a few questions.

## Repo Contents
The repo contains the following files:
1. reids.json : provided JSON data.
2. reids.sql : DDL code to create db. See reids.py for creating database.
3. reids.db : Database/datawarehouse that contains 
4. reids.py : Code to parse JSON and insert valeus into database
5. reids.ipynb : Jupyter notebook to quickly test out dash and visualize charts
6. website.py : Potential file to create local server to use for hosting dashboard on website.
