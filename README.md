# Breakfast-DASH Dashboard

Welcome to the Breakfast-DASH(board)! This interactive Dash application provides insights into a fictional breakfast stand named Reid's, which operates in a commercial downtown area from Monday to Friday, 8 am to 11 am. The stand offers a selection of drinks (coffee, tea, sodas) and food items (egg & sausage, oatmeal), catering to various dietary preferences including vegan and keto options. Reid's uses the Ordr Point of Sale (POS) system, utilizing the Ordr API to access order data. The body of the Order API message is available in a JSON document.

## Access the Dashboard
Explore the Breakfast-DASH dashboard through the following link: [Breakfast-DASH Dashboard](http://b-dash-iframe-env.eba-trun3m23.us-east-2.elasticbeanstalk.com)

## Description
This project was undertaken to explore the capabilities of Dash, a Python framework for building interactive web applications. The project began with creating a minimal Dash app within a Jupyter notebook, with the goal of gaining familiarity with Dash's features and functionalities. Modifying sample code provided a better understanding of Dash's potential.

The next phase involved designing a custom dashboard. Leveraging a small dataset from Reid's, containing information about their sales of food and drinks, the project aimed to visualize insights effectively. The data was originally obtained during a Johns Hopkins data science course as part of an ETL-focused lab assignment. The lab involved extracting relevant values from a denormalized JSON document, constructing a SQL database using Python and SQL DDL, and performing basic EDA through SQL queries.

### Dashboard Features
The Breakfast-DASH(board) has the following features:
1. **Food and Drink Sales Representation:**
   - A table and bar chart showcasing the proportion of food and drink items sold.
   - The bar chart offers interactive checkboxes to toggle the display of food and drink items, enhancing user engagement.
2. **Daily Gross Revenue Trend:**
   - A line chart illustrating the daily gross revenue trend.
   
The resulting Dash application was deployed using AWS Beanstalk, ensuring accessibility for users.

## Challenges and Lessons Learned
The Breakfast-DASH project presented several challenges and valuable lessons:
1. **Mix of Technologies:** Using Python/Pandas to simply some of the HTML, CSS, and JavaScript was interesting to see and my familiarity with Pandas made the code easy to understand and change.
2. **Inline CSS Considerations:** The use of inline CSS to style elements introduced complexity and readability challenges in the codebase.
3. **Interactivity Complexity:** Increasing the interactivity of the dashboard directly impacted its complexity.
4. **AWS Deployment Hurdles:** Navigating AWS Elastic Beanstalk for application hosting posed challenges, especially regarding configuring the EC2 Role Policies.

The Breakfast-DASH project provided a foundational understanding of interactive dashboards, resulting in a user-friendly and informative dashboard for Reid's breakfast stand operations, that can also be improved in the future.

## Required Packages Libraries 
virtualenv
pandas
dash
See virtual environment libraries and requirements.txt

## Python version
Python 3.10

## Repository Contents
1. **reids.json**: This file contains the provided JSON data, serving as the foundation for the project's dataset.
2. **reids.sql**: The SQL Data Definition Language (DDL) code is present here, facilitating the creation of the database structure. Refer to the instructions below and **reids.py** for the code to create and populate the database.
3. **reids.db**: This file contains the database, storing the organized data for analysis and visualization.
4. **reids.py**: Inside this file, you'll find the code responsible for parsing the JSON data and inserting the corresponding values into the database.
5. **reids.ipynb**: This Jupyter notebook serves as an experimentation ground for testing out Dash functionalities and rapidly visualizing charts. The notebook also details effort in customizing dashboards using HTML and CSS modifications.
6. **application.py**: The main code in creating the dashboard, that will become a web application deployed via AWS Elastic Beanstalk.
7. **BreakfastDashEnv folder**: This directory contains files pertinent to the virtual environment setup. It isolates packages and dependencies, ensuring a controlled environment for the project's needs.
8. **requirements.txt**: Here, you'll discover a list of packages and dependencies compiled for deployment of the application using AWS Elastic Beanstalk.
9. **.ebextensions folder**: This directory houses configuration files tailored for AWS Elastic Beanstalk. While not mandatory, these files may be needed for future efforts.
10. **myfile.zip**: Enclosed within this zip file are **application.py**, **requirements.txt**, **reids.db**, and the contents of **.ebextensions**. This zip file is upload to AWS Elastic Beanstalk during environment configuration.

## Setup and Instructions (This setup can be automated/expedited in future releases)
### Database
1. Open terminal window and type the following and press Enter after typing each command.
2. 'python3.10 reids.py' to create and populate the database.
3. 'ls | grep db' to confirm creation of reids.db. Alternatively use a file explorer to view that reids.db was created.
4. View or query the database to ensure it is populated with data. 
### Virtual Environment (venv)
1. Open terminal window and type the following and press Enter after typing each command.
1. 'python3.10 -m pip install --user virtualenv' to install virtualenv
2. 'python3.10 -m venv BreakfastDashEnv' to create BreakfastDashEnv venv
3. 'source BreakfastDashEnv/bin/activate' to this command activates venv
4. Install libraries using instructions from the dash documentation https://dash.plotly.com/installation
5. 'jupyter-lab' to open notebooks or do any development work in this venv
6. 'deactivate' to leave virtual environment
### Run local server of dash app
1. Open terminal window and type the following and press Enter after typing each command.
2. 'python3.10 application.py'
3. Open a web browser and go to the local host address on part 8080: 'http://127.0.0.1:8080'
### Configuring files for AWS Elastic Beanstalk
1. Enter the BreakfastDashEnv venv via a terminal session. Press enter as needed after each terminal window command.
2. Ensure the python dash file is titled 'application.py'
3. Ensure the following text is added to 'application.py':
  app = Dash(__name__)
  app.scripts.config.serve_locally = True
  application = app.server
... and the application is ran with the following line. The important part of this line are 'application' and 'port=8080':
  application.run(debug=False, port=8080)
4. 'pip freeze > requirements.txt' to create a text file with the programs and libraries required for the venv
5. 'zip myfiles.zip requirements.txt application.py reids.db .ebextensions' to create a zip file of the files that will be uploaded via AWS Elastic Beanstalk
### Deploy project via AWS Elastic Bean Stalk
1. Log into AWS console, go to Elastic Beanstalk, and Select 'Create Application'
2. Type an Application Name
3. Select Create
4. Select Create new environment
5. Ensure Web server environment is selected under Environment tier
6. Ensure Managed platform is selected under Platform type
7. Select Python under Platform
8. Ensure Platform branch is Python3.11 running on 64bit Amazon Linux 2023
9. Ensure Platform version is 4.0.3 (recommended)
10. Select Upload your code under Application code
11. Enter any valid values for version label
12. Select Local file (can also first upload myfile.zip to s3 and use the Public S3 URL)
13. Select Choose File
14. Navigate to the project directory in the file explorer that pops up
15. Select myfile.zip and follow any prompts to upload the file.
16. Select next
17. Select Use an existing service role under Service role
18. Select aws-elasticbeanstalk-service-role under Existing service roles
19. Select aws-elasticbeanstalk-ec2-role under EC2 Instance profile
    If this role does not exist, utilize these instructions to create the role and attach the following policies in AWS IAM. You may or may not need to restart this process.
    https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/iam-instanceprofile.html
    AWSElasticBeanstalkWebTier
    AWSElasticBeanstalkWorkerTier
    AWSElasticBeanstalkMulticontainerDocker
20. Select next
21. Select Skip to review or next throug the remaining pages
22. Review configurations
23. Select Submit
24. Wait for AWS to to fully launch the environment. The Health Status should be green and say 'OK'
25. Select Go to Environemnt to access the dash app.

## Using the Web App
Whether on a local server or hosted on AWS, here are some tips for using the dashboard:

1. Hover over bar charts of points on the line charts highlight and obtain the values of specific data points.
2. Click, drag, and release your mouse cursor to zoom in on charts.
3. Similarly, double click anywhere on a chart to zoom out.
4. Toggle the checboxes to remove and insert food and drink items from the bar chart. Future iterations will make this bar chart even more flexible!
