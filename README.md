# Project Title

This project addresses the current crisis of climate change's impact on agricultural productivity by introducing an innovative approach that integrates Hadoop's HDFS with machine learning algorithms. Our approach focuses on pre-harvest crop production forecasting, delivering precise projections that are vital to efficient management strategies in the midst of climate variability. The project is aligned with the UN's 2030 Sustainable Development Goal of eradicating hunger, using extensive weather analysis to produce nutrient-rich crops that can withstand changing climatic circumstances, thereby optimizing crop selections and fostering sustainable farming practices.

## Dataset

The datasets are stored under the `dataset` folder. 
1. crop_data.csv
Columns: Includes information on food ID, name, various nutrients (name, amount, symbol), unit, labels, and agricultural data like N, P, K values, temperature, humidity, pH, and rainfall.
Rows: 327,400
Preview: Lists nutrients for crops like Coffee, including protein content and various environmental requirements for growth.

2. crop_human.csv
Columns: Agricultural data (N, P, K, temperature, humidity, pH, rainfall), labels for crops, and human categories they are suited for.
Rows: 2,200
Preview: Contains agronomic data and suitability categories for humans, indicating which crops are recommended for specific human nutritional needs.

3. pre_future.csv
Columns: Yearly and monthly precipitation forecasts (YEAR, monthly values JAN-DEC, and seasonal summaries MAM, JJA, SON, DJF, ANN).
Rows: 132
Preview: Provides historical and future precipitation data, including yearly averages and monthly distributions.

4. temp_future.csv
Columns: Yearly and monthly temperature forecasts similar to pre_future.csv.
Rows: 132
Preview: Contains temperature forecasts, detailing monthly and annual averages, which are crucial for agricultural planning.

5. vap_future.csv
Columns: Yearly and monthly vapor pressure data structured similarly to the previous future forecast datasets.
Rows: 132
Preview: Offers vapor pressure forecasts, providing additional climatic parameters for agricultural analysis.

## Machine Learning Algorithms

The models and machine learning algorithms are located in the `machinelearningalgo` folder. Describe the algorithms you've implemented, their purpose, and any libraries or frameworks used.

## Dash UI

The Dash application code for the UI is found under `dashui`. To display the UI, run the `app.py` file. Note: Change the path of the model to the required path as per your local or server setup.

### Running the Dash Application

Provide step-by-step instructions to run the Dash application. 
python app.py

##Setup HDFS
Setup the dataset (.csv) files in Hadoop
