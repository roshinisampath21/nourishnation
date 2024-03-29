from dash import Dash, html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import pickle
from io import StringIO
import requests
from collections import Counter

# Initialize the Dash application
app = Dash(__name__, title="Crop Recommendation System")

# Function to fetch CSV data from URL
def fetch_csv_data(url):
    response = requests.get(url)
    return pd.read_csv(StringIO(response.text))

# Function to predict crops from saved model and user inputs
def predict_crops_from_saved_model(year, month, model, label_encoder, crop_to_human_category):
    # ... your code as provided for fetching data and predicting ...
     webhdfs_url = "http://127.0.0.1:9870/webhdfs/v1/user/nourishnation/vap_future.csv?op=OPEN"
     response_vap = requests.get(webhdfs_url)
     vap_data = pd.read_csv(StringIO(response_vap.text))

     webhdfs_url = "http://127.0.0.1:9870/webhdfs/v1/user/nourishnation/pre_future.csv?op=OPEN"
     response_pre = requests.get(webhdfs_url)
     pre_data = pd.read_csv(StringIO(response_pre.text))

     webhdfs_url = "http://127.0.0.1:9870/webhdfs/v1/user/nourishnation/temp_future.csv?op=OPEN"
     response_temp = requests.get(webhdfs_url)
     temp_data = pd.read_csv(StringIO(response_temp.text))

     # Filter data for the selected year
     vap_data_year = vap_data[vap_data['YEAR'] == year]
     pre_data_year = pre_data[pre_data['YEAR'] == year]
     temp_data_year = temp_data[temp_data['YEAR'] == year]

     # Extract weather data for the specified year and month
     selected_month_data = {
        'temperature': temp_data_year[month].values,
        'humidity': vap_data_year[month].values,
        'ph': np.round(np.random.uniform(5, 7, size=len(temp_data_year)), 6),
        'rainfall': pre_data_year[month].values
    }

    # Prepare the data for prediction
     selected_weather_data = pd.DataFrame(selected_month_data)

    # Make predictions using the loaded model
     predictions = model.predict(selected_weather_data)
     predicted_crops = label_encoder.inverse_transform(predictions)

    # Map each predicted crop to its human category
     predicted_crops_with_category = [(crop, crop_to_human_category.get(crop, 'Unknown category')) for crop in predicted_crops]

     return predicted_crops_with_category

# Load the model and label encoder from disk once during setup
with open('/Users/roshinisampath/Crop-Recommendation/data/crop_prediction_model.pkl', 'rb') as file:  # Replace with actual path
        model, label_encoder = pickle.load(file)

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Crop Recommendation System"),
    dcc.Input(id='input-year', type='number', placeholder="Enter Year", debounce=True),
    dcc.Dropdown(
        id='input-month',
        options=[{'label': m, 'value': m} for m in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']],
        placeholder="Select Month"
    ),
    html.Button('Predict', id='predict-button', n_clicks=0),
    html.Div(id='prediction-result')
])

@app.callback(
    Output('prediction-result', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('input-year', 'value'), State('input-month', 'value')]
)
def update_output(n_clicks, year, month):
    if n_clicks > 0 and year and month:
        crop_human_url = "http://127.0.0.1:9870/webhdfs/v1/user/nourishnation/crop_human.csv?op=OPEN"
        crop_data = fetch_csv_data(crop_human_url)
        crop_to_human_category = dict(zip(crop_data['label'], crop_data['human_category']))
        
        predicted_crops_with_category = predict_crops_from_saved_model(year, month, model, label_encoder, crop_to_human_category)
        crop_counts = Counter([crop for crop, category in predicted_crops_with_category])
        priority_list = [crop for crop, count in crop_counts.most_common()]
        
        all_possible_crops = set(crop_to_human_category.keys())
        non_predicted_crops = all_possible_crops - set(priority_list)
        complete_priority_list = priority_list + list(non_predicted_crops)
        
        # Format the output for display
        output = [html.H4(f"Complete list of crops for {year}-{month} ranked by suitability (with human categories):")]
        output.extend([html.P(f"{crop} (Human category: {crop_to_human_category.get(crop, 'Unknown category')})") for crop in complete_priority_list])
        return output
    
    return "Enter year and month, then click predict."

if __name__ == '__main__':
    app.run_server(debug=True)
