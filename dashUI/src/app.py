import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import plotly.graph_objs as go  # Import Plotly graph objects
import requests
from io import StringIO

# Initialize the Dash application with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Crop Recommendation System")


# Function placeholders for fetching data and predicting
def fetch_csv_data(filename):
    return pd.read_csv(filename)

def predict_crops_from_saved_model(year, month, model, label_encoder, crop_to_human_category):
    # ... your code as provided for fetching data and predicting ...
    # Example code:
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

# Load the saved model and label encoder
with open('/Users/roshinisampath/Crop-Recommendation/data/crop_prediction_model.pkl', 'rb') as file:
    model, label_encoder = pickle.load(file)

# Layout of the Dash app with inputs one below the other and semi-transparent table
app.layout = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(html.H1("Crop Recommendation System", className="text-center mb-4", style={'color': '#228B22'}), width=12),
                ),
                dbc.Row(
                    dbc.Col(dcc.Input(id='input-year', type='number', placeholder="Enter Year", className="mb-2 form-control"), width={"size": 4, "offset": 4}),
                ),
                dbc.Row(
                    dbc.Col(dcc.Dropdown(
                        id='input-month',
                        options=[{'label': m, 'value': m} for m in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']],
                        placeholder="Select Month",
                    ), width={"size": 4, "offset": 4}),
                ),
                dbc.Row(
                    dbc.Col(html.Button('Predict', id='predict-button', n_clicks=0, className="btn btn-success mt-2"), width={"size": 4, "offset": 4}),
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div(id='prediction-result'),width = 6),
                        dbc.Col(html.Div(id='prediction-result-2'), width=6)
                    ]
                ),
                dbc.Row(
                    dbc.Col(html.Div(id='temperature-graph'), width=12),  # Added a div for the graph
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div(id='precipitation-graph'),width = 6),
                        dbc.Col(html.Div(id='vapor-pressure-graph'), width=6)
                    ]
                )
            ],
            fluid=True,
            style={
                "height": "100vh",
                "background-image": "url('/assets/crops-background.jpg')",
                "background-size": "cover",
                "background-position": "center center",
            },
        ),
    ],
    style={
        "display": "flex",
        "flex-direction": "column",
        "justify-content": "center",
    },
)

@app.callback(
    [Output('prediction-result', 'children'),Output('prediction-result-2', 'children'),
     Output('temperature-graph', 'children'),
     Output('precipitation-graph', 'children'),
     Output('vapor-pressure-graph', 'children')],  # Output for the graph
    [Input('predict-button', 'n_clicks')],
    [State('input-year', 'value'), State('input-month', 'value')]
)
def update_output(n_clicks, year, month):
    if n_clicks > 0 and year and month:
        # Your existing code to call the prediction function and process results
        # For example
        crop_human_url = "http://127.0.0.1:9870/webhdfs/v1/user/nourishnation/crop_human.csv?op=OPEN"
        crop_data = fetch_csv_data(crop_human_url)
        crop_to_human_category = dict(zip(crop_data['label'], crop_data['human_category']))
        
        predicted_crops_with_category = predict_crops_from_saved_model(year, month.upper(), model, label_encoder, crop_to_human_category)
        crop_counts = Counter([crop for crop, category in predicted_crops_with_category])
        priority_list = [crop for crop, count in crop_counts.most_common()]
        
        all_possible_crops = set(crop_to_human_category.keys())
        non_predicted_crops = all_possible_crops - set(priority_list)
        complete_priority_list = priority_list + list(non_predicted_crops)

        # Convert the prediction result into a list of dictionaries for dash_table.DataTable
        data_table = [{"Crop": crop, "Category": crop_to_human_category.get(crop, 'Unknown category')} for crop in complete_priority_list]

        # Generating the temperature graph
        months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        # Generating the temperature graph
        webhdfs_url = "http://127.0.0.1:9870/webhdfs/v1/user/nourishnation/temp_future.csv?op=OPEN"
        response_temp = requests.get(webhdfs_url)
        temp_data = pd.read_csv(StringIO(response_temp.text))
        temp_data_year = temp_data[temp_data['YEAR'] == year]

        temperature_data = [temp_data_year[month].values[0] for month in months]

        temperature_graph = dcc.Graph(
            figure={
                'data': [
                    {'x': months, 'y': temperature_data, 'type': 'line', 'name': 'Temperature'},
                ],
                'layout': {
                    'title': f'Temperature for the Year {year}',
                    'xaxis': {'title': 'Month'},
                    'yaxis': {'title': 'Temperature (°C)'},
                    'paper_bgcolor': 'rgba(255, 255, 255, 0.5)',  # Semi-transparent background for the whole graph
                    'plot_bgcolor': 'rgba(255, 255, 255, 0.5)' 
                }
            }
        )
        webhdfs_url = "http://127.0.0.1:9870/webhdfs/v1/user/nourishnation/pre_future.csv?op=OPEN"
        response_pre = requests.get(webhdfs_url)
        prec_data = pd.read_csv(StringIO(response_pre.text))
        prec_data_year = prec_data[prec_data['YEAR'] == year]

        precipitation_data = [prec_data_year[month].values[0] for month in months]

        precipitation_graph = dcc.Graph(
            figure={
                'data': [
                    {'x': months, 'y': precipitation_data, 'type': 'line', 'name': 'Temperature'},
                ],
                'layout': {
                    'title': f'Precipitation for the Year {year}',
                    'xaxis': {'title': 'Month'},
                    'yaxis': {'title': 'millimiters (mm)'},
                    'paper_bgcolor': 'rgba(255, 255, 255, 0.5)',  # Semi-transparent background for the whole graph
                    'plot_bgcolor': 'rgba(255, 255, 255, 0.5)' 
                }
            }
        )

        #vapor pressure graph
        webhdfs_url = "http://127.0.0.1:9870/webhdfs/v1/user/nourishnation/vap_future.csv?op=OPEN"
        response_vap = requests.get(webhdfs_url)
        vapor_data = pd.read_csv(StringIO(response_vap.text))
        vapor_data_year = vapor_data[vapor_data['YEAR'] == year]

        vapor_pressure_data = [vapor_data_year[month].values[0] for month in months]

        vapor_pressure_graph = dcc.Graph(
            figure={
                'data': [
                    {'x': months, 'y': vapor_pressure_data, 'type': 'line', 'name': 'vapor pressure'},
                ],
                'layout': {
                    'title': f'Vapor Pressure for the Year {year}',
                    'xaxis': {'title': 'Month'},
                    'yaxis': {'title': 'torr (mm Hg)'},
                    'paper_bgcolor': 'rgba(255, 255, 255, 0.5)',  # Semi-transparent background for the whole graph
                    'plot_bgcolor': 'rgba(255, 255, 255, 0.5)' 
                }
            }
        )
        monthly_important_crops = []
        available_months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        for month in available_months:
            predicted_crops_with_category = predict_crops_from_saved_model(year, month, model, label_encoder, crop_to_human_category)
            # crop_counts = Counter([crop for crop, category in predicted_crops_with_category])
            # most_important_crop, _ = crop_counts.most_common(1)[0]
            # human_category = crop_to_human_category.get(most_important_crop, 'Unknown category')
            # monthly_important_crops.append((month, most_important_crop, human_category))
            if predicted_crops_with_category:
                for i in predicted_crops_with_category:
                    first_crop, human_category = predicted_crops_with_category[0]
            else:
                first_crop, human_category = 'Unknown crop', 'Unknown category'
            monthly_important_crops.append((month, first_crop, human_category))
    
        # Create DataFrame for displaying in DataTable
        df = pd.DataFrame(monthly_important_crops, columns=['Month', 'Crop', 'Human Category'])
 
        table_heading = html.H3("Predicted Crops", style={'color': '#228B22', 'textAlign': 'center'})
        table_heading_2 = html.H3("Top Crops", style={'color': '#228B22', 'textAlign': 'center'});
        return (html.Div([table_heading,dash_table.DataTable(
            data=data_table,
            columns=[{"name": "Crop", "id": "Crop"}, {"name": "Human category", "id": "Category"}],
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgba(248, 248, 248, 0.5)'},
                {'if': {'row_index': 'even'}, 'backgroundColor': 'rgba(255, 255, 255, 0.5)'}
            ],
            style_header={
                'backgroundColor': 'rgba(30, 30, 30, 0.8)',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center',
            },
            style_cell={
                'padding': '10px',
                'textAlign': 'center',
                'color': 'black',
                'border': '1px solid grey'
            },
            page_size=10,  # Set page_size to 10 for pagination
        )]),html.Div([table_heading_2,
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
             style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgba(248, 248, 248, 0.5)'},
                {'if': {'row_index': 'even'}, 'backgroundColor': 'rgba(255, 255, 255, 0.5)'}
            ],
            style_header={
                'backgroundColor': 'rgba(30, 30, 30, 0.8)',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center',
            },
            style_cell={
                'padding': '10px',
                'textAlign': 'center',
                'color': 'black',
                'border': '1px solid grey'
            },
            page_size=10
        )
    ]), temperature_graph, precipitation_graph, vapor_pressure_graph)
    
    # Return a default message when the page loads for the first time
    return (html.Div("Enter year and month, then click predict.", style={'color': 'black'}),None, None, None, None)

if __name__ == '__main__':
    app.run_server(debug=True)
