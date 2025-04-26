import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import base64
import requests
import numpy as np
from PIL import Image
from io import BytesIO

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # Expose the server for deployment

# Backend API URL
API_URL = "http://localhost:8000"

# Load and prepare image
image_path = '/home/limbach/TUM_AI_Makeathon_2025/dev_data/input/DJI_20250424193300_0151_V.jpeg'

def image_to_base64(image_path):
    """Convert image to both raw and data-URI base64 formats"""
    with open(image_path, "rb") as image_file:
        raw_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        data_uri = f"data:image/png;base64,{raw_base64}"
        return raw_base64, data_uri

# Initialize the backend with our image
try:
    raw_base64, display_base64 = image_to_base64(image_path)
    response = requests.post(
        f"{API_URL}/set_image",
        json={"image_base64": raw_base64}  # Send raw base64 to backend
    )
    if response.status_code != 200:
        print("Failed to initialize backend:", response.text)
    else:
        print("Backend initialized successfully")
except Exception as e:
    print(f"Initialization error: {str(e)}")

# App layout
app.layout = html.Div([
    html.H1("SAM2 Interactive Segmentation"),
    html.Div([
        dcc.Graph(
            id="segmentation-image",
            figure={
                'data': [{
                    'type': 'image',
                    'source': display_base64,  # Use data-URI version for display
                }],
                'layout': {
                    'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0},
                    'clickmode': 'event+select',
                }
            },
            config={'scrollZoom': True},
            style={'height': '80vh'}
        ),
    ]),
    html.Div(id='mask-info'),
    html.Button('Reset Points', id='reset-button', n_clicks=0),
])

@app.callback(
    Output('segmentation-image', 'figure'),
    Output('mask-info', 'children'),
    Input('segmentation-image', 'clickData'),
    Input('reset-button', 'n_clicks'),
    State('segmentation-image', 'figure'),
)
def handle_click(clickData, reset_clicks, current_figure):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, ""
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'reset-button':
        response = requests.post(f"{API_URL}/reset")
        if response.status_code == 200:
            # Return to original image display
            fig = go.Figure(go.Image(source=display_base64))
            fig.update_layout(
                margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
                clickmode='event+select'
            )
            return fig, "Points reset"
        return dash.no_update, "Reset failed"
    
    if clickData:
        click_x = clickData['points'][0]['x']
        click_y = clickData['points'][0]['y']
        
        response = requests.post(
            f"{API_URL}/add_point",
            json={"x": click_x, "y": click_y, "label": 1}
        )
        
        if response.status_code == 200:
            data = response.json()
            fig = go.Figure(go.Image(source=display_base64))
            
            # Add masks to the figure (with data: prefix)
            for mask_base64 in data['masks']:
                fig.add_trace(go.Image(
                    source=f"data:image/png;base64,{mask_base64}",
                    opacity=0.5
                ))
            
            # Add points to the figure
            for pt in data['points']:
                fig.add_trace(go.Scatter(
                    x=[pt[0]], y=[pt[1]],
                    mode='markers',
                    marker=dict(color='red', size=10)
                ))
            
            fig.update_layout(
                margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
                clickmode='event+select'
            )
            
            message = f"Point added at ({click_x:.1f}, {click_y:.1f}). Best score: {max(data['scores']):.3f}"
            return fig, message
        
        return dash.no_update, "Failed to process point"
    
    return dash.no_update, ""

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)