"""
Geoserve API and Web Application

This module provides a FastAPI application for serving GeoCLIP model predictions
and a web interface for image geolocation prediction.

It includes endpoints for image upload, prediction, and a web frontend.
The application supports automatic device selection (CUDA, MPS, or CPU) and
provides robust error handling and logging.

Attributes:
    logger (logging.Logger): Logger for the module.
    app (FastAPI): The FastAPI application instance.

Usage:
    This module is intended to be run as a FastAPI application.
    Use 'uvicorn geoclip.app:app' to run the server.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from geoclip import GeoCLIP
from PIL import Image
from io import BytesIO
import logging
from typing import Dict, List
import torch
import folium
from folium.plugins import HeatMap
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def select_device() -> torch.device:
    """
    Selects the most appropriate device for model execution.

    Returns:
        torch.device: The selected device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        logger.info("CUDA is available. Using GPU.")
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("MPS is available. Using Metal GPU.")
        return torch.device("mps")
    else:
        logger.info("No GPU available. Using CPU.")
        return torch.device("cpu")

app = FastAPI(
    title="Geoserve API",
    description="API and Web Application for image geolocation prediction using GeoCLIP",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="geoclip/static"), name="static")

# Initialize model
device = select_device()
model = GeoCLIP(from_pretrained=True).to(device)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main page of the web application."""
    with open("geoclip/templates/index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 5):
    """
    Predicts geolocation for the uploaded image and returns prediction results and heatmap.

    Args:
        file (UploadFile): The uploaded image file.
        top_k (int): The number of top predictions to return. Defaults to 5.

    Returns:
        Dict: A dictionary containing predictions, heatmap HTML, and image data.

    Raises:
        HTTPException: If the prediction fails.
    """
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        top_pred_gps, top_pred_prob = model.predict(image, top_k)
        
        predictions = [
            {"lat": float(lat), "lon": float(lon), "probability": float(prob)}
            for (lat, lon), prob in zip(top_pred_gps, top_pred_prob)
        ]

        heatmap = generate_heatmap(top_pred_gps, top_pred_prob)
        
        # Convert image to base64 for frontend display
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return JSONResponse({
            "predictions": predictions,
            "heatmap": heatmap,
            "image": img_str
        })
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Checks the health status of the API.

    Returns:
        Dict[str, str]: A dictionary indicating the API status.
    """
    return {"status": "healthy"}

def generate_heatmap(top_pred_gps, top_pred_prob):
    """
    Generates a heatmap based on the predicted locations.

    Args:
        top_pred_gps (torch.Tensor): Top predicted GPS coordinates.
        top_pred_prob (torch.Tensor): Probabilities of top predictions.

    Returns:
        str: HTML string of the generated heatmap.
    """
    top_n_coordinates = 10
    gps_coordinates = top_pred_gps.tolist()[:top_n_coordinates]
    probabilities = top_pred_prob.tolist()[:top_n_coordinates]

    total_prob = sum(probabilities)
    normalized_probs = [prob / total_prob for prob in probabilities]

    weighted_coordinates = [(lat, lon, weight) for (lat, lon), weight in zip(gps_coordinates, normalized_probs)]

    avg_lat = sum(lat for lat, lon, weight in weighted_coordinates) / len(weighted_coordinates)
    avg_lon = sum(lon for lat, lon, weight in weighted_coordinates) / len(weighted_coordinates)

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=2.2)

    magma = {
        0.0: '#932667', 0.2: '#b5367a', 0.4: '#d3466b',
        0.6: '#f1605d', 0.8: '#fd9668', 1.0: '#fcfdbf'
    }

    HeatMap(weighted_coordinates, gradient=magma).add_to(m)

    top_coordinate = gps_coordinates[0]
    top_probability = normalized_probs[0]

    folium.Marker(
        location=top_coordinate,
        popup=f"Top Prediction: {top_coordinate} with probability {top_probability:.4f}",
        icon=folium.Icon(color='orange', icon='star')
    ).add_to(m)

    return m._repr_html_()

def create_app() -> FastAPI:
    """Return the FastAPI application instance."""
    return app

def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the API using uvicorn."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server()
