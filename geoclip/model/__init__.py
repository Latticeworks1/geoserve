"""
GeoCLIP: A powerful and intuitive image geolocalization library.

This library provides tools for predicting geographical locations from images
using the GeoCLIP (Geographical CLIP) model.

Main components:
- GeoCLIP: The main class for image geolocalization.
- ImageEncoder: Encodes images into feature vectors.
- LocationEncoder: Encodes geographical coordinates into feature vectors.
- locate: A simplified interface for quick predictions.
- utils: Utility functions for data handling, evaluation, and other operations.

Example usage:
    from geoclip import locate

    # Predict location for a single image
    result = locate.image("path/to/image.jpg")
    lat, lon, prob = result[0][0]
    print(f"Predicted location: ({lat:.6f}, {lon:.6f}), Confidence: {prob:.4f}")

For more advanced usage, refer to the documentation of individual modules.
"""

from .GeoCLIP import GeoCLIP
from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder
from .locate import locate, image
from .utils import set_seed, load_gps_data, haversine_distance, evaluate_predictions

__all__ = [
    "GeoCLIP", 
    "ImageEncoder", 
    "LocationEncoder", 
    "locate", 
    "image", 
    "set_seed", 
    "load_gps_data", 
    "haversine_distance", 
    "evaluate_predictions"
]

__version__ = "1.0.0"
