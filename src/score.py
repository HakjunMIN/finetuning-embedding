from __future__ import annotations

import json
import logging
import os
from typing import Dict, List

from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None

# Azure ML logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init() -> None:
    """
    This function is called when the container is initialized/started.
    """
    global _model
    try:
        # Azure ML sets AZUREML_MODEL_DIR to the path where the model is located
        model_dir = os.environ.get("AZUREML_MODEL_DIR")
        logger.info(f"AZUREML_MODEL_DIR: {model_dir}")
        
        if not model_dir:
            raise ValueError("AZUREML_MODEL_DIR environment variable not set")
        
        # List contents to debug
        logger.info(f"Contents of AZUREML_MODEL_DIR: {os.listdir(model_dir)}")
        
        # Azure ML might nest the model in subdirectories
        # Try to find the model files
        model_path = model_dir
        
        # Check if there's a subdirectory (common when registering from job outputs)
        subdirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        if subdirs:
            # Try the first subdirectory
            potential_path = os.path.join(model_dir, subdirs[0])
            logger.info(f"Found subdirectory: {subdirs[0]}")
            logger.info(f"Contents: {os.listdir(potential_path)}")
            
            # Check if it contains model files
            if any(f in os.listdir(potential_path) for f in ['config.json', 'model.safetensors', 'pytorch_model.bin']):
                model_path = potential_path
        
        logger.info(f"Loading model from: {model_path}")
        _model = SentenceTransformer(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        raise


def run(raw_data: str) -> str:
    """
    This function is called for every invocation of the endpoint.
    
    Expected input format:
    {
        "inputs": {
            "data": ["text1", "text2", ...]
        }
    }
    OR
    {
        "texts": ["text1", "text2", ...]
    }
    """
    try:
        data = json.loads(raw_data)
        logger.info(f"Received data: {data}")
        
        # Support both formats: {"inputs": {"data": [...]}} and {"texts": [...]}
        if "inputs" in data and "data" in data["inputs"]:
            texts = data["inputs"]["data"]
        elif "texts" in data:
            texts = data["texts"]
        else:
            raise ValueError("Input must contain either 'inputs.data' or 'texts' field with a list of strings")
        
        if not isinstance(texts, list):
            raise ValueError("'texts' or 'inputs.data' must be a list of strings")
        
        if _model is None:
            raise RuntimeError("Model is not loaded")
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = _model.encode(texts, normalize_embeddings=True).tolist()
        
        result = {"embeddings": embeddings}
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise
