#!/usr/bin/env python3
"""
Test script for the Magma API service.
This script tests if the API is running correctly and can load the model.
"""
import argparse
import base64
import sys
import time

import requests


def test_health(base_url):
    """Test the health endpoint"""
    print("\nTesting health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
        if response.status_code == 200:
            if response.json().get("model_loaded"):
                print("Health check passed! Model is loaded.")
            else:
                print("Health check response OK but model is not loaded yet.")
            return True
        else:
            print("Health check failed!")
            return False
    except Exception as e:
        print(f"Error connecting to health endpoint: {str(e)}")
        return False

def test_prediction(base_url, image_path=None):
    """Test the prediction endpoint with an image"""
    if not image_path:
        print("\nSkipping prediction test (no image provided)")
        return None
        
    print(f"\nTesting prediction endpoint with image: {image_path}")
    try:
        # Load and encode the image
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            
        # First try with base64
        print("Testing base64 prediction...")
        encoded_img = base64.b64encode(img_data).decode('utf-8')
        response = requests.post(
            f"{base_url}/predict",
            json={"image": encoded_img, "prompt": "What can you see in this image?"},
            timeout=60
        )
        
        if response.status_code == 200:
            print("Base64 prediction succeeded!")
            result = response.json()
            print(f"Response text: {result.get('text_response', '')[:100]}...")
            print(f"Normalized actions: {result.get('normalized_actions', [])}")
            print(f"Delta values: {result.get('delta_values', [])}")
        else:
            print(f"Base64 prediction failed! Status: {response.status_code}")
            print(f"Response: {response.text}")
            
        # Now try with file upload
        print("\nTesting file upload prediction...")
        files = {"file": open(image_path, "rb")}
        data = {"prompt": "What can you see in this image?"}
        response = requests.post(
            f"{base_url}/predict_from_file",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            print("File upload prediction succeeded!")
            result = response.json()
            print(f"Response text: {result.get('text_response', '')[:100]}...")
        else:
            print(f"File upload prediction failed! Status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"Error testing prediction: {str(e)}")
        return None

def wait_for_service(base_url, max_retries=10, retry_delay=5):
    """Wait for the service to be available"""
    print(f"Waiting for service at {base_url}...")
    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"Service is up after {i+1} attempts!")
                return True
            else:
                print(f"Attempt {i+1}/{max_retries}: Service responded with status {response.status_code}")
        except Exception as e:
            print(f"Attempt {i+1}/{max_retries}: {str(e)}")
        
        if i < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    print("Service did not become available in time")
    return False

def main():
    parser = argparse.ArgumentParser(description="Test the Magma API service")
    parser.add_argument("--url", default="http://localhost:8080", help="Base URL of the API")
    parser.add_argument("--image", help="Path to an image file for testing prediction")
    parser.add_argument("--wait", action="store_true", help="Wait for service to be available")
    args = parser.parse_args()
    
    if args.wait:
        if not wait_for_service(args.url):
            sys.exit(1)
    
    if test_health(args.url):
        test_prediction(args.url, args.image)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
