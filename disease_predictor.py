import sys
import json
import joblib
import numpy as np
import cv2
from PIL import Image
import io
import base64
from skimage.feature import hog, local_binary_pattern
import traceback

def preprocess_image(image_data):
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((128, 128))
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # HOG
        hog_feat = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
        # LBP
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 8 + 3), range=(0, 8 + 2))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-6)
        
        # Color histogram
        hist = cv2.calcHist([img_np], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Combine features
        features = np.concatenate([hog_feat, lbp_hist, hist])
        return features.reshape(1, -1)
    except Exception as e:
        print(f"Image preprocessing error: {str(e)}", file=sys.stderr)
        raise

def predict_disease(image_data):
    try:
        # Load models
        disease_model = joblib.load('disease_rf_model.pkl')
        disease_label_map = joblib.load('disease_label_map.pkl')
        
        # Process image
        features = preprocess_image(image_data)
        
        # Make prediction
        pred = disease_model.predict(features)[0]
        label = disease_label_map[pred]
        
        return {"prediction": label}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    try:
        # Read input from stdin
        input_json = sys.stdin.read().strip()
        
        # Debug output
        print(f"Debug - Raw input length: {len(input_json)} chars", file=sys.stderr)
        
        # Handle potential double-encoded JSON
        # Try first as regular JSON
        try:
            input_data = json.loads(input_json)
            
            # Check if the input_data is a string that might contain JSON
            if isinstance(input_data, str):
                try:
                    # Try to parse it again as JSON
                    print(f"Input appears to be a JSON string, trying to parse again", file=sys.stderr)
                    input_data = json.loads(input_data)
                except json.JSONDecodeError:
                    # If it's not valid JSON, keep it as is
                    pass
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"JSON parsing error: {str(e)}"}))
            sys.exit(1)
            
        # Validate input structure
        if not isinstance(input_data, dict):
            print(json.dumps({"error": f"Expected JSON object, got {type(input_data).__name__}"}))
            sys.exit(1)
        
        # Get image data
        image_data = input_data.get('image', '')
        
        if not image_data:
            print(json.dumps({"error": "No image data provided"}))
            sys.exit(1)
            
        # Predict
        result = predict_disease(image_data)
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        # Catch all other exceptions
        error_message = {"error": str(e), "traceback": traceback.format_exc()}
        print(json.dumps(error_message))
        sys.exit(1) 