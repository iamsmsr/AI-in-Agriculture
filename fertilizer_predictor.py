import sys
import json
import joblib
import numpy as np
import traceback

def predict_fertilizer(features):
    try:
        # Load models and scalers
        fertilizer_model = joblib.load('fertilizer_stacking_model.pkl')
        fertilizer_label_encoder = joblib.load('fertilizer_label_encoder.pkl')
        fertilizer_scaler = joblib.load('fertilizer_scaler.pkl')
        
        # Process features
        features_array = np.array(features).reshape(1, -1)
        features_scaled = fertilizer_scaler.transform(features_array)
        
        # Make prediction
        pred = fertilizer_model.predict(features_scaled)[0]
        fert_name = fertilizer_label_encoder.inverse_transform([pred])[0]
        
        return {"prediction": fert_name}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    try:
        # Read input from stdin
        input_json = sys.stdin.read().strip()
        
        # Debug output
        print(f"Debug - Raw input: {input_json}", file=sys.stderr)
        
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
            print(json.dumps({"error": f"JSON parsing error: {str(e)}", "input": input_json}))
            sys.exit(1)
            
        # Validate input structure
        if not isinstance(input_data, dict):
            print(json.dumps({"error": f"Expected JSON object, got {type(input_data).__name__}"}))
            sys.exit(1)
            
        # Get features
        features = input_data.get('features', [])
        
        # Validate features
        if not isinstance(features, list):
            print(json.dumps({"error": f"Expected features to be a list, got {type(features).__name__}"}))
            sys.exit(1)
            
        # Predict
        result = predict_fertilizer(features)
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        # Catch all other exceptions
        error_message = {"error": str(e), "traceback": traceback.format_exc()}
        print(json.dumps(error_message))
        sys.exit(1) 