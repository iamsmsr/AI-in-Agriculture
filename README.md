# Agri AI - Smart Agriculture Application

**An Ensemble Stacking Approach with Web Application for Crop, Fertilizer, and Disease Recommendation
A modern, user-friendly application that provides AI-powered agricultural recommendations using machine learning models.**

## Live Demo

**Try it out:** [https://agri-ai-325y.onrender.com/](https://agri-ai-325y.onrender.com/)

## Features

- **Crop Recommendation**: Suggests the best crops to plant based on soil composition and environmental factors
- **Fertilizer Recommendation**: Recommends the optimal fertilizer based on soil nutrient levels and crop type
- **Plant Disease Detection**: Identifies plant diseases from leaf images

## Technology Stack

- **Frontend**: HTML, CSS, JavaScript with modern UI
- **Backend**: Node.js with Express.js
- **Machine Learning**: Pre-trained Python models integrated with Node.js via python-shell
- **Image Processing**: OpenCV and scikit-image for disease detection

## Prerequisites

- Node.js (v14+ recommended)
- Python 3.6+ with pip
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository

   ```
   git clone <repository-url>
   ```

2. Install Node.js dependencies

   ```
   npm install
   ```

3. Install Python dependencies
   ```
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the Express.js server:

   ```
   npm start
   ```

   For development with auto-reload:

   ```
   npm run dev
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## API Endpoints

- **POST /predict_crop**: Recommends crops based on soil and environmental parameters
  - Parameters: N, P, K, temperature, humidity, pH, rainfall
- **POST /predict_fertilizer**: Suggests fertilizers based on soil, crop, and environmental factors
  - Parameters: Temperature, Moisture, Rainfall, pH, N, P, K, Carbon, Soil type, Crop type
- **POST /predict_disease**: Identifies plant diseases from leaf images
  - Parameters: Image file upload

## Project Structure

- `server.js`: Express.js server that handles HTTP requests
- `index.html`: Frontend interface with modern UI
- Python scripts:
  - `crop_predictor.py`: Crop recommendation logic
  - `fertilizer_predictor.py`: Fertilizer recommendation logic
  - `disease_predictor.py`: Disease identification from images
- Pre-trained models:
  - `*_model.pkl`: Machine learning models
  - `*_scaler.pkl`: Feature scalers
  - `*_label_encoder.pkl`: Label encoders

## Deployment

The application is deployed on Render:

- **URL**: [https://agri-ai-325y.onrender.com/](https://agri-ai-325y.onrender.com/)
- **Platform**: Render (Free tier)
- **Build Command**: `npm install && pip install -r requirements.txt`
- **Start Command**: `node server.js`

## Troubleshooting

- If you encounter Python module errors, ensure all dependencies are installed:

  ```
  pip install scikit-learn==1.5.1 numpy opencv-python pandas joblib
  ```

- For issues with the Node.js server, check the console logs with the detailed debugging information

## License

[MIT License](LICENSE)
