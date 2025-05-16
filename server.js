const express = require("express");
const cors = require("cors");
const multer = require("multer");
const { PythonShell } = require("python-shell");
const fs = require("fs");
const path = require("path");

// Add debug mode flag
const DEBUG = true;

// Debug logger function
function debug(message, data = null) {
  if (DEBUG) {
    if (data) {
      console.log(`[DEBUG] ${message}`, data);
    } else {
      console.log(`[DEBUG] ${message}`);
    }
  }
}

const app = express();
const port = 5000;
const upload = multer({ storage: multer.memoryStorage() });

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname)));

debug("Middleware initialized");

// Check if Python models exist
function checkModelsExist() {
  const requiredModels = [
    "crop_stacking_model.pkl",
    "crop_label_encoder.pkl",
    "crop_scaler.pkl",
    "fertilizer_stacking_model.pkl",
    "fertilizer_label_encoder.pkl",
    "fertilizer_scaler.pkl",
    "disease_rf_model.pkl",
    "disease_label_map.pkl",
  ];

  debug("Checking for model files...");
  const missingModels = [];

  for (const model of requiredModels) {
    if (!fs.existsSync(path.join(__dirname, model))) {
      missingModels.push(model);
    }
  }

  if (missingModels.length > 0) {
    debug("Missing model files:", missingModels);
    return false;
  }

  debug("All model files found");
  return true;
}

// Check models on startup
const modelsExist = checkModelsExist();
if (!modelsExist) {
  console.error("WARNING: Some model files are missing. Predictions may fail.");
}

// Serve index.html
app.get("/", (req, res) => {
  debug("Serving index.html");
  res.sendFile(path.join(__dirname, "index.html"));
});

// Crop prediction endpoint
app.post("/predict_crop", (req, res) => {
  debug("Crop prediction request received", req.body);
  const { features } = req.body;

  if (!features || !Array.isArray(features) || features.length !== 7) {
    debug("Invalid crop features provided", features);
    return res
      .status(400)
      .json({ error: "Invalid features. 7 numeric values required." });
  }

  const options = {
    mode: "json",
    pythonPath: "python", // Use 'python3' if needed
    pythonOptions: ["-u"], // unbuffered
    scriptPath: __dirname,
    stdin: true,
    stderrParser: (line) => {
      console.log(`Python stderr: ${line}`);
      return line;
    },
  };

  debug("Starting Python process for crop prediction");
  const pyshell = new PythonShell("crop_predictor.py", options);

  // Add timeout for Python process
  const timeout = setTimeout(() => {
    debug("Python process timed out (30s)");
    pyshell.terminate();
    res.status(504).json({
      error: "Processing timed out. Python script took too long to respond.",
    });
  }, 30000);

  // Create data object and send it directly (not as a JSON string)
  const inputData = { features };
  debug("Data to be sent to Python", inputData);

  // Write raw data directly to stdin
  pyshell.send(JSON.stringify(inputData));

  pyshell.on("message", (message) => {
    clearTimeout(timeout);
    debug("Received response from Python process", message);
    res.json(message);
  });

  pyshell.on("error", (err) => {
    clearTimeout(timeout);
    debug("Python Error:", err.message);
    console.error("Python Error:", err);
    res.status(500).json({ error: err.message });
  });

  pyshell.end((err, exitCode, exitSignal) => {
    debug("Python process ended", { err, exitCode, exitSignal });
    if (err) console.error("End Error:", err);
  });
});

// Fertilizer prediction endpoint
app.post("/predict_fertilizer", (req, res) => {
  debug("Fertilizer prediction request received", req.body);
  const { features } = req.body;

  if (!features || !Array.isArray(features) || features.length !== 14) {
    debug("Invalid fertilizer features provided", features);
    return res
      .status(400)
      .json({ error: "Invalid features. Expected 14 values." });
  }

  const options = {
    mode: "json",
    pythonPath: "python", // Use 'python3' if needed
    pythonOptions: ["-u"], // unbuffered
    scriptPath: __dirname,
    stdin: true,
    stderrParser: (line) => {
      console.log(`Python stderr: ${line}`);
      return line;
    },
  };

  debug("Starting Python process for fertilizer prediction");
  const pyshell = new PythonShell("fertilizer_predictor.py", options);

  // Add timeout for Python process
  const timeout = setTimeout(() => {
    debug("Python process timed out (30s)");
    pyshell.terminate();
    res.status(504).json({
      error: "Processing timed out. Python script took too long to respond.",
    });
  }, 30000);

  // Create data object and send it directly (not as a JSON string)
  const inputData = { features };
  debug("Data to be sent to Python", inputData);

  // Write raw data directly to stdin
  pyshell.send(JSON.stringify(inputData));

  pyshell.on("message", (message) => {
    clearTimeout(timeout);
    debug("Received response from Python process", message);
    res.json(message);
  });

  pyshell.on("error", (err) => {
    clearTimeout(timeout);
    debug("Python Error:", err.message);
    console.error("Python Error:", err);
    res.status(500).json({ error: err.message });
  });

  pyshell.end((err, exitCode, exitSignal) => {
    debug("Python process ended", { err, exitCode, exitSignal });
    if (err) console.error("End Error:", err);
  });
});

// Disease prediction endpoint
app.post("/predict_disease", upload.single("image"), (req, res) => {
  debug("Disease prediction request received");

  if (!req.file) {
    debug("No image uploaded");
    return res.status(400).json({ error: "No image uploaded" });
  }

  // Convert image to base64
  const imageBase64 = req.file.buffer.toString("base64");
  debug("Image converted to base64, size:", imageBase64.length);

  const options = {
    mode: "json",
    pythonPath: "python", // Use 'python3' if needed
    pythonOptions: ["-u"], // unbuffered
    scriptPath: __dirname,
    stdin: true,
    stderrParser: (line) => {
      console.log(`Python stderr: ${line}`);
      return line;
    },
  };

  debug("Starting Python process for disease prediction");
  const pyshell = new PythonShell("disease_predictor.py", options);

  // Add timeout for Python process
  const timeout = setTimeout(() => {
    debug("Python process timed out (60s)");
    pyshell.terminate();
    res.status(504).json({
      error: "Processing timed out. Python script took too long to respond.",
    });
  }, 60000);

  // Create data object and send it directly (not as a JSON string)
  const inputData = { image: imageBase64 };
  debug("Image data prepared for Python. Length:", imageBase64.length);

  // Write raw data directly to stdin
  pyshell.send(JSON.stringify(inputData));

  pyshell.on("message", (message) => {
    clearTimeout(timeout);
    debug("Received response from Python process", message);
    res.json(message);
  });

  pyshell.on("error", (err) => {
    clearTimeout(timeout);
    debug("Python Error:", err.message);
    console.error("Python Error:", err);
    res.status(500).json({ error: err.message });
  });

  pyshell.end((err, exitCode, exitSignal) => {
    debug("Python process ended", { err, exitCode, exitSignal });
    if (err) console.error("End Error:", err);
  });
});

// Test endpoint for quick validation of server functionality
app.get("/test", (req, res) => {
  debug("Test endpoint hit");
  res.json({ status: "Server is running", time: new Date().toISOString() });
});

// Start server
app.listen(port, () => {
  debug("Server started successfully");
  console.log(`Server running at http://localhost:${port}`);
  console.log(`Debug mode: ${DEBUG ? "ON" : "OFF"}`);
});
