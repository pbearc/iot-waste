// --- Get DOM Elements ---
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureBtn = document.getElementById("captureBtn");
const resultDiv = document.getElementById("result");
const errorDiv = document.getElementById("error");
const context = canvas.getContext("2d");

// Feedback elements
const feedbackSection = document.getElementById("feedbackSection");
const feedbackYesBtn = document.getElementById("feedbackYesBtn");
const feedbackNoBtn = document.getElementById("feedbackNoBtn");
const correctionDetails = document.getElementById("correctionDetails");
const correctLabelSelect = document.getElementById("correctLabelSelect");
const submitCorrectionBtn = document.getElementById("submitCorrectionBtn");
const feedbackMessage = document.getElementById("feedbackMessage");

// --- Configuration ---
const CLOUD_VM_IP = "34.124.177.124";
// !!! ------------------------------------------------------- !!!
const CLOUD_API_BASE_URL = `http://${CLOUD_VM_IP}:5000`;
const PREDICT_URL = `${CLOUD_API_BASE_URL}/predict`;
const SUBMIT_CORRECTION_URL = `${CLOUD_API_BASE_URL}/submit_correction`;

// IMPORTANT: MUST MATCH TRAINING ORDER AND app.py
const CLASS_LABELS = [
  "cardboard",
  "glass",
  "metal",
  "paper",
  "plastic",
  "trash",
];

const MODEL_INPUT_WIDTH = 224; // Standard size for many image classification models
const MODEL_INPUT_HEIGHT = 224; // Typically square for classification models
// -------------------------------------------------

// --- State Variables ---
let stream = null; // Holds the camera stream
let currentImageDataUrl = null; // Stores the most recently captured image base64 data

// --- Utility Functions ---

// Add this new function to properly resize for prediction
function resizeImageForPrediction(sourceCanvas) {
  // Create a temporary canvas at the exact size needed by the model
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = MODEL_INPUT_WIDTH;
  tempCanvas.height = MODEL_INPUT_HEIGHT;
  const tempCtx = tempCanvas.getContext("2d");

  // Draw the captured image, scaling to fit the model's expected size
  tempCtx.drawImage(sourceCanvas, 0, 0, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);

  // Return the properly sized image data URL
  return tempCanvas.toDataURL("image/jpeg", 0.9);
}

function displayError(message) {
  console.error("Error:", message);
  errorDiv.textContent = `Error: ${message}`;
  errorDiv.style.display = "block"; // Show error div
  resultDiv.textContent = "Prediction Failed";
  resultDiv.classList.remove("loading");
  feedbackSection.style.display = "none"; // Hide feedback on error
  // Don't disable capture button permanently on non-camera errors
  if (!message.toLowerCase().includes("camera")) {
    captureBtn.disabled = false;
    captureBtn.textContent = "Classify Waste";
  } else {
    captureBtn.textContent = "Camera Error";
    captureBtn.disabled = true;
  }
}

function hideError() {
  errorDiv.textContent = "";
  errorDiv.style.display = "none";
}

function populateLabelDropdown() {
  correctLabelSelect.innerHTML = ""; // Clear existing options
  CLASS_LABELS.forEach((label) => {
    const option = document.createElement("option");
    option.value = label;
    option.textContent = label.charAt(0).toUpperCase() + label.slice(1); // Capitalize
    correctLabelSelect.appendChild(option);
  });
}

// --- Core Functions ---
async function setupCamera() {
  hideError(); // Clear previous errors
  resultDiv.textContent = "Initializing Camera...";
  resultDiv.classList.add("loading");
  try {
    const constraints = {
      video: {
        facingMode: "environment", // Prefer back camera
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
    };
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
        "Camera access (getUserMedia) is not supported by this browser."
      );
    }
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      // Use onloadedmetadata for initial setup
      console.log("Camera stream metadata loaded.");
      canvas.width = video.videoWidth; // Set canvas size based on actual stream dimensions
      canvas.height = video.videoHeight;
      console.log(`Canvas dimensions set: ${canvas.width}x${canvas.height}`);
    };
    video.onplaying = () => {
      console.log("Camera stream playing.");
      captureBtn.textContent = "Classify Waste";
      captureBtn.disabled = false;
      resultDiv.textContent = "Ready to Classify";
      resultDiv.classList.remove("loading");
      hideError(); // Ensure error is hidden on success
    };
    populateLabelDropdown(); // Populate correction options
  } catch (err) {
    let errorMsg = `Could not access camera. `;
    if (
      err.name === "NotAllowedError" ||
      err.name === "PermissionDeniedError"
    ) {
      errorMsg +=
        "Permission denied. Please grant camera access via browser settings.";
    } else if (
      err.name === "NotFoundError" ||
      err.name === "DevicesNotFoundError"
    ) {
      errorMsg += "No camera found.";
    } else if (
      err.name === "NotReadableError" ||
      err.name === "TrackStartError"
    ) {
      errorMsg += "Camera might be in use or unreadable.";
    } else {
      errorMsg += `(${err.name || "Unknown Error"})`;
    }
    displayError(errorMsg);
    // Keep resultDiv indicating failure
    resultDiv.textContent = "Camera Error";
    resultDiv.classList.remove("loading");
    captureBtn.disabled = true; // Disable button if camera failed
    captureBtn.textContent = "Camera Error";
  }
}

async function captureAndPredict() {
  hideError(); // Clear previous errors
  if (!stream || !stream.active || video.readyState < video.HAVE_CURRENT_DATA) {
    displayError("Camera stream not ready. Please wait or reload.");
    return;
  }

  // Reset UI state for new prediction
  captureBtn.disabled = true;
  captureBtn.textContent = "Processing...";
  resultDiv.textContent = "Capturing & Sending...";
  resultDiv.classList.add("loading");
  currentImageDataUrl = null; // Clear previous image data
  feedbackSection.style.display = "none";
  correctionDetails.style.display = "none";
  feedbackMessage.textContent = "";
  submitCorrectionBtn.disabled = true; // Disable until needed

  try {
    // Ensure canvas size matches video frame size
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Store the full resolution version for correction submissions
    const fullResImageDataUrl = canvas.toDataURL("image/jpeg", 0.9);
    currentImageDataUrl = fullResImageDataUrl;

    // Then resize to model's expected input size
    const modelReadyImageDataUrl = resizeImageForPrediction(canvas);

    console.log(`Sending normalized image to: ${PREDICT_URL}`);
    console.log(
      `Image dimensions - Original: ${canvas.width}x${canvas.height}, Model Input: ${MODEL_INPUT_WIDTH}x${MODEL_INPUT_HEIGHT}`
    );

    // --- Send to Backend for Prediction ---
    const response = await fetch(PREDICT_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
      },
      body: JSON.stringify({ image: modelReadyImageDataUrl }),
    });

    console.log(`Received response status: ${response.status}`);
    resultDiv.classList.remove("loading"); // Remove loading state once response received

    if (!response.ok) {
      let errorDetail = `Prediction failed (HTTP ${response.status})`;
      try {
        const errorData = await response.json();
        errorDetail = errorData.error || errorDetail;
      } catch {
        /* Ignore if response body isn't JSON */
      }
      throw new Error(errorDetail);
    }

    const result = await response.json();
    console.log("Received prediction:", result);

    // Display the prediction
    if (result.prediction) {
      resultDiv.textContent = `Prediction: ${result.prediction}`;
      if (result.confidence) {
        resultDiv.textContent += ` (${result.confidence})`;
      }
      feedbackSection.style.display = "block"; // Show feedback options
      submitCorrectionBtn.disabled = false; // Enable submit button now image data exists
    } else {
      throw new Error(result.error || "Received unexpected response format.");
    }
  } catch (err) {
    displayError(err.message || "An unknown error occurred during prediction.");
    currentImageDataUrl = null; // Clear image data on error
  } finally {
    // Re-enable capture button if not disabled by camera error
    if (captureBtn.textContent !== "Camera Error") {
      captureBtn.disabled = false;
      captureBtn.textContent = "Classify Waste";
    }
  }
}

async function submitCorrection() {
  if (!currentImageDataUrl) {
    displayError("Cannot submit correction: No image data available.");
    return;
  }
  const selectedLabel = correctLabelSelect.value;
  if (!selectedLabel) {
    displayError("Cannot submit correction: No label selected.");
    return;
  }

  console.log(`Submitting correction: Label=${selectedLabel}`);
  feedbackMessage.textContent = "Submitting correction...";
  feedbackMessage.style.color = "orange";
  submitCorrectionBtn.disabled = true;
  feedbackYesBtn.disabled = true;
  feedbackNoBtn.disabled = true;

  try {
    const response = await fetch(SUBMIT_CORRECTION_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
      },
      body: JSON.stringify({
        image: currentImageDataUrl,
        label: selectedLabel,
      }),
    });

    if (response.ok) {
      const result = await response.json();
      console.log("Correction submitted:", result.message);
      feedbackMessage.textContent = "Correction submitted. Thank you!";
      feedbackMessage.style.color = "green";
      // Hide feedback section after a delay
      setTimeout(() => {
        feedbackSection.style.display = "none";
        correctionDetails.style.display = "none"; // Ensure details are hidden too
      }, 2500);
    } else {
      let errorDetail = `Correction submission failed (HTTP ${response.status})`;
      try {
        const errorData = await response.json();
        errorDetail = errorData.error || errorDetail;
      } catch {
        /* Ignore if response body isn't JSON */
      }
      throw new Error(errorDetail);
    }
  } catch (err) {
    console.error("Error submitting correction:", err);
    feedbackMessage.textContent = `Error: ${
      err.message || "Submission failed"
    }`;
    feedbackMessage.style.color = "red";
    // Re-enable buttons on failure
    submitCorrectionBtn.disabled = false;
    feedbackYesBtn.disabled = false;
    feedbackNoBtn.disabled = false;
  }
}

// --- Event Listeners ---
captureBtn.addEventListener("click", captureAndPredict);

feedbackYesBtn.addEventListener("click", () => {
  feedbackMessage.textContent = "Feedback noted. Thank you!";
  feedbackMessage.style.color = "green";
  correctionDetails.style.display = "none";
  feedbackYesBtn.disabled = true; // Disable buttons after choice
  feedbackNoBtn.disabled = true;
  // Optionally hide the whole section after a delay
  setTimeout(() => {
    feedbackSection.style.display = "none";
    // Re-enable for next prediction cycle? Or handled by captureAndPredict reset.
    feedbackYesBtn.disabled = false;
    feedbackNoBtn.disabled = false;
  }, 2000);
});

feedbackNoBtn.addEventListener("click", () => {
  correctionDetails.style.display = "block"; // Show dropdown and submit button
  feedbackMessage.textContent = ""; // Clear message
  feedbackYesBtn.disabled = true; // Disable Yes/No once No is chosen
  feedbackNoBtn.disabled = true;
});

submitCorrectionBtn.addEventListener("click", submitCorrection);

// --- Initialization ---
document.addEventListener("DOMContentLoaded", () => {
  // Check for camera support before attempting setup
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    setupCamera();
  } else {
    displayError(
      "Camera access (getUserMedia) is not supported by this browser."
    );
    captureBtn.disabled = true;
    captureBtn.textContent = "Unsupported Browser";
  }
});

// Optional: Function to stop camera (can be called from Android if needed)
function stopCameraStream() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    console.log("Camera stream stopped via JS call.");
    video.srcObject = null;
    stream = null;
    captureBtn.disabled = true;
    captureBtn.textContent = "Camera Off";
  }
}
