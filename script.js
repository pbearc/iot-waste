const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureBtn = document.getElementById("captureBtn");
const resultDiv = document.getElementById("result");
const errorDiv = document.getElementById("error");
const context = canvas.getContext("2d");

let stream = null; // To hold the camera stream

// Function to handle errors
function displayError(message) {
  console.error(message);
  errorDiv.textContent = `Error: ${message}`;
  resultDiv.textContent = "Prediction failed";
  resultDiv.classList.remove("loading");
  captureBtn.disabled = true; // Disable button on error
}

// Access the camera
async function setupCamera() {
  try {
    // Request video stream
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" }, // Prefer back camera
    });
    video.srcObject = stream;

    // Wait for the video metadata to load to get dimensions
    video.onloadedmetadata = () => {
      console.log("Camera stream started.");
      // Set canvas dimensions once video is ready
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      captureBtn.textContent = "Classify";
      captureBtn.disabled = false; // Enable button
      errorDiv.textContent = ""; // Clear previous errors
    };
  } catch (err) {
    // Handle specific errors or provide general message
    if (err.name === "NotAllowedError") {
      displayError(
        "Camera access denied. Please allow camera permission in your browser settings."
      );
    } else if (err.name === "NotFoundError") {
      displayError(
        "No camera found. Make sure a camera is connected and enabled."
      );
    } else {
      displayError(`Could not access camera - ${err.name}: ${err.message}`);
    }
    captureBtn.textContent = "Camera Error";
  }
}

// Function to capture frame and send for prediction
async function captureAndPredict() {
  if (!stream || !stream.active) {
    displayError("Camera stream is not active.");
    setupCamera(); // Try to restart camera
    return;
  }

  captureBtn.disabled = true;
  captureBtn.textContent = "Processing...";
  resultDiv.textContent = "Capturing and predicting...";
  resultDiv.classList.add("loading");
  errorDiv.textContent = ""; // Clear previous errors

  try {
    // Ensure canvas dimensions match current video dimensions (if resizable)
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw the current video frame onto the hidden canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert the canvas image to a base64 data URL (JPEG format for potentially smaller size)
    const imageDataUrl = canvas.toDataURL("image/jpeg", 0.9); // Quality 0.9

    // Send the image data to the backend prediction endpoint
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image: imageDataUrl }), // Send as JSON
    });

    if (!response.ok) {
      // Try to get error message from backend response body
      let errorMsg = `HTTP error! Status: ${response.status}`;
      try {
        const errorData = await response.json();
        errorMsg = errorData.error || errorMsg; // Use backend error if available
      } catch (jsonError) {
        // Ignore if response body is not valid JSON
        console.warn("Could not parse error response JSON:", jsonError);
      }
      throw new Error(errorMsg);
    }

    // Parse the JSON response from the backend
    const result = await response.json();

    // Display the prediction
    if (result.prediction) {
      resultDiv.textContent = `Predicted: ${result.prediction} (Confidence: ${
        result.confidence || "N/A"
      })`;
    } else if (result.error) {
      throw new Error(`Backend error: ${result.error}`);
    } else {
      throw new Error(
        "Received an unexpected response format from the server."
      );
    }
    resultDiv.classList.remove("loading");
  } catch (err) {
    displayError(`Prediction failed - ${err.message}`);
  } finally {
    // Re-enable the button after processing is complete (success or fail)
    captureBtn.disabled = false;
    captureBtn.textContent = "Classify";
  }
}

// --- Event Listeners ---
// Add event listener to the capture button
captureBtn.addEventListener("click", captureAndPredict);

// Initialize the camera when the page loads
window.addEventListener("load", setupCamera);

// Optional: Stop the camera stream when the page is closed or navigated away from
window.addEventListener("beforeunload", () => {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    console.log("Camera stream stopped.");
  }
});
