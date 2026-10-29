/**
 * Crop Disease Detection - Frontend Application Controller
 * Handles drag-and-drop, image preview, API health checking,
 * and asynchronous inference submission.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const btnBrowse = document.getElementById("btn-browse-file");
  const previewContainer = document.getElementById("preview-container");
  const imagePreview = document.getElementById("image-preview");
  const previewFilename = document.getElementById("preview-filename");
  const previewFilesize = document.getElementById("preview-filesize");
  const previewDimensions = document.getElementById("preview-dimensions");
  const btnResetPreview = document.getElementById("btn-reset-preview");
  const btnCancel = document.getElementById("btn-cancel");
  const btnSubmit = document.getElementById("btn-submit-analysis");
  const btnSpinner = document.getElementById("btn-spinner");
  const btnIcon = document.getElementById("btn-icon");
  const btnText = document.getElementById("btn-text");
  const messageContainer = document.getElementById("message-container");

  // Result Elements
  const resultsCard = document.getElementById("results-card");
  const uploadCard = document.getElementById("upload-card");
  const resultPredictedDisease = document.getElementById("result-predicted-disease");
  const resultHealthPill = document.getElementById("result-health-pill");
  const resultConfidencePercent = document.getElementById("result-confidence-percent");
  const resultConfidenceBar = document.getElementById("result-confidence-bar");
  const resultCropName = document.getElementById("result-crop-name");
  const resultClassKey = document.getElementById("result-class-key");
  const resultAlternativesList = document.getElementById("result-alternatives-list");
  const resultDescription = document.getElementById("result-description");
  const resultSymptoms = document.getElementById("result-symptoms");
  const resultPrevention = document.getElementById("result-prevention");
  const resultManagement = document.getElementById("result-management");
  const resultDisclaimerText = document.getElementById("result-disclaimer-text");
  const btnAnalyzeAnother = document.getElementById("btn-analyze-another");

  // Navbar Model Status Indicator
  const modelStatusIndicator = document.getElementById("model-status-indicator");

  let selectedFile = null;

  // 1. Initial Engine Health Check
  checkEngineHealth();

  async function checkEngineHealth() {
    if (!modelStatusIndicator) return;
    try {
      const response = await fetch("/api/health");
      if (!response.ok) throw new Error("Health check returned status " + response.status);
      const data = await response.json();

      if (data.model_loaded) {
        modelStatusIndicator.className = "status-pill online";
        modelStatusIndicator.innerHTML = `
          <i class="fa-solid fa-circle-check text-success"></i>
          <span>CNN Ready (${data.num_classes} Classes)</span>
        `;
        modelStatusIndicator.title = `Model loaded from ${data.model_path}`;
      } else {
        modelStatusIndicator.className = "status-pill offline";
        modelStatusIndicator.innerHTML = `
          <i class="fa-solid fa-triangle-exclamation text-warning"></i>
          <span>Model Offline (Train Model)</span>
        `;
        modelStatusIndicator.title = data.error || "Model artifact not found.";
      }
    } catch (err) {
      console.warn("Could not retrieve model health status:", err);
      modelStatusIndicator.className = "status-pill offline";
      modelStatusIndicator.innerHTML = `
        <i class="fa-solid fa-plug-circle-xmark"></i>
        <span>API Offline</span>
      `;
    }
  }

  // 2. Drag & Drop Event Listeners
  if (dropzone && fileInput) {
    btnBrowse.addEventListener("click", (e) => {
      e.stopPropagation();
      fileInput.click();
    });

    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });

    ["dragleave", "dragend"].forEach((evt) => {
      dropzone.addEventListener(evt, () => dropzone.classList.remove("dragover"));
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
      }
    });
  }

  // 3. File Processing & Preview
  function handleFileSelection(file) {
    clearMessage();

    // Extension validation
    const allowedExtensions = ["jpg", "jpeg", "png", "webp"];
    const ext = file.name.split(".").pop().toLowerCase();
    if (!allowedExtensions.includes(ext)) {
      showMessage(`Unsupported file format ('.${ext}'). Please select a JPG, JPEG, PNG, or WEBP image.`, "danger");
      return;
    }

    // Size limit validation (10 MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
      showMessage(`File size (${(file.size / (1024 * 1024)).toFixed(2)} MB) exceeds the maximum allowed limit of 10 MB.`, "danger");
      return;
    }

    selectedFile = file;

    // Display metadata
    previewFilename.textContent = file.name;
    previewFilesize.textContent = formatBytes(file.size);

    // Read and preview
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;

      // Extract image dimensions once loaded
      const tempImg = new Image();
      tempImg.onload = () => {
        previewDimensions.textContent = `${tempImg.naturalWidth} \u00D7 ${tempImg.naturalHeight} px`;
      };
      tempImg.src = e.target.result;

      dropzone.classList.add("d-none");
      previewContainer.classList.remove("d-none");
    };
    reader.readAsDataURL(file);
  }

  // 4. Reset & Clear Handlers
  if (btnResetPreview) btnResetPreview.addEventListener("click", resetSelection);
  if (btnCancel) btnCancel.addEventListener("click", resetSelection);

  function resetSelection() {
    selectedFile = null;
    fileInput.value = "";
    imagePreview.src = "";
    previewContainer.classList.add("d-none");
    dropzone.classList.remove("d-none");
    clearMessage();
  }

  // 5. Submit for Analysis
  if (btnSubmit) {
    btnSubmit.addEventListener("click", async () => {
      if (!selectedFile) {
        showMessage("Please select a leaf photograph first.", "warning");
        return;
      }

      setLoading(true);
      clearMessage();

      const formData = new FormData();
      formData.append("image", selectedFile);

      try {
        const response = await fetch("/api/predict", {
          method: "POST",
          body: formData
        });

        const data = await response.json();

        if (!response.ok) {
          const errorMsg = data.details || data.error || `Server returned error (${response.status})`;
          const instructions = data.instructions ? `\n\nSuggestion: ${data.instructions}` : "";
          throw new Error(errorMsg + instructions);
        }

        renderPredictionResults(data);
      } catch (err) {
        console.error("Prediction error:", err);
        showMessage(err.message || "Failed to complete prediction. Please verify your file.", "danger");
      } finally {
        setLoading(false);
      }
    });
  }

  // 6. Render Prediction Results
  function renderPredictionResults(data) {
    if (!resultsCard) return;

    // Predicted Class Header
    resultPredictedDisease.textContent = data.display_name || `${data.crop} - ${data.disease}`;

    // Health pill status
    if (data.is_healthy) {
      resultHealthPill.className = "badge-disease badge-healthy";
      resultHealthPill.innerHTML = '<i class="fa-solid fa-circle-check me-1"></i> Healthy Foliage';
    } else {
      resultHealthPill.className = "badge-disease badge-infected";
      resultHealthPill.innerHTML = '<i class="fa-solid fa-triangle-exclamation me-1"></i> Pathogen Detected';
    }

    // Confidence metric
    const confPercent = data.confidence_percentage || (data.confidence * 100).toFixed(1);
    resultConfidencePercent.textContent = `${confPercent}%`;
    resultConfidenceBar.style.width = `${confPercent}%`;

    // Metadata
    resultCropName.textContent = data.crop || "Unknown";
    resultClassKey.textContent = data.predicted_class || "N/A";

    // Top alternative predictions
    resultAlternativesList.innerHTML = "";
    if (data.top_predictions && data.top_predictions.length > 0) {
      data.top_predictions.forEach((item) => {
        const div = document.createElement("div");
        div.className = "alternative-item";
        div.innerHTML = `
          <div class="text-truncate me-2" style="max-width: 200px;">
            <span class="badge bg-secondary me-1">#${item.rank}</span>
            <span class="fw-semibold">${item.display_name}</span>
          </div>
          <div class="d-flex align-items-center gap-2">
            <span class="fw-bold">${item.confidence_percent}%</span>
          </div>
        `;
        resultAlternativesList.appendChild(div);
      });
    } else {
      resultAlternativesList.innerHTML = '<p class="small text-muted mb-0">No alternative scores available.</p>';
    }

    // Agronomic profile
    const info = data.disease_information || {};
    resultDescription.textContent = info.description || "No agronomic description provided.";

    // Symptoms
    renderList(resultSymptoms, info.symptoms, "No specific foliar symptoms cataloged.");
    // Prevention
    renderList(resultPrevention, info.prevention, "Standard good agricultural practices apply.");
    // Management
    renderList(resultManagement, info.management, "Consult a local agricultural extension specialist.");

    // Disclaimer
    if (data.disclaimer) {
      resultDisclaimerText.textContent = data.disclaimer;
    }

    // Show Results
    resultsCard.classList.remove("d-none");
    resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderList(container, items, fallbackText) {
    container.innerHTML = "";
    if (items && items.length > 0) {
      items.forEach((txt) => {
        const li = document.createElement("li");
        li.textContent = txt;
        container.appendChild(li);
      });
    } else {
      const li = document.createElement("li");
      li.className = "text-muted fst-italic";
      li.textContent = fallbackText;
      container.appendChild(li);
    }
  }

  // 7. Analyze Another Button
  if (btnAnalyzeAnother) {
    btnAnalyzeAnother.addEventListener("click", () => {
      resultsCard.classList.add("d-none");
      resetSelection();
      uploadCard.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  }

  // Helper UI functions
  function setLoading(isLoading) {
    if (isLoading) {
      btnSubmit.disabled = true;
      btnSpinner.classList.remove("d-none");
      btnIcon.classList.add("d-none");
      btnText.textContent = "Analyzing Foliage...";
    } else {
      btnSubmit.disabled = false;
      btnSpinner.classList.add("d-none");
      btnIcon.classList.remove("d-none");
      btnText.textContent = "Run CNN Diagnosis";
    }
  }

  function showMessage(text, type = "info") {
    if (!messageContainer) return;
    messageContainer.className = `alert alert-${type} mx-auto mb-4`;
    messageContainer.innerHTML = `
      <div class="d-flex align-items-center justify-content-between">
        <div>
          <i class="fa-solid fa-triangle-exclamation me-2"></i>
          <span>${text}</span>
        </div>
        <button type="button" class="btn-close" aria-label="Close" onclick="this.parentElement.parentElement.classList.add('d-none')"></button>
      </div>
    `;
    messageContainer.classList.remove("d-none");
  }

  function clearMessage() {
    if (!messageContainer) return;
    messageContainer.classList.add("d-none");
    messageContainer.textContent = "";
  }

  function formatBytes(bytes, decimals = 1) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
  }
});
