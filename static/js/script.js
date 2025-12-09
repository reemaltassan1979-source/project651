const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadSection = document.getElementById('uploadSection');
const previewSection = document.getElementById('previewSection');
const resultsSection = document.getElementById('resultsSection');
const loadingSection = document.getElementById('loadingSection');
const imagePreview = document.getElementById('imagePreview');
const classifyBtn = document.getElementById('classifyBtn');

let selectedFile = null;

// Drag and drop functionality
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Click to upload (except when clicking the button)
uploadArea.addEventListener('click', (e) => {
    // Don't trigger if clicking the browse button
    if (e.target.closest('.browse-btn')) {
        return;
    }
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Handle file selection
function handleFile(file) {
    // Check file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image file (JPG, PNG, GIF, or BMP)');
        return;
    }

    // Check file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        alert('File size should not exceed 16MB');
        return;
    }

    selectedFile = file;

    // Preview the image
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadSection.style.display = 'none';
        previewSection.style.display = 'block';
        resultsSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Classify button click
classifyBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        alert('Please select an image first');
        return;
    }

    // Show loading
    previewSection.style.display = 'none';
    loadingSection.style.display = 'block';

    // Create form data
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            alert('Error: ' + (data.error || 'Unknown error occurred'));
            resetUpload();
        }
    } catch (error) {
        alert('Error connecting to server: ' + error.message);
        resetUpload();
    } finally {
        loadingSection.style.display = 'none';
    }
});

// Display results
function displayResults(data) {
    document.getElementById('predictedClass').textContent = data.predicted_class.toUpperCase();
    document.getElementById('confidenceText').textContent = `Confidence: ${data.confidence.toFixed(2)}%`;
    
    // Animate confidence bar
    const confidenceFill = document.getElementById('confidenceFill');
    setTimeout(() => {
        confidenceFill.style.width = data.confidence + '%';
    }, 100);

    // Display all predictions
    const allPredictionsDiv = document.getElementById('allPredictions');
    allPredictionsDiv.innerHTML = '';
    
    data.all_predictions.forEach((pred) => {
        const predItem = document.createElement('div');
        predItem.className = 'prediction-item';
        predItem.innerHTML = `
            <span class="prediction-item-name">${pred.class}</span>
            <span class="prediction-item-confidence">${pred.confidence.toFixed(2)}%</span>
        `;
        allPredictionsDiv.appendChild(predItem);
    });

    resultsSection.style.display = 'block';
}

// Reset upload
function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    uploadSection.style.display = 'block';
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
    loadingSection.style.display = 'none';
    
    // Reset confidence bar
    document.getElementById('confidenceFill').style.width = '0%';
}

// Prevent default drag behavior on the whole page
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    document.body.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}
