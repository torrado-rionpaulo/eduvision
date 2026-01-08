document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('resultsSection').style.display = 'none';

    const startCameraBtn = document.getElementById('startCamera');
    const cameraSection = document.getElementById('cameraSection');
    const video = document.getElementById('video');
    const captureBtn = document.getElementById('capture');
    const canvas = document.getElementById('canvas');
    const uploadPhotoInput = document.getElementById('uploadPhoto');
    const normalVisionBox = document.getElementById('normalVisionBox');
    
    // --- New elements for analysis button and status ---
    const runAnalysisButton = document.getElementById('runAnalysisButton');
    const statusMessageBox = document.getElementById('statusMessage'); // NEW: Targeting the new HTML element

    let stream = null;
    let currentFile = null; // Global variable to hold the file/blob for analysis

    function updateFilteredImages(imageDataURL) {
        document.getElementById('protanopiaImg').src = imageDataURL;
        document.getElementById('protanopiaImg').style.filter = 'url(#protanopia)';

        document.getElementById('deuteranopiaImg').src = imageDataURL;
        document.getElementById('deuteranopiaImg').style.filter = 'url(#deuteranopia)';

        document.getElementById('tritanopiaImg').src = imageDataURL;
        document.getElementById('tritanopiaImg').style.filter = 'url(#tritanopia)';
    }
    
    function dataURLtoBlob(dataURL) {
        const byteString = atob(dataURL.split(',')[1]);
        const mimeString = dataURL.split(',')[0].split(':')[1].split(';')[0];
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
        }
        return new Blob([ab], { type: mimeString });
    }

    // --- EVENT LISTENERS ---

    startCameraBtn.addEventListener('click', async () => {
        cameraSection.style.display = 'block';
        document.getElementById('resultsSection').style.display = 'none';
        statusMessageBox.innerHTML = ''; // Clear status
        runAnalysisButton.style.display = 'none';
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
        } catch (err) {
            alert('Camera access denied or not available.');
            console.error(err);
        }
    });

    uploadPhotoInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }

            cameraSection.style.display = 'none';
            document.getElementById('resultsSection').style.display = 'none';
            statusMessageBox.innerHTML = ''; // Clear status
            
            const reader = new FileReader();
            reader.onload = function (event) {
                const imageData = event.target.result;
                // Display preview in normalVisionBox
                normalVisionBox.innerHTML = `<img src="${imageData}" alt="Uploaded Image" class="vision-image" />`;
                updateFilteredImages(imageData);
            };
            reader.readAsDataURL(file);

            currentFile = file;
            runAnalysisButton.style.display = 'inline-block';
        }
    });
    
    captureBtn.addEventListener('click', () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/png');
        normalVisionBox.innerHTML = `<img src="${imageData}" alt="Captured Image" class="vision-image" />`;
        updateFilteredImages(imageData);

        const blob = dataURLtoBlob(imageData);
        currentFile = new File([blob], 'captured.png', { type: 'image/png' });

        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }

        cameraSection.style.display = 'none';
        statusMessageBox.innerHTML = ''; // Clear status
        runAnalysisButton.style.display = 'inline-block';
    });
    
    // Handle the explicit Run Analysis button click
    runAnalysisButton.addEventListener('click', () => {
        if (currentFile) {
            sendImageToBackend(currentFile);
            runAnalysisButton.style.display = 'none';
        } else {
            alert("Please select or capture an image first.");
        }
    });

    // --- BACKEND COMMUNICATION (FIXED to use #statusMessage) ---

    async function sendImageToBackend(imageInput) {
        // *** FIX: Display status in the dedicated box, NOT the image box ***
        statusMessageBox.innerHTML = `<p><strong>Status:</strong> Running analysis... Please wait.</p>`;
        
        const formData = new FormData();
        const file = imageInput instanceof File
            ? imageInput
            : new File([imageInput], 'captured.png', { type: 'image/png' });
        
        formData.append('file', file); 

        // Helper function to handle error display and cleanup
        function displayError(message) {
            statusMessageBox.innerHTML = `<p style="color:red;">Error: ${message}</p>`;
        }

        try {
            const response = await fetch('/classify-image', { 
                method: 'POST',
                body: formData
            });

            console.log("📡 Response status:", response.status);
            
            if (!response.ok) {
                const errorText = await response.text();
                displayError(`Server Error (${response.status}): ${errorText.substring(0, 100)}...`);
                console.error("Server Error:", errorText);
                return;
            }

            const text = await response.text();
            console.log("📨 Raw response text:", text);

            let data;
            try {
                data = JSON.parse(text);
                console.log("✅ Parsed response:", data);
                
                // SUCCESS: Clear status message and update results
                statusMessageBox.innerHTML = '';
                updateResults(data);

            } catch (err) {
                console.error("❌ Failed to parse JSON:", err);
                displayError('Server returned invalid JSON format.');
            }
        } catch (err) {
            console.error("🚫 Fetch error:", err);
            displayError(`Network Error: Image analysis failed. Is your backend running on port 5000?`);
        }
    }

    // --- RENDERING FUNCTION ---
    
    function formatMisclassifiedList(misreads) {
        if (!misreads || misreads.length === 0) return '<p style="color:#1e8449; margin: 5px 0 0 0; font-size:0.9em;">(No misclassified segments)</p>';
        let html = '<div class="misreads"><strong>Misclassified Segments:</strong><ul style="margin: 5px 0 0 15px; padding: 0;">';
        misreads.slice(0, 5).forEach(m => { html += `<li style="font-size:0.9em;">${m}</li>`; });
        if (misreads.length > 5) html += `<li style="font-size:0.9em;">...and ${misreads.length - 5} more.</li>`;
        html += '</ul></div>';
        return html;
    };
    
    function updateResults(data) {
        document.getElementById('resultsSection').style.display = 'block';

        // The key mapping from Python JSON to HTML Card ID
        const cardMap = {
            'Normal': document.querySelector('#normalCard ul'),
            'Protan': document.querySelector('#protanopiaCard ul'),
            'Deutan': document.querySelector('#deuteranopiaCard ul'),
            'Tritan': document.querySelector('#tritanopiaCard ul')
        };
        
        const simulations = data.simulations;

        for (const [type, ul] of Object.entries(cardMap)) {
            const simData = simulations[type]; 
            
            if (!ul || !simData || !simData.analysis || !simData.color_metrics) {
                console.warn(`Skipping render for ${type}: Missing data or element.`);
                if (ul) ul.innerHTML = '<li style="color:orange;">Analysis data not available.</li>';
                continue;
            }

            const analysis = simData.analysis;
            const metrics = simData.color_metrics;
            const verdictClass = analysis.verdict ? analysis.verdict.replace(/\s/g, '') : 'Unknown';
            
            // Generate the list content based on the Flask structure
            ul.innerHTML = `
                <li class="result-verdict ${verdictClass}"><strong>Final Accessibility Rating:</strong> ${analysis.verdict}</li>
                <li><strong>Overall Accessibility Score:</strong> ${analysis.accessibility_percent}%</li>
                <li><strong>Technical Accessibility Score:</strong> ${metrics.classification_accuracy}%</li>
                <li><strong>Identified Segments:</strong> ${metrics.identified_segments.length}/7</li>
                ${formatMisclassifiedList(metrics.misclassified_segments)}
            `;
        }
    }
});
