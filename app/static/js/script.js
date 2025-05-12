document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('image-input');
    const uploadButton = document.getElementById('upload-button');
    const progress = document.getElementById('progress');
    const error = document.getElementById('error');
    const result = document.getElementById('result');
    const resultImage = document.getElementById('result-image');
    const downloadLink = document.getElementById('download-link');
    const newImageButton = document.getElementById('new-image');

    // Handle drag and drop
    const dropZone = document.querySelector('.file-input-wrapper');
    const previewContainer = document.createElement('div');
    previewContainer.className = 'preview-container hidden';
    dropZone.appendChild(previewContainer);
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropZone.classList.add('highlight');
    }

    function unhighlight() {
        dropZone.classList.remove('highlight');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        updateFileName();
        showPreview(files[0]);
    }

    // Update file name display and show preview
    fileInput.addEventListener('change', (e) => {
        updateFileName();
        showPreview(e.target.files[0]);
    });

    function updateFileName() {
        const fileName = fileInput.files[0]?.name;
        const label = document.querySelector('.custom-file-input');
        label.textContent = fileName || 'Выберите изображение или перетащите его сюда';
    }

    function showPreview(file) {
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            previewContainer.innerHTML = `
                <div class="preview-image-container">
                    <img src="${e.target.result}" alt="Предпросмотр" class="preview-image">
                </div>
            `;
            previewContainer.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    // Handle form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const file = fileInput.files[0];
        if (!file) {
            showError('Пожалуйста, выберите изображение.');
            return;
        }

        // Check file type
        const fileType = file.type;
        if (!['image/jpeg', 'image/png'].includes(fileType)) {
            showError('Пожалуйста, выберите изображение в формате JPEG или PNG.');
            return;
        }

        // Check file size (4MB max)
        if (file.size > 4 * 1024 * 1024) {
            showError('Размер файла не должен превышать 4МБ.');
            return;
        }

        // Prepare form data
        const formData = new FormData(form);
        
        try {
            // Show progress and hide other elements
            progress.classList.remove('hidden');
            error.classList.add('hidden');
            result.classList.add('hidden');
            uploadButton.disabled = true;

            // Send request
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Не удалось обработать изображение');
            }

            // Show result
            resultImage.src = data.result_thumbnail;
            downloadLink.href = data.result_url;
            result.classList.remove('hidden');
            form.classList.add('hidden');

        } catch (err) {
            showError(err.message);
        } finally {
            progress.classList.add('hidden');
            uploadButton.disabled = false;
        }
    });

    // Handle "Create Another" button
    newImageButton.addEventListener('click', () => {
        form.reset();
        form.classList.remove('hidden');
        result.classList.add('hidden');
        error.classList.add('hidden');
        previewContainer.classList.add('hidden');
        updateFileName();
    });

    function showError(message) {
        error.textContent = message;
        error.classList.remove('hidden');
    }
}); 