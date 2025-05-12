document.addEventListener('DOMContentLoaded', () => {
    const imageInput = document.getElementById('imageInput');
    const previewImage = document.getElementById('previewImage');
    const loader = document.getElementById('loader');
    const uploadForm = document.querySelector('.upload-form');
    const generateForm = document.querySelector('.generate-form');

    // Предпросмотр загруженного изображения
    imageInput.addEventListener('change', () => {
        const file = imageInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });

    // Обработка отправки формы загрузки
    uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        loader.style.display = 'flex';
        
        const formData = new FormData(uploadForm);
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(data => {
            alert(data);
            loader.style.display = 'none';
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Произошла ошибка при загрузке файла');
            loader.style.display = 'none';
        });
    });

    // Обработка отправки формы генерации
    generateForm.addEventListener('submit', (e) => {
        e.preventDefault();
        loader.style.display = 'flex';
        
        const formData = new FormData(generateForm);
        fetch('/generate', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(data => {
            alert(data);
            loader.style.display = 'none';
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Произошла ошибка при генерации раскраски');
            loader.style.display = 'none';
        });
    });
}); 