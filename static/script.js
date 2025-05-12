document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const imageInput = document.getElementById('imageInput');
    const previewImage = document.getElementById('previewImage');
    const resultImage = document.getElementById('resultImage');
    const loader = document.getElementById('loader');

    // Предпросмотр выбранного изображения
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
                resultImage.style.display = 'none'; // Скрываем предыдущий результат
            }
            reader.readAsDataURL(file);
        }
    });

    // Обработка отправки формы
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(uploadForm);
        loader.style.display = 'flex';

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }
            // Отображаем результат
            resultImage.src = data.result_url;
            resultImage.style.display = 'block';
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Произошла ошибка при обработке изображения');
        })
        .finally(() => {
            loader.style.display = 'none';
        });
    });
}); 