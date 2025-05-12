document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const imageInput = document.getElementById('imageInput');
    const previewImage = document.getElementById('previewImage');
    const resultImage = document.getElementById('resultImage');
    const loader = document.getElementById('loader');
    const lightButton = document.getElementById('lightButton');
    const proButton = document.getElementById('proButton');

    // Предпросмотр выбранного изображения
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
                resultImage.style.display = 'none'; // Скрываем предыдущий результат
                
                // Показываем кнопки после выбора изображения
                lightButton.style.display = 'inline-block';
                proButton.style.display = 'inline-block';
            }
            reader.readAsDataURL(file);
        }
    });

    // Функция для обработки изображения
    function processImage(mode) {
        if (!imageInput.files[0]) {
            alert('Пожалуйста, выберите изображение');
            return;
        }

        const formData = new FormData(uploadForm);
        formData.append('mode', mode); // Добавляем режим обработки
        loader.style.display = 'flex';

        // Обновляем текст загрузки в зависимости от режима
        const loaderText = loader.querySelector('p');
        loaderText.textContent = mode === 'light' ? 
            'Создаём раскраску...' : 
            'Создаём раскраску с помощью AI...';

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                if (data.error.includes('billing_hard_limit_reached')) {
                    alert('Достигнут лимит использования AI. Пожалуйста, попробуйте Light версию или повторите позже.');
                } else {
                    alert(data.error);
                }
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
    }

    // Обработчики для кнопок
    lightButton.addEventListener('click', function(e) {
        e.preventDefault();
        processImage('light');
    });

    proButton.addEventListener('click', function(e) {
        e.preventDefault();
        processImage('pro');
    });
}); 