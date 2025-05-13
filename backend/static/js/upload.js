document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.querySelector('input[type=\"file\"]');
    if (fileInput) {
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                alert('Image selected: ' + fileInput.files[0].name);
            }
        });
    }
});