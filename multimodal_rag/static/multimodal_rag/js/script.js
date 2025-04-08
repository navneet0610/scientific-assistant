function startSearch() {
    const query = document.getElementById('searchQuery').value;
    const imageFile = document.getElementById('file').files[0]; // Get the image file from the input

    // If both image and text are empty, return early
    if (!query && !imageFile) {
        alert('Please provide either a query or an image.');
        return;
    }

    // Show the loading indicator
    document.getElementById('loadingIndicator').style.display = 'block';

    // Hide the results if visible
    document.getElementById('results').innerHTML = '';

    // Create a FormData object to send both the query and the image (if any)
    const formData = new FormData();

    // If there is a query, add it to the FormData
    if (query) {
        formData.append('query', query);
    }

    // If an image is selected, add it to the FormData
    if (imageFile) {
        formData.append('image', imageFile);
    }

    // Add is_image flag to indicate whether it's an image search or not
    formData.append('is_image', imageFile ? 'true' : 'false');

    // Make POST request to the server using FormData (multipart/form-data)
    fetch('/multimodal/search/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Hide the loading indicator
        document.getElementById('loadingIndicator').style.display = 'none';

        if (data.error) {
            alert(data.error);
        } else {
            displayResults(data);
        }
    })
    .catch(error => {
        // Hide the loading indicator
        document.getElementById('loadingIndicator').style.display = 'none';
        alert('Error fetching results: ' + error.message);
    });
}

function refineText(text) {
    let refinedText = text.replace(/\\^([a-zA-Z0-9]+)/g, "<sup>$1</sup>");
    refinedText = refinedText.replace(/\\Gamma/g, "Γ");
    refinedText = refinedText.replace(/\\lambda/g, "λ");
    refinedText = refinedText.replace(/\\alpha/g, "α");
    refinedText = refinedText.replace(/\\epsilon/g, "ε");
    refinedText = refinedText.replace(/&/g, "&;"); // Convert & to &amp;
    refinedText = refinedText.replace(/</g, "&lt;");  // Convert < to &lt;
    refinedText = refinedText.replace(/>/g, "&gt;");  // Convert > to &gt;
    refinedText = refinedText.replace(/\\n/g, "<br>");
    refinedText = refinedText.replace(/\\t/g, " ");  // Tab replaced with HTML space (emsp)
    refinedText = refinedText.replace(/\\&/g, "&;");
    return refinedText;
}


// Display Search Results Dynamically
function displayResults(results) {
    const resultsContainer = document.getElementById('results');

    results.forEach(item => {
        const resultTile = document.createElement('div');
        resultTile.classList.add('result-tile');

        let imagesHtml = '';
        item.images.forEach(image => {
          // If a caption exists, show it, otherwise use a default message "No caption"
          let captionText = image.caption || 'No caption';
          imagesHtml += `
          <div class="image-container">
            <a href="/static/images/${image.image_name}" target="_blank">
                <img src="/static/images/${image.image_name}" alt="${captionText}" />
            </a>
            <div class="caption">${captionText}</div>
          </div>
    `;
});

       resultTile.innerHTML = `
            <div class="result-details">
                <div class="result-title">${refineText(item.title)}</div>
                <div class="result-abstract">${refineText(item.abstract)}</div>
                <div class="result-caption">Image Description: ${refineText(item.caption)}</div>
                <div class="result-citation">Citation Count: ${item.citationCount}</div>
                <div class="result-pdf">
                    <a href="https://arxiv.org/pdf/${item.arxiv_id}.pdf" target="_blank">Download PDF</a>
                </div>
            </div>
            <div class="result-images">${imagesHtml}</div>
        `;
        resultsContainer.appendChild(resultTile);

    });
}