document.addEventListener("DOMContentLoaded", function () {
    const searchForm = document.getElementById("searchForm");
    const searchInput = document.getElementById("searchQuery");
    const resultsTable = document.getElementById("resultsTable");
    const tableBody = resultsTable.querySelector("tbody");
    const tableHead = resultsTable.querySelector("thead");
    const tableContainer = document.getElementById("tableContainer");
    const loadingScreen = document.getElementById("loadingScreen");
    const closeResultsBtn = document.getElementById("closeResults");
    const imageInput = document.getElementById("imageInput");
    const recordButton = document.getElementById("recordAudio");
    const recordingStatus = document.getElementById("recordingStatus");

    // Hide elements initially
    loadingScreen.style.display = "none";
    tableContainer.classList.remove("visible");
    closeResultsBtn.classList.remove("visible");

    let mediaRecorder;
    let audioChunks = [];

    // Show loading animation
    function showLoadingAnimation() {
        loadingScreen.style.display = "flex";
    }

    function hideLoadingAnimation() {
        loadingScreen.style.display = "none";
    }

    // Function to perform search with extracted text
    function performSearch(query) {
        if (!query.trim()) {
            alert("Please enter a search query.");
            return;
        }

        showLoadingAnimation();

        fetch(`/search/?q=${encodeURIComponent(query)}`)
            .then(response => response.json())
            .then(data => {
                hideLoadingAnimation();
                tableBody.innerHTML = "";

                if (data.results && data.results.length > 0) {
                    tableContainer.classList.add("visible");
                    tableHead.classList.add("visible");
                    closeResultsBtn.classList.add("visible");

                    data.results.forEach(result => {
                        const row = document.createElement("tr");
                        row.innerHTML = `
                            <td>${result.title}</td>
                            <td>${result.authors}</td>
                            <td>${result.year}</td>
                            <td>${result.category}</td>
                            <td>
                                <span class="abstract">
                                    ${result.abstract.slice(0, 100)}...
                                    <span class="abstract-popup">${result.abstract}</span>
                                </span>
                            </td>
                            <td><a href="${result.pdf_url}" target="_blank">PDF</a></td>
                        `;
                        tableBody.appendChild(row);
                    });
                } else {
                    tableContainer.classList.remove("visible");
                    closeResultsBtn.classList.remove("visible");
                }
            })
            .catch(error => {
                hideLoadingAnimation();
                console.error("Error fetching search results:", error);
            });
    }

    // Handle text-based search submission
    searchForm.addEventListener("submit", function (event) {
        event.preventDefault();
        performSearch(searchInput.value);
    });

    // Handle Image Upload and Extract Text Before Searching
    imageInput.addEventListener("change", function () {
        const file = imageInput.files[0];
        if (file) {
            const formData = new FormData();
            formData.append("image", file);

            fetch("/search/process_image/", { method: "POST", body: formData })
                .then(response => response.json())
                .then(data => {
                    if (data.extracted_text && data.extracted_text.trim()) {
                        searchInput.value = data.extracted_text.trim();
                        performSearch(data.extracted_text);
                    } else {
                        alert("No text could be extracted from the image.");
                    }
                })
                .catch(error => console.error("Image processing failed:", error));
        }
    });

    // Handle Audio Recording and Extract Text Before Searching
    recordButton.addEventListener("click", async () => {
        if (!mediaRecorder) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);

                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
                    audioChunks = [];

                    const formData = new FormData();
                    formData.append("audio", audioBlob);

                    fetch("/search/process_audio/", { method: "POST", body: formData })
                        .then(response => response.json())
                        .then(data => {
                            if (data.transcribed_text && data.transcribed_text.trim()) {
                                searchInput.value = data.transcribed_text.trim();
                                performSearch(data.transcribed_text);
                            } else {
                                alert("No text could be extracted from the audio.");
                            }
                        })
                        .catch(error => console.error("Audio processing failed:", error));
                };

                mediaRecorder.start();
                recordingStatus.innerText = "Recording... (Click Again to Stop)";
            } catch (error) {
                console.error("Error accessing microphone:", error);
            }
        } else {
            mediaRecorder.stop();
            mediaRecorder = null;
            recordingStatus.innerText = "";
        }
    });

    // Close Results and Clear Search Bar
    closeResultsBtn.addEventListener("click", function () {
        tableContainer.classList.remove("visible");
        closeResultsBtn.classList.remove("visible");
        searchInput.value = ""; // Reset search input
    });
});
