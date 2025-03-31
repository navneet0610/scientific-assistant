document.addEventListener("DOMContentLoaded", function () {
    const searchForm = document.getElementById("searchForm");
    const resultsTable = document.getElementById("resultsTable");
    const tableBody = resultsTable.querySelector("tbody");
    const tableHead = resultsTable.querySelector("thead");
    const tableContainer = document.getElementById("tableContainer");
    const loadingScreen = document.getElementById("loadingScreen");
    const loadingText = document.getElementById("loadingText");
    const closeResultsBtn = document.getElementById("closeResults");

    // Hide elements initially
    loadingScreen.style.display = "none";
    tableContainer.classList.remove("visible");
    closeResultsBtn.classList.remove("visible");

    // Show loading animation
    function showLoadingAnimation() {
        loadingText.textContent = "MARUTI"; // Full fade effect
        loadingScreen.style.display = "flex";
    }

    function hideLoadingAnimation() {
        loadingScreen.style.display = "none";
    }

    // Handle search submission
    searchForm.addEventListener("submit", function (event) {
        event.preventDefault();

        const query = document.getElementById("searchQuery").value.trim();
        if (!query) {
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
    });

    // Close Results
    closeResultsBtn.addEventListener("click", function () {
        tableContainer.classList.remove("visible");
        closeResultsBtn.classList.remove("visible");
    });
});
