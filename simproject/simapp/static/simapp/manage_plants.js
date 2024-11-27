document.addEventListener("DOMContentLoaded", function () {
    const saveBtn = document.getElementById("saveBtn");
    const deleteBtn = document.getElementById("deleteBtn");
    const plantForm = document.getElementById("plantForm");
    const clearBtn = document.getElementById("clear");
    const csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute('content');


    // Function to handle form submission via AJAX
    saveBtn.addEventListener("click", function () {
        const formData = new FormData(plantForm);

        fetch(managePlantsUrl, {
            method: "POST",
            body: formData,
            headers: {
                "X-CSRFToken": csrfToken,
            },
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Reload page without plant_id in URL after save
                window.history.pushState({}, document.title, window.location.pathname);
                window.location.reload();
            } else {
                console.error("Form errors:", data.errors);
                alert("Error saving plant: " + JSON.stringify(data.errors));
            }
        })
        .catch(error => console.error("Error:", error));
    });
    if (clearBtn) {
        clearBtn.addEventListener("click", function () {
            const url = new URL(window.location.href);
            url.search = "";  // Clear query parameters
            window.history.pushState({}, document.title, url.toString());
            window.location.reload();
        });
    }

    // Function to handle plant deletion via AJAX
    if (deleteBtn) {
        deleteBtn.addEventListener("click", function () {
            const formData = new FormData(plantForm);
            formData.append("delete", true);  // Indicate deletion action
            fetch(managePlantsUrl, {
                method: "POST",
                body: formData,
                headers: {
                    "X-CSRFToken": csrfToken,
                },
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Reload page without plant_id in URL after delete
                    window.history.pushState({}, document.title, window.location.pathname);
                    window.location.reload();
                } else {
                    alert("Error deleting plant.");
                }
            })
            .catch(error => console.error("Error:", error));
        });
    }
});  

