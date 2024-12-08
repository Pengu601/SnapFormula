document.addEventListener("DOMContentLoaded", function() {
    const captureButton = document.getElementById('capture');
    
    if (captureButton) {
        captureButton.addEventListener('click', function() {
            chrome.tabs.captureVisibleTab(null, { format: "png" }, function(dataUrl) {
                chrome.runtime.sendMessage({ action: "saveScreenshot", dataUrl });
            });
        });
    } else {
        console.error("Element with ID 'capture' not found.");
    }
});
