chrome.action.onClicked.addListener((tab) => {
    // When the user clicks the extension icon, capture the visible tab
    chrome.tabs.captureVisibleTab(null, {format: "png"}, function(dataUrl) {
        if (chrome.runtime.lastError) {
            console.error(chrome.runtime.lastError.message);
        } else {
            saveScreenshot(dataUrl);
        }
    });
});

function saveScreenshot(dataUrl) {
    chrome.downloads.download({
        url: dataUrl,
        filename: 'screenshot_' + Date.now() + '.png',  // Add timestamp to avoid overwriting
        saveAs: false  // Set to false to save it directly without prompting
    }, function(downloadId) {
        if (chrome.runtime.lastError) {
            console.error("Download failed: " + chrome.runtime.lastError.message);
        } else {
            console.log("Screenshot saved to Downloads folder with download ID:", downloadId);
        }
    });
}

