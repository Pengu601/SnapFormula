// background.js
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "saveScreenshot") {
        saveScreenshot(message.dataUrl);
    }
});

function saveScreenshot(dataUrl) {
    console.log("Attempting to download screenshot...");

    chrome.downloads.download({
        url: dataUrl,
        filename: 'screenshot_' + Date.now() + '.png',
        saveAs: false
    }, function(downloadId) {
        if (chrome.runtime.lastError) {
            console.error("Download failed:", chrome.runtime.lastError.message);
        } else {
            console.log("Screenshot saved, download ID:", downloadId);
        }
    });
}
