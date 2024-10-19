document.addEventListener("DOMContentLoaded", function() {
    // First, ensure the element exists
    const drawingBox = document.getElementById("drawing-box");
    console.log("Drawing box element:", drawingBox);
  
    if (drawingBox) {
      drawingBox.addEventListener("click", function() {
        console.log("Drawing box clicked. Attempting to capture screenshot.");
  
        chrome.tabs.captureVisibleTab(null, {}, function(imageUrl) {
          if (chrome.runtime.lastError) {
            console.error("Error capturing tab:", chrome.runtime.lastError);
            return;
          }
  
          console.log("Screenshot captured, URL:", imageUrl);
  
          chrome.downloads.download({
            url: imageUrl,
            filename: 'screenshot.png'
          }, function(downloadId) {
            if (chrome.runtime.lastError) {
              console.error("Error during download:", chrome.runtime.lastError);
            } else {
              console.log("Download started with ID:", downloadId);
            }
          });
        });
      });
    } else {
      console.error("Drawing box element not found.");
    }
  });
  