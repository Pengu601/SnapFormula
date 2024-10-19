chrome.runtime.onMessage.addEventListener((request, sender, sendResponse) => {
  if (request.type === 'downloadImage') {
    chrome.downloads.download({
      url: request.dataUrl,
      filename: request.filename
    }, (downloadId) => {
      if (chrome.runtime.lastError) {
        console.error(chrome.runtime.lastError.message);
      } else {
        console.log(`Download started with ID: ${downloadId}`);
      }
    });
  }
});
