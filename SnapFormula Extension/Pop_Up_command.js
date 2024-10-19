document.getElementById('captureBtn').addEventListener('click', () => {
  chrome.tabs.captureVisibleTab(null, { format: 'png' }, function (dataUrl) {
    document.getElementById('screenshot').src = dataUrl;
  });
});
