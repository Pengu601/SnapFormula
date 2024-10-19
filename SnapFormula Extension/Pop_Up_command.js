
document.getElementById('activateSnippingTool').addEventListener('click', () => {
  console.log('Button clicked');
  chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
    chrome.scripting.executeScript({
      target: {tabId: tabs[0].id},
      files: ['Snipping_Function.js']
    });
  });
});

chrome.runtime.onInstalled.addListener(() => {
  console.log('Debugger Ready');
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if(request.action === "captureVisibleTab"){
    // Capture the visible tab
  
  }
})