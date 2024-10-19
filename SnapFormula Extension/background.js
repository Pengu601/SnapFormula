
chrome.runtime.onInstalled.addListener(() => {
  console.log('Screen Capture Extension Installed');
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if(request.action === "captureVisibleTab"){
    // Capture the visible tab
    console.log('test');
    chrome.tabs.captureVisibleTab(null, {format: 'png'}, function (dataUrl) {
      console.log(dataUrl);
      sendResponse({dataUrl}); //sends the response containg the encoded dataUrl for the screencapture to the snipping_function file
    });
  }
  return true; //makes listener asynchronous
  
});

chrome.commands.onCommand.addListener((command) => {
  if(command === "takeScreenshot"){
    
    chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
      chrome.scripting.executeScript({
        target: {tabId: tabs[0].id},
        files: ['Snipping_Function.js']
      });
    });
  }
});
