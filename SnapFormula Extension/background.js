chrome.runtime.onInstalled.addListener(() => {
  console.log('Screen Capture Extension Installed');
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if(request.action === "captureVisibleTab"){
    // Capture the visible tab
    console.log('test');
    chrome.tabs.captureVisibleTab(null, {format: 'png'}, function (dataUrl) {
      console.log(dataUrl);
      sendResponse({dataUrl});
    });
  }
  return true;
  
});

chrome.commands.onCommand.addListener((command, sender, sendResponse) => {
  if(command === "takeScreenshot"){
    chrome.runtime.sendMessage({ action: "doScreenshot" }, (response) => {
      
    });  
  }
});