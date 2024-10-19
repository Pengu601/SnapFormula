window.addEventListener('DOMContentLoaded', (event) => {
  document?.getElementById('activateSnippingTool').addEventListener('click', () => {
    console.log('Button clicked');
    chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
      chrome.scripting.executeScript({
        target: {tabId: tabs[0].id},
        files: ['Snipping_Function.js']
      });
    });
  });
});  

