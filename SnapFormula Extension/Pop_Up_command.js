
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

// chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
//   if(request.action === "captureVisibleTab"){
//     // Capture the visible tab
//     chrome.tabs.captureVisibleTab(null, {format: 'png'}, function (dataUrl) {
//       let img = new Image();
//       img.src = dataUrl;
//       img.onload = function () {
//         // Create a canvas to crop the screenshot
//         let canvas = document.createElement('canvas');
//         canvas.width = Math.abs(endX - startX);
//         canvas.height = Math.abs(endY - startY);
//         let ctx = canvas.getContext('2d');

//         // Calculate the crop position and size
//         let sx = Math.min(startX, endX);
//         let sy = Math.min(startY, endY);
//         let sw = Math.abs(endX - startX);
//         let sh = Math.abs(endY - startY);

//         // Draw the cropped portion on the canvas
//         ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);

//         if (canvas.width > 0 && canvas.height > 0) {
//           // Convert the canvas to a data URL (image format)
//           let croppedImageUrl = canvas.toDataURL('image/png');
//           console.log("Data URL:", croppedImageUrl);  // Debugging log
//           console.log("Filename:", `${filename}.png`);

//           // Create a download link and trigger the download
//           let downloadLink = document.createElement('a');
//           downloadLink.href = croppedImageUrl;
//           downloadLink.download = `${filename}.png`; // Use user-provided filename

//           document.body.appendChild(downloadLink); // Append to body to ensure it's in the DOM
//           downloadLink.click(); // Trigger the download

//           // Clean up by removing the link from the DOM
//           document.body.removeChild(downloadLink);

          
//         } else {
//           console.error("Invalid selection area for the screenshot.");
//         }

//       };
//     });
//   }
// })