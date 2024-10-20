// // Create the overlay
// let overlay = document.createElement('div');
// overlay.style.position = 'fixed';
// overlay.style.top = '0';
// overlay.style.left = '0';
// overlay.style.width = '100vw';
// overlay.style.height = '100vh';
// overlay.style.background = 'rgba(0, 0, 0, 0.5)';
// overlay.style.zIndex = '9999';
// overlay.style.cursor = 'crosshair';
// document.body.appendChild(overlay);
// `
// // Variables to track the selection box
// let startX, startY, endX, endY, selectionBox;
// let isSelecting = false;

// // Mouse down event to start selection
// overlay.addEventListener('mousedown', (e) => {
//   isSelecting = true;
//   startX = e.clientX;
//   startY = e.clientY;
//   selectionBox = document.createElement('div');
//   selectionBox.style.position = 'fixed';
//   selectionBox.style.border = '2px dashed #fff';
//   selectionBox.style.zIndex = '10000';
//   document.body.appendChild(selectionBox);
// });

// // Mouse move event to resize selection box
// overlay.addEventListener('mousemove', (e) => {
//   if (!isSelecting) return;
//   endX = e.clientX;
//   endY = e.clientY;
//   selectionBox.style.left = `${Math.min(startX, endX)}px`;
//   selectionBox.style.top = `${Math.min(startY, endY)}px`;
//   selectionBox.style.width = `${Math.abs(endX - startX)}px`;
//   selectionBox.style.height = `${Math.abs(endY - startY)}px`;
// });

// // Mouse up event to finalize selection
// overlay.addEventListener('mouseup', async () => {
//   isSelecting = false;

//   // Remove the overlay and selection box
//   document.body.removeChild(overlay);
//   document.body.removeChild(selectionBox);

//   // Capture the visible tab
//   chrome.tabs.captureVisibleTab(null, {format: 'png'}, function (dataUrl) {
//     let img = new Image();
//     img.src = dataUrl;
//     img.onload = function () {
//       // Create a canvas to crop the screenshot
//       let canvas = document.createElement('canvas');
//       canvas.width = Math.abs(endX - startX);
//       canvas.height = Math.abs(endY - startY);
//       let ctx = canvas.getContext('2d');

//       // Calculate the crop position and size
//       let sx = Math.min(startX, endX);
//       let sy = Math.min(startY, endY);
//       let sw = Math.abs(endX - startX);
//       let sh = Math.abs(endY - startY);

//       // Draw the cropped portion on the canvas
//       ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);

//       // Display the cropped screenshot (or save it)
//       let croppedImageUrl = canvas.toDataURL();
//       let screenshotWindow = window.open();
//       screenshotWindow.document.write('<img src="' + croppedImageUrl + '"/>');
//     };
//   });
// });
