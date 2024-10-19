// Create the overlay for the snipping tool
var overlay;
overlay = document.createElement('div');
overlay.id ='screenshot-overlay'
overlay.style.position = 'fixed';
overlay.style.top = '0';
overlay.style.left = '0';
overlay.style.width = '100vw';
overlay.style.height = '100vh';
overlay.style.background = 'rgba(0, 0, 0, 0.5)';
overlay.style.zIndex = '9998';

document.body.appendChild(overlay);

// Variables to track the selection box
var startX, startY, endX, endY, selectionBox;
var isSelecting = false;

// Mouse down event to start selection
overlay.addEventListener('mousedown', (e) => {
  isSelecting = true;
  startX = e.clientX;
  startY = e.clientY;
  selectionBox = document.createElement('div');
  selectionBox.style.position = 'fixed';
  selectionBox.style.border = '2px solid #fff';
  selectionBox.style.zIndex = '9999';
  document.body.appendChild(selectionBox);
});

// Mouse move event to resize selection box
overlay.addEventListener('mousemove', (e) => {
  if (!isSelecting) return;
  endX = e.clientX;
  endY = e.clientY;
  selectionBox.style.left = `${Math.min(startX, endX)}px`;
  selectionBox.style.top = `${Math.min(startY, endY)}px`;
  selectionBox.style.width = `${Math.abs(endX - startX)}px`;
  selectionBox.style.height = `${Math.abs(endY - startY)}px`;
});


// Mouse up event to finalize selection and trigger download
document.addEventListener('mouseup', (e) => {
  
  e.stopPropagation();
  
  isSelecting = false;

  // Remove the overlay and selection box
  document.body.removeChild(selectionBox);
  document.body.removeChild(overlay);
  

  // Prompt user for filename
  

  chrome.runtime.sendMessage({ action: "captureVisibleTab" }, (response) => {
    const img = new Image();
    img.src = response.dataUrl;
    img.onload = function () {
      // Create a canvas to crop the screenshot
      let canvas = document.createElement('canvas');
      canvas.width = Math.abs(endX - startX);
      canvas.height = Math.abs(endY - startY);
      let ctx = canvas.getContext('2d');

      // Calculate the crop position and size
      let sx = Math.min(startX, endX);
      let sy = Math.min(startY, endY);
      let sw = Math.abs(endX - startX);
      let sh = Math.abs(endY - startY);

      // Draw the cropped portion on the canvas
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);

      if (canvas.width > 0 && canvas.height > 0) {
        // Convert the canvas to a data URL (image format)
        let croppedImageUrl = canvas.toDataURL('image/png');
        console.log("Data URL:", croppedImageUrl);  // Debugging log
        console.log("Filename:", `image.png`);

        // Create a download link and trigger the download
        let downloadLink = document.createElement('a');
        downloadLink.href = croppedImageUrl;
        downloadLink.download = `image.png`; // Use user-provided filename
        downloadLink.click(); //Triggers download link click, downloading file for the region capture
      }
    }
  });
  // Capture the visible tab

  
});

