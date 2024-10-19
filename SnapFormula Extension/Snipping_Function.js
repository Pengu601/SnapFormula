// Create the overlay for the snipping tool

let overlay = document.createElement('div');
overlay.style.position = 'fixed';
overlay.style.top = '0';
overlay.style.left = '0';
overlay.style.width = '100vw';
overlay.style.height = '100vh';
overlay.style.background = 'rgba(0, 0, 0, 0.5)';
overlay.style.zIndex = '9999';
overlay.style.cursor = 'crosshair';

if(document.body.contains(overlay)){
  document.body.removeChild(overlay);
}
document.body.appendChild(overlay);

// Variables to track the selection box
let startX, startY, endX, endY, selectionBox;
let isSelecting = false;

// Mouse down event to start selection
overlay.addEventListener('mousedown', (e) => {
  isSelecting = true;
  startX = e.clientX;
  startY = e.clientY;
  selectionBox = document.createElement('div');
  selectionBox.style.position = 'fixed';
  selectionBox.style.border = '2px dashed #fff';
  selectionBox.style.zIndex = '10000';
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
overlay.addEventListener('mouseup', async () => {
  isSelecting = false;

  // Remove the overlay and selection box
  document.body.removeChild(overlay);
  document.body.removeChild(selectionBox);

  // Prompt user for filename
  let filename = prompt("Enter a filename for the screenshot:", "screenshot");
  if (!filename || filename.trim() === "") {
    console.error("Invalid filename. Exiting without saving.");
    return; // Exit if no filename is provided
  }

  chrome.runtime.sendMessage({ action: "captureVisibleTab" }, (response) => {
    // if (response && response.dataUrl) {
    //   console.log(response.dataUrl); // Use the dataUrl (e.g., to download or display)
    // } else {
    //   console.error("Failed to capture visible tab.");
    // }
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
        console.log("Filename:", `${filename}.png`);

        // Create a download link and trigger the download
        let downloadLink = document.createElement('a');
        downloadLink.href = croppedImageUrl;
        downloadLink.download = `${filename}.png`; // Use user-provided filename
        downloadLink.click();
      }
    }
  });
  // Capture the visible tab

  
});
