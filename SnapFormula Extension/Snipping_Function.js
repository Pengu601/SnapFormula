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
  

  // 
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

            // Show a popup asking what the user wants to do
            showPopup(croppedImageUrl);
        }
    };
});

// Function to show the popup with options
function showPopup(croppedImageUrl) {
    // Create the popup container
    let popup = document.createElement('div');
    popup.style.position = 'fixed';
    popup.style.top = '50%';
    popup.style.left = '50%';
    popup.style.transform = 'translate(-50%, -50%)';
    popup.style.backgroundColor = '#fff';
    popup.style.padding = '20px';
    popup.style.borderRadius = '10px';
    popup.style.boxShadow = '0px 0px 10px rgba(0,0,0,0.5)';
    popup.style.zIndex = '9999';

    // Add text to the popup
    let popupText = document.createElement('p');
    popupText.innerText = 'What would you like to do?';
    popupText.style.color = 'black'; // Set text color to black
    popupText.style.fontFamily = 'Arial, sans-serif'; // Optional: Change font style
    popup.appendChild(popupText);

    // Create buttons with blue background and white text
    let createButton = (text, onClick) => {
        let button = document.createElement('button');
        button.innerText = text;
        button.style.backgroundColor = '#007BFF'; // Blue background
        button.style.color = '#fff'; // White text
        button.style.border = 'none';
        button.style.padding = '10px 20px';
        button.style.margin = '10px';
        button.style.borderRadius = '5px';
        button.style.cursor = 'pointer';
        button.style.fontFamily = 'Arial, sans-serif'; // Optional: Change font style
        button.style.fontSize = '14px'; // Optional: Set font size
        button.onclick = onClick;
        button.onmouseenter = () => button.style.backgroundColor = '#0056b3'; // Hover effect
        button.onmouseleave = () => button.style.backgroundColor = '#007BFF'; // Return to original color
        return button;
    };

    let downloadButton = createButton('Download Image', function () {
        downloadImage(croppedImageUrl);  // Call the download function
        document.body.removeChild(popup);  // Remove the popup after selection
    });

    let saveTextButton = createButton('Save Image Text', function () {
        saveImageText();  // Placeholder function for saving image text
        document.body.removeChild(popup);
    });

    let bothButton = createButton('Both', function () {
        saveImageText();  // Placeholder function for saving image text
        downloadImage(croppedImageUrl);  // Download the image
        document.body.removeChild(popup);
    });

    // Add buttons to the popup
    popup.appendChild(downloadButton);
    popup.appendChild(saveTextButton);
    popup.appendChild(bothButton);

    // Add the popup to the document
    document.body.appendChild(popup);
}

// Function to download the image
function downloadImage(croppedImageUrl) {
    // Create a download link and trigger the download
    let downloadLink = document.createElement('a');
    downloadLink.href = croppedImageUrl;
    downloadLink.download = `image.png`;  // Use user-provided filename
    downloadLink.click();  // Triggers download link click, downloading the file
}

// Placeholder function to save image text
function saveImageText() {
    console.log("Saving image text (not implemented yet).");
}
  // Capture the visible tab
});


