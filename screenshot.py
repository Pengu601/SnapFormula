import tkinter as tk
from PIL import ImageGrab
import io
import win32clipboard
 # Global variables for rectangle drawing
rect = None
start_x = None
start_y = None

#creates global root

# Function to start capturing the region
def on_button_press(event):
    global start_x, start_y
    start_x = event.x
    start_y = event.y

# Function to update the drawn rectangle while dragging
def on_move_press(event, canvas):
    global rect
    canvas.delete(rect) #deletes current rectangle
    curX, curY = (event.x, event.y) #gets current x and y coordinates
    rect = canvas.create_rectangle(start_x, start_y, curX, curY, outline='white', width=2, fill="maroon3") #creates new selection rectangle

# Function to complete the selection and take a screenshot of the selected region
def on_button_release(event, root):
    global rect, start_x, start_y
    end_x, end_y = (event.x, event.y) #gets ending x and y coordinates

    # Get absolute coordinates on the screen
    x1 = root.winfo_rootx() + start_x
    y1 = root.winfo_rooty() + start_y
    x2 = root.winfo_rootx() + end_x
    y2 = root.winfo_rooty() + end_y

    #Calculate the region as left, top width, and height respectively
    
    
    # Take a screenshot of the selected area
    img = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    
    # img.save("capture.png")
    
    sendToClipbaord(img) #save to clipboard
    
    # Close the window after capturing the region
    root.quit()

def sendToClipbaord(img):
    output = io.BytesIO()
    img.convert("RGB").save(output, format="BMP") #converts it to BMP format which can be stored to clipboard
    data = output.getvalue()[14:] #remove first 14 bytes which is for the name of the image

    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
    win32clipboard.CloseClipboard()

def takeScreenshot(root):
    
    # Create the main window
    
    root.attributes("-fullscreen", True, "-transparent", "maroon3")  # Fullscreen for better area selection
    root.attributes("-alpha", .3) # makes the solid color transparent so you can see the screen
    root.lift()
    root.attributes("-topmost", True)
    
    # Create a canvas where the selection rectangle will be drawn
    canvas = tk.Canvas(root, cursor="cross", bg="grey3")
    canvas.pack(fill=tk.BOTH, expand=True)

   
    # Bind mouse events for region selection
    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", lambda event: on_move_press(event, canvas))
    canvas.bind("<ButtonRelease-1>", lambda event: on_button_release(event, root))


    # Start the Tkinter event loop
    root.mainloop()