import tkinter as tk
import tkinter.ttk as ttk
import sv_ttk
from screenshot import takeScreenshot
def app(win):
    
    win.title("SnapFormula")
    
    win.geometry("500x500")
    button = ttk.Button(win, text="test",command = lambda: takeScreenshot(win) )#command = takeScreenshot(root)
    button.pack()
    
    sv_ttk.use_dark_theme()
    
    win.mainloop()    
