from screenshot import takeScreenshot
from app import app
import tkinter as tk
import tkinter.ttk as ttk
def main():
    root = tk.Tk()
    app(root)
    # takeScreenshot()

if __name__ == "__main__":
    main()