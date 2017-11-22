import Tkinter as tk
import tkFileDialog as filedialog

def fileDialog():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    string='.wav'
    if file_path.find(string, len(file_path)-5, len(file_path)) == -1:
        print "Not a .wav file, Openning Default file"
        file_path='sc03_16m.wav'
    return file_path
