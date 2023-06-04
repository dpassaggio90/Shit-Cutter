import os
import subprocess
from tkinter import Tk, Button, Label, filedialog
from PIL import ImageTk, Image
import urllib.request
import cv2
import numpy as np

input_folder_path = ""
output_folder_path = ""
selected_files = []

def select_input_files():
    global selected_files
    selected_files = filedialog.askopenfilenames(filetypes=[("File video", "*.mp4")])
    input_files_label.config(text="\n".join(selected_files))

def select_output_folder():
    global output_folder_path
    output_folder_path = filedialog.askdirectory()
    output_folder_label.config(text=output_folder_path)

def stabilize_video(input_file, output_file):
    cap = cv2.VideoCapture(input_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    prev_frame = None
    prev_pts = None
    prev_good_pts = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, frame_gray, prev_pts, None)
            
            good_pts = pts[status == 1]
            prev_good_pts = prev_good_pts[status == 1]
            
            transform, _ = cv2.estimateAffinePartial2D(prev_good_pts, good_pts, method=cv2.RANSAC)
            
            if transform is not None:
                transform = transform.astype(np.float32)  # Convert the transformation matrix to float32
                
                stabilized_frame = cv2.warpAffine(frame, transform, (width, height))
                out.write(stabilized_frame)
            else:
                out.write(frame)
        else:
            out.write(frame)
        
        prev_frame = frame_gray
        prev_pts = cv2.goodFeaturesToTrack(prev_frame, maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)
        prev_good_pts = prev_pts
    
    cap.release()
    out.release()

def process_videos():
    if selected_files and output_folder_path:
        for input_file in selected_files:
            filename = os.path.basename(input_file)
            output_file = os.path.join(output_folder_path, filename)
            stabilize_video(input_file, output_file)
        output_label.config(text="Elaborazione completata.")
    else:
        output_label.config(text="Seleziona file di input e cartella di output.")

# Create the window
window = Tk()
window.title("Taglia Clip di merda")
window.geometry("400x400")

# Download the image and save it locally
image_url = "https://i.ibb.co/HrQQP4m/Senza-titolo-1.png"
image_filename = "image.png"
urllib.request.urlretrieve(image_url, image_filename)

# Load the image
image = Image.open(image_filename)
image = image.resize((400, 200), Image.LANCZOS)  # Resize the image to fit the window
photo = ImageTk.PhotoImage(image)

# Create a label to display the image
image_label = Label(window, image=photo)
image_label.pack()

# Labels
Label(window, text="File di input selezionati:").pack()
input_files_label = Label(window, text="")
input_files_label.pack()
Label(window, text="Cartella di output:").pack()
output_folder_label = Label(window, text="")
output_folder_label.pack()
output_label = Label(window, text="")
output_label.pack()

# Buttons
Button(window, text="Seleziona file di input", command=select_input_files).pack()
Button(window, text="Seleziona cartella di output", command=select_output_folder).pack()
Button(window, text="Elabora video", command=process_videos).pack()

# Start the window's event loop
window.mainloop()
