# -*- coding: utf-8 -*-
"""
Created on Tue May  3 17:25:28 2022

@author: voer874
"""

from tkinter import Tk, Text, TOP, BOTH, X, N, LEFT, RIGHT, filedialog, messagebox
from tkinter.ttk import Frame, Label, Entry, Button
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
import glob
import zipfile
import pickle
import time
import moviepy.editor as mpy
import shutil
from matplotlib.widgets import SpanSelector

# Good habit to put your GUI in a class to make it self-contained
class SimpleDialog(Frame):

    def __init__(self):
        super().__init__()
        # self allow the variable to be used anywhere in the class
        self.output1 = ''
        self.output2 = ''
        self.output3 = ''
        self.output4 = ''
        self.output5 = ''
        self.output6 = ''
        self.output7 = ''
        self.output8 = ''
        self.initUI()

    def initUI(self):

        self.master.title("Gif creator")
        self.pack(fill=BOTH, expand=True)

        frame1 = Frame(self)
        frame1.pack(fill=X)

        lbl1 = Label(frame1, text="Select ion or atom (Ex. Fe1 O1 or Fe)", width=32)
        lbl1.pack(side=LEFT, padx=5, pady=10)

        self.entry1 = Entry(frame1, textvariable=self.output1)
        self.entry1.pack(fill=X, padx=5, expand=True)

        frame2 = Frame(self)
        frame2.pack(fill=X)

        lbl2 = Label(frame2, text="Time per frame (min)", width=32)
        lbl2.pack(side=LEFT, padx=5, pady=10)

        self.entry2 = Entry(frame2)
        self.entry2.pack(fill=X, padx=5, expand=True)

        frame3 = Frame(self)
        frame3.pack(fill=X)

        lbl3 = Label(frame3, text="Upper Threshold", width=32)
        lbl3.pack(side=LEFT, padx=5, pady=10)

        self.entry3 = Entry(frame3)
        self.entry3.pack(fill=X, padx=5, expand=True)
        
        frame8 = Frame(self)
        frame8.pack(fill=X)

        lbl8 = Label(frame8, text="Lower Threshold", width=32)
        lbl8.pack(side=LEFT, padx=5, pady=10)

        self.entry8 = Entry(frame8)
        self.entry8.pack(fill=X, padx=5, expand=True)

        frame4 = Frame(self)
        frame4.pack(fill=X)

        lbl4 = Label(frame4, text="Resolution square size (Å)", width=32)
        lbl4.pack(side=LEFT, padx=5, pady=10)

        self.entry4 = Entry(frame4)
        self.entry4.pack(fill=X, padx=5, expand=True)

        frame5 = Frame(self)
        frame5.pack(fill=X)

        lbl5 = Label(frame5, text="Name your gif", width=32)
        lbl5.pack(side=LEFT, padx=5, pady=10)
        
        self.entry5 = Entry(frame5)
        self.entry5.pack(fill=X, padx=5, expand=True)
        
        frame6 = Frame(self)
        frame6.pack(fill=X)

        btn1 = Button(frame6, text="Choose pkl", command=self.get_file)
        btn1.pack(padx=5, pady=10)
        
        frame7 = Frame(self)
        frame7.pack(fill=X)
        
        btn2 = Button(frame7, text="Choose save folder", command=self.save_folder)
        btn2.pack(padx=5, pady=10)
        
        frame8 = Frame(self)
        frame8.pack(fill=X)
        
        # Command tells the form what to do when the button is clicked
        btn = Button(frame8, text="Submit", command=self.onSubmit)
        btn.pack(padx=5, pady=10)

    def onSubmit(self):

        self.output1 = self.entry1.get()
        self.output2 = self.entry2.get()
        self.output3 = self.entry3.get()
        self.output4 = self.entry4.get()
        self.output5 = self.entry5.get()
        self.output8 = self.entry8.get()
        self.quit()
        
    def get_file(self):
        
        self.output6 = filedialog.askopenfilename(parent=self)
        
        messagebox.showinfo(
        title='Selected File',
        message=self.output6
        )
        
    def save_folder(self):
        
        self.output7 = filedialog.askdirectory(parent=self)
        messagebox.showinfo(
        title='Selected Folder',
        message=self.output7
        )

def main():

    # This part triggers the dialog
    root = Tk()
    root.geometry("500x400")
    app = SimpleDialog()
    root.mainloop()
    # Here we can act on the form components or
    # better yet, copy the output to a new variable
    user_input = (app.output1, app.output2, app.output3, app.output4, app.output5, app.output6, app.output7, app.output8)
    # Get rid of the error message if the user clicks the
    # close icon instead of the submit button
    # Any component of the dialog will no longer be available
    # past this point
    try:
        root.destroy()
    except:
        pass
    # To use data outside of function
    # Can either be used in __main__
    # or by external script depending on
    # what calls main()
    return user_input

#%%
def plotConc(lpos, select_ion, filename, savepath):
    conc = np.zeros((len(lpos['time']), ))
    
    num_sel = 0.0
    num_atom = 0.0
    
    isAtom = select_ion.replace(' ', '').isalpha()
    
    for i in range(len(lpos['time'])):
        num_atom += sum(int(x) for x in lpos['ion'][i] if x.isdigit())
        if isAtom:
            split_ion = lpos['ion'][i].split()
            for sel in split_ion:
                if select_ion in sel:
                    num_sel += sum(int(x) for x in sel if x.isdigit())        
        if not isAtom:
            if select_ion == lpos['ion'][i]:
                num_sel += 1   
        conc[i] = num_sel / num_atom
    
    df = pd.DataFrame({"Time" : lpos['time'].to_numpy(), "Concentration" : conc})
    df.to_csv(os.path.join(savepath, filename + '_ConcHist.csv'), index=False)
    
    plt.plot(lpos['time'], conc, 'k')
    plt.xlabel('Minutes')
    plt.ylabel('% Ion')
    plt.title('Concentration History of ' + select_ion)
    plt.savefig(os.path.join(savepath, filename + '_ConcHist'))
    plt.show()        
    
#%%
def plotMisc(ato, savepath, filename):
    plt.plot(ato['time'], ato['Pressure'], 'k')
    plt.xlabel('Minutes')
    plt.ylabel('Torr')
    plt.title('Pressure History')
    plt.savefig(os.path.join(savepath, filename + '_PresHist'))
    plt.show()
    
    plt.plot(ato['time'], ato['Voltage'], 'k')
    plt.xlabel('Minutes')
    plt.ylabel('V')
    plt.title('Voltage History')
    plt.savefig(os.path.join(savepath, filename + '_VoltHist'))
    plt.show()    
    
#%%
def makeGifImage(lpos, filename, path, select_ion, newpath, max_value, min_value):
    #plot frames individually instead of making a gif
    # n = 20 #number of contours to calculate
    #split_data = np.array_split(lpos, n)
    
    time_chunk = int(data[1])
    
    # for t in range(0, int(round(lpos['time'].iloc[-1])), time_chunk):
    #     tsect = lpos[lpos['time'] < (t + time_chunk)]
    #     tsect = tsect[tsect['time'] > t]
    
    # sys.exit()
    
    lpos[['Det_x', 'Det_y']] = lpos[['Det_x', 'Det_y']] * 50
    
    
    grid_adjust = int(round(np.amax(np.abs(lpos[['x', 'y']].to_numpy())) + 1))
    grid_size = 2 * grid_adjust
    
    grid_max = np.amax(np.abs(lpos[['x', 'y']].to_numpy())) + grid_adjust
    grid_min = np.amin(np.abs(lpos[['x', 'y']].to_numpy())) + grid_adjust
    
    # conc_grid_sel = np.zeros((grid_size, grid_size))
    # conc_grid_ion = np.zeros((grid_size, grid_size))
    
    # conc_grid_sel_sq = np.zeros((grid_size, grid_size))
    # conc_grid_ion_sq = np.zeros((grid_size, grid_size))
    
    new_x = np.arange(0, grid_size, 1)
    new_y = np.arange(0, grid_size, 1)
    new_x = new_x - grid_adjust
    new_y = new_y - grid_adjust
    
    X_grid, Y_grid = np.meshgrid(new_x, new_y)
    
    # w_x = lpos['x'].to_numpy() + grid_adjust
    # w_y = lpos['y'].to_numpy() + grid_adjust
    
    # for i in range(w_x.size):
    #     conc_grid_ion[int(round(w_x[i])), int(round(w_y[i]))] += sum(int(x) for x in lpos['ion'][i] if x.isdigit())
    
    sq = int(data[3])
    
    for t in range(0, int(round(lpos['time'].iloc[-1])), time_chunk):
        plt.figure()    
        
        name = str(t)
        
        tsect = lpos[lpos['time'] < (t + time_chunk)]
        tsect = tsect[tsect['time'] > t]
        
        conc_grid_sel = np.zeros((grid_size, grid_size))
        conc_grid_ion = np.zeros((grid_size, grid_size))
        
        conc_grid_sel_sq = np.zeros((grid_size, grid_size))
        conc_grid_ion_sq = np.zeros((grid_size, grid_size))
        
        tsect_np = tsect.to_numpy()
        
        fs_x = tsect_np[:, 0] + grid_adjust
        fs_y = tsect_np[:, 1] + grid_adjust
        
        isAtom = select_ion.replace(' ', '').isalpha()
        
        for i in range(len(tsect_np[:, -1])):
            conc_grid_ion[int(round(fs_x[i])), int(round(fs_y[i]))] += sum(int(x) for x in tsect_np[i, -1] if x.isdigit())
            if isAtom:
                split_ion = tsect_np[i, -1].split()
                for sel in split_ion:
                    if select_ion in sel:
                        conc_grid_sel[int(round(fs_x[i])), int(round(fs_y[i]))] += sum(int(x) for x in sel if x.isdigit())        
            if not isAtom:
                if select_ion == tsect_np[i, -1]:
                    conc_grid_sel[int(round(fs_x[i])), int(round(fs_y[i]))] += 1   
            
        for k in range(0, grid_size, sq):        
            for h in range(0, grid_size, sq):
                conc_grid_sel_sq[h:h + sq, k:k + sq] = np.sum(conc_grid_sel[h:h + sq, k:k + sq])
                conc_grid_ion_sq[h:h + sq, k:k + sq] = np.sum(conc_grid_ion[h:h + sq, k:k + sq])
        
        conc_grid = conc_grid_sel_sq / conc_grid_ion_sq
        
        conc_grid[np.isnan(conc_grid)] = 0
        
        max_conc = np.amax(conc_grid)
        
        conc_grid = (conc_grid / max_conc) * 100
    
        clev = np.arange(0, 100, 1)
    
        normalize = matplotlib.colors.Normalize(vmin=0, vmax=np.amax(conc_grid))
    
        conc_grid[conc_grid > max_value] = 100
        
        conc_grid[conc_grid < min_value] = 0
        
    
        # conc_grid[conc_grid == 0] = np.nan
    
    
        # TO CHANGE COLOR
        # GO TO https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html 
        # AND CHANGE plt.cm.gnuplot2 in the plt.contourf parameters
        # TO plt.cm.COLORMAP 
        # COLORMAP can be replaced with any color code from the above link
    
    
        plt.contourf(X_grid, Y_grid, conc_grid, clev, cmap=plt.cm.gnuplot2, extend='both')
        plt.colorbar(label='% Ion')
        plt.title('Concentration of ' + select_ion)
        
        
        
        plt.xlabel('Å')
        plt.ylabel('Å')
        
        plt.text((-grid_adjust * 0.75), (-grid_adjust * 0.90), f'{t} min', color='white', horizontalalignment='center', verticalalignment='center')
        
        plt.gca().set_aspect('equal')
        
        plt.savefig(os.path.join(newpath, filename + '_' + name ))
        plt.show()    
        
#%%        
def makeGif(filename, newpath, savepath):
    fps = 4
    gif_name = filename
    
    #the section below was for reading out the jpg files in a specific folder
    # parent_dir = newpath
    # for jpg_file in glob.glob(os.path.join(parent_dir, '*.jpg')):
    #     print (jpg_file)
    
    # results = [os.path.basename(f) for f in glob.glob(os.path.join(parent_dir, '*.jpg'))]
    
    file_list = sorted(glob.glob(os.path.join(newpath, "*.png")), key=os.path.getmtime)
    
    clip = mpy.ImageSequenceClip(file_list, fps=fps)
    (w, h) = clip.size
    
    #cropClip = clip.crop(width = h/1.7, height = h/1.7, x_center = w/2, y_center = h/2)
    
    # final = clip.crop(x1 = 420, x2 = 855, y1 = 140, y2 = 570)
    #take this line too
    #cropClip = rotateClip.crop(x1=410, x2=740, y1 = 110, y2 = 440)
    
    #take the line below out of the comment
    #cropClip = clip.crop(x1=410, x2=740, y1 = 110, y2 = 440)
    #final = cropClip.vfx.mask_color(color = [255, 255, 255])
    clip.write_gif('{}.gif'.format(os.path.join(savepath, gif_name)), fps=fps)
    
    shutil.rmtree(newpath)    
 
#%%
def plotArea(lpos, select_ion, filename, savepath):
    answer = 'y'
    count = 1
    while answer == 'y':
    
        xcenter = int(input('X center coordinate: '))
        ycenter = int(input('Y center coordinate: '))
        
        r = int(input('Choose radius: '))
        
        xmin = lambda t: -np.sqrt(r ** 2 - (t - ycenter) ** 2) + xcenter
        xmax = lambda t: np.sqrt(r ** 2 - (t - ycenter) ** 2) + xcenter
        ymin = lambda t: -np.sqrt(r ** 2 - (t - xcenter) ** 2) + ycenter
        ymax = lambda t: np.sqrt(r ** 2 - (t - xcenter) ** 2) + ycenter
        
        span_atom = 0.0
        span_sel = 0.0
        
        xray = lpos['x'].to_numpy()
        yray = lpos['y'].to_numpy()
        
        ray = np.empty((len(lpos['time']), ), dtype=object)
        tray = np.empty((len(lpos['time']), ))
        
        for l in range(len(xray)):
            if xray[l] < xmax(yray[l]) and yray[l] < ymax(xray[l]) and xray[l] > xmin(yray[l]) and yray[l] > ymin(xray[l]):
                ray[l] = lpos['ion'][l]
                tray[l] = lpos['time'][l]
        
        ray = ray[ray != None]
        tray = tray[tray != 0]
        span_conc = np.zeros((len(tray), ))
        
        isAtom = select_ion.replace(' ', '').isalpha()
        
        for h in range(len(ray)):
            span_atom += sum(int(x) for x in ray[h] if x.isdigit())
            if isAtom:
                split_ion = ray[h].split()
                for sel in split_ion:
                    if select_ion in sel:
                        span_sel += sum(int(x) for x in sel if x.isdigit())        
            if not isAtom:
                if select_ion == ray[h]:
                    span_sel += 1   
            span_conc[h] = span_sel / span_atom
            
        df = pd.DataFrame({"Time" : tray, "Concentration" : span_conc})
        df.to_csv(os.path.join(savepath, filename + '_SectConcHist' + str(count) + '.csv'), index=False)        
            
        plt.plot(tray, span_conc, 'k')
        plt.xlabel('Minutes')
        plt.ylabel('% Ion')
        plt.title('Concentration History of ' + select_ion + ' circle w/ radius ' + str(r) + 'Å and center (' + str(xcenter) + ', ' + str(ycenter) + ')')
        plt.savefig(os.path.join(savepath, filename + '_SectConcHist' + str(count)))
        plt.show()  
        count += 1
        
        answer = input('Do you want to plot another area? (y)es or (n)o: ')      
#%%

start = 'y'

while start == 'y':
    # Allow dialog to run either as a script or called from another program
    if __name__ == '__main__':
        data = main()
        # This shows the outputs captured when called directly as `python dual_input.py`
    #### End of dual_input.py code dialog code file #### 
        
    zip_file = data[5]
    archive = zipfile.ZipFile(zip_file, 'r')    
    ato_file = archive.open(archive.namelist()[0], 'r')
    lpos_file = archive.open(archive.namelist()[1], 'r')
    
    ato = pickle.load(ato_file)
    lpos = pickle.load(lpos_file)
        
    select_ion = data[0]
    filename = data[4]
    path = data[6]
    
    max_value = int(data[2])
    min_value = int(data[7])

    newpath = os.path.join(path, filename)
    
    os.mkdir(newpath)

    savepath = os.path.join(path, filename + ' Plots')
    
    os.mkdir(savepath)
    
    plotConc(lpos, select_ion, filename, savepath)
    
    plotMisc(ato, savepath, filename)
    
    makeGifImage(lpos, filename, path, select_ion, newpath, max_value, min_value)
    
    makeGif(filename, newpath, savepath)
        
    plotArea(lpos, select_ion, filename, savepath)
    
    start = input('Create another gif? (y)es or (n)o: ')    
    