# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:43:08 2021

@author: wirt021
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 08:39:30 2021

@author: wirt021

Read APT .ato files


"""
from tkinter import Tk, Text, TOP, BOTH, X, N, LEFT, RIGHT, filedialog, messagebox
from tkinter.ttk import Frame, Label, Entry, Button
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
import sys
import os
import zipfile
from time import sleep

#%%
class SimpleDialog(Frame):

    def __init__(self):
        super().__init__()
        # self allow the variable to be used anywhere in the class
        self.output1 = ''
        self.output2 = ''
        self.output3 = ''
        self.initUI()

    def initUI(self):

        self.master.title("ATO Processor")
        self.pack(fill=BOTH, expand=True)

        frame1 = Frame(self)
        frame1.pack(fill=X)

        btn1 = Button(frame1, text="Choose IVAS files", command=self.select_ivas)
        btn1.pack(padx=5, pady=10)

        frame2 = Frame(self)
        frame2.pack(fill=X)

        lbl2 = Label(frame2, text="Enter file name", width=13)
        lbl2.pack(side=LEFT, padx=5, pady=10)

        self.entry2 = Entry(frame2)
        self.entry2.pack(fill=X, padx=5, expand=True)

        frame3 = Frame(self)
        frame3.pack(fill=X)

        btn2 = Button(frame3, text="Choose pkl folder", command=self.save_folder)
        btn2.pack(padx=5, pady=10)
        
        frame8 = Frame(self)
        frame8.pack(fill=X)
        
        # Command tells the form what to do when the button is clicked
        btn = Button(frame8, text="Submit", command=self.onSubmit)
        btn.pack(padx=5, pady=10)

    def onSubmit(self):

        self.output2 = self.entry2.get()
        self.quit()
        
    def select_ivas(self):
        
        self.output1 = filedialog.askdirectory(parent=self)
        messagebox.showinfo(
        title='Selected IVAS Files',
        message=self.output1
        )

    def save_folder(self):
        
        self.output3 = filedialog.askdirectory(parent=self)
        messagebox.showinfo(
        title='Selected pkl folder',
        message=self.output3
        )

def main():

    # This part triggers the dialog
    root = Tk()
    root.geometry("200x200")
    app = SimpleDialog()
    root.mainloop()
    # Here we can act on the form components or
    # better yet, copy the output to a new variable
    user_input = (app.output1, app.output2, app.output3)
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

# Allow dialog to run either as a script or called from another program
if __name__ == '__main__':
    data = main()

#%%
# %matplotlib qt


def read_epos(f):
    dt = np.dtype([
        ('x','>f'),
        ('y','>f'),
        ('z','>f'),
        ('Da','>f'),
        ('TOF','>f'),
        ('DCV','>f'),
        ('Pulse_v','>f'),
        ('Det_x','>f'),
        ('Det_y','>f'),
        ('pslep','>I'),
        ('ipp','I'),])
    data_ = np.fromfile(f, dtype=dt)
    pddata = pd.DataFrame(data_)
    return pddata    

def read_ato(f):
    
    dt_ATO = np.dtype([
        ('x','<f4'),
        ('y','<f4'),
        ('z','<f4'),
        ('Da','<f4'),
        ('ClustID','<f4'),
        ('Pulse_N', '<f'),
        ('DCV','<f4'),    
        ('TOF','<f4'),
        ('Det_x','<f4'),
        ('Det_y','<f4'),
        ('Pulse_v','<f4'),
        ('VVolt_legacy','<f4'),    
        ('FourierR_legacy','<f4'),
        ('Fourier_Legacy','<f4')])
    data_ = np.fromfile(f, offset = 8, dtype=dt_ATO)
    pddata = pd.DataFrame(data_)
    return pddata

#correct pulse number
def adjust_pulse(pddata):
    pulse = pddata['Pulse_N'] # turns pulse numbers into array
    adjust = 0 # adjusting factor
    adjusted_pulse = np.zeros(pulse.shape)
    zeros = np.where(pulse == 0)[0] # the location of all the resets
    # print(zeros)
    
    skipZero = False
    
    # calculates adjusted pulse number by adding i * 24 bit max int to the corresponding zero
    adjusted_pulse[0:zeros[0]] = pulse[0:zeros[0]] 
    
    dropped_index = np.array([])
    
    for change in range(zeros.size - 1):
        first_index = zeros[change]
        second_index = zeros[change + 1]
        
        isNeighbor = first_index + 1 == second_index
        
        if isNeighbor == True:
            skipZero = True
            dropped_index = np.append(dropped_index, first_index)
            continue
        
        adjust += 16777215.0
        adjusted_pulse[first_index:second_index] = pulse[first_index:second_index] + adjust
    
        # print(adjust)
    
    if skipZero == True:
        adjusted_pulse[zeros[-1]:] = pulse[zeros[-1]:] + 16777215.0 * (zeros.size - dropped_index.size)

    else:
        adjusted_pulse[zeros[-1]:] = pulse[zeros[-1]:] + 16777215.0 * (zeros.size)
        
    # print(16777215 * zeros.size)
            
    pddata['Pulse_N'] = adjusted_pulse
    for drop in dropped_index:
        pddata = pddata.drop(drop)
    return pddata

#calculate time from frequency and number of pulses
def ato_time(pddata):
    
    pulse = pddata['Pulse_N']
    
    freq = pddata['Freq'].to_numpy()
    
    calc_time = np.zeros(pulse.shape)
    
    freq_change = np.where(freq[:-1] != freq[1:])[0] + 1
    
    if freq_change.size == 0:
          
        calc_time = pulse / freq[0] 
    
    else:
    
        calc_time[0:freq_change[0]] = pulse[0:freq_change[0]] / freq[0]
        
        time_adjust = calc_time[freq_change[0] - 1]
        
        for i in range(freq_change.size - 1):
            start_sect = freq_change[i]
            finish_sect = freq_change[i + 1]
            time_sect = pulse[start_sect:finish_sect] - pulse[start_sect - 1]
            calc_time[start_sect:finish_sect] = time_sect / freq[start_sect] + time_adjust
            # print(time_adjust)
            time_adjust = calc_time[finish_sect - 1]

        # print(time_adjust)            
        calc_time[freq_change[-1]:] = (pulse[freq_change[-1]:] - pulse[freq_change[-1] - 1]) / freq[freq_change[-1]] + time_adjust
        
        # # calculates time by dividing the difference between pulse_n by the corresponding frequency
        # total_time = pulse[0] / freq[0]
        # calc_time[0] = total_time
        # for j in range(pulse.size - 1):
        #     total_time += (pulse[j + 1] - pulse[j])/ freq[j + 1]
        #     calc_time[j + 1] = total_time

    calc_time = calc_time / 1000.
    
    pddata['time'] = calc_time / 60.
    
    return pddata
    
#add freqhist
def ato_freqhist(pddata, freqhist):

    pulse = pddata['Pulse_N']
    
    data = np.genfromtxt(freqhist, delimiter=',')
    freq_data = data[1:, 1] 
    ion_sequence = data[1:, 0].astype(int)
    
    assign_freq = np.zeros(pulse.shape)  
    
    assign_freq[0:ion_sequence[0]] = freq_data[0]
    
    assign_freq[0:ion_sequence[0]] = freq_data[0]
    
    firstsect = ion_sequence[0] 
    
    for i in range(ion_sequence.size):
        secondsect = ion_sequence[i + 1]
        assign_freq[firstsect:secondsect] = freq_data[i]
        firstsect = secondsect
        if ion_sequence[i] > assign_freq.size:
            break
    
    pddata['Freq'] = assign_freq    
    
    return pddata

#add pressure
def ato_pressure(pddata, pressure):
    
    pulse = pddata['Pulse_N']
    
    data = np.genfromtxt(pressure, delimiter=',')
    pres_data = data[1:, 1] 
    ion_sequence = data[1:, 0].astype(int)
    
    assign_pres = np.zeros(pulse.shape)  
    
    first_sect = 0
    
    for i in range(ion_sequence.size):
        second_sect = ion_sequence[i]
        assign_pres[first_sect:second_sect] = pres_data[i]
        first_sect = second_sect
        
    pddata['Pressure'] = assign_pres
       
    return pddata

#add voltage
def ato_voltage(pddata, voltage):

    pulse = pddata['Pulse_N']
    
    data = np.genfromtxt(voltage, delimiter=',')
    volt_data = data[1:, 1] 
    ion_sequence = data[1:, 0].astype(int)
    
    assign_volt = np.zeros(pulse.shape)  
    
    first_sect = 0
    
    for i in range(ion_sequence.size):
        second_sect = ion_sequence[i]
        assign_volt[first_sect:second_sect] = volt_data[i]
        first_sect = second_sect
        
    pddata['Voltage'] = assign_volt    
    
    return pddata

def read_rrng(f):
    rf = f

    patterns = re.compile(r'Ion([0-9]+)=([A-Za-z0-9]+).*|Range([0-9]+)=(\d+.\d+) +(\d+.\d+) +Vol:(\d+.\d+) +([A-Za-z:0-9 ]+) +Color:([A-Z0-9]{6})')

    ions = []
    rrngs = []
    for line in rf:
        m = patterns.search(line)
        if m:
            if m.groups()[0] is not None:
                ions.append(m.groups()[:2])
            else:
                rrngs.append(m.groups()[2:])

    ions = pd.DataFrame(ions, columns=['number','name'])
    ions.set_index('number',inplace=True)
    rrngs = pd.DataFrame(rrngs, columns=['number','lower','upper','vol','comp','colour'])
    rrngs.set_index('number',inplace=True)
    
    rrngs[['lower','upper','vol']] = rrngs[['lower','upper','vol']].astype(float)
    rrngs[['comp','colour']] = rrngs[['comp','colour']].astype(str)
    
    return ions,rrngs



def label_ions(pos,rrngs):
    pos['comp'] = ''
    pos['colour'] = '#FFFFFF'
    
    for n,r in rrngs.iterrows():
        pos.loc[(pos.Da >= r.lower) & (pos.Da <= r.upper),['comp','colour']] = [r['comp'],'#' + r['colour']]
    
    pos = pos[pos['comp'] != ''] #remove values with no mass in rrng
    pos.reset_index(inplace = True, drop = True)
    pos['ion'] = pos['comp'].str.replace(':','') #remove colons to make more user friendly
    ion_list = pos['ion'].unique() #produce list of ions for user to select from
    return ion_list, pos #ion_list is list of ions in rrng file, pos is mass labeled file

def filter_pos(keep_list, lpos): #only keeps ions associated with user input values
    lpos_trimmed = lpos[lpos['ion'].isin(keep_list)]
    return lpos_trimmed

def make_gif(data_, n, filepath, ion): #make and save a gif from the dataframe, number of contours to calculate, and filepath

    split_data = np.array_split(data_, n)
    fig, ax = plt.subplots()
    
    grid_adjust = int(round(np.amax(np.abs(lpos[['x', 'y']].to_numpy())) + 1))
    grid_size = 2 * grid_adjust
    
    def animate_contour(data):
        
        for stack in data:
            ax.clear()

            first_section = stack
            
            conc_grid_sel = np.zeros((grid_size, grid_size))
            conc_grid_ion = np.zeros((grid_size, grid_size))
            
            fs_x = first_section['x'].to_numpy() + grid_adjust
            fs_y = first_section['y'].to_numpy() + grid_adjust
            
            new_x = np.arange(0, grid_size, 1)
            new_y = np.arange(0, grid_size, 1)
            
            for i in range(fs_x.size):
                conc_grid_ion[int(round(fs_x[i])), int(round(fs_y[i]))] += sum(int(x) for x in lpos['ion'][i] if x.isdigit())
                split_ion = lpos['ion'][i].split()
                for sel in split_ion:
                    if select_ion in sel:
                        conc_grid_sel[int(round(fs_x[i])), int(round(fs_y[i]))] += sum(int(x) for x in sel if x.isdigit())
                
            sq = 5
                
            for k in range(0, grid_size, sq):        
                for h in range(0, grid_size, sq):
                    conc_grid_sel[h:h + sq, k:k + sq] = np.sum(conc_grid_sel[h:h + sq, k:k + sq])
                    conc_grid_ion[h:h + sq, k:k + sq] = np.sum(conc_grid_ion[h:h + sq, k:k + sq])
            
            conc_grid = conc_grid_sel / conc_grid_ion
            
            conc_grid[np.isnan(conc_grid)] = 0
            
            conc_grid = (conc_grid / np.amax(conc_grid)) * 100 
            
            new_x, new_y = np.meshgrid(new_x, new_y)
            
            plt.contourf(new_x, new_y, conc_grid, 100, cmap='gnuplot2')
            
            t = round(stack['time'].iloc[0], 0) #calculate first time in the stack to display on plot
            ax.text(1,1, f'{t} s', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    
    plt.gca().set_aspect('equal')
    
    ani = animation.FuncAnimation(fig, animate_contour(split_data))
    ani.save(r'C:\Users\voer874\Documents\APT\gifs\contour.gif')
    
    
#%%
#insert file path here as f, keep r in front of quotes to prevent line break error e.g. r"C:\etc"w

files = os.listdir(data[0])


#Choose the ATO file
with open(os.path.join(data[0], files[1]), 'r') as f:
    f_ato = f
    ato = read_ato(f_ato)
# f_ato = r"C:\Users\voer874\Documents\APT\R31_19008\R31_19008\recons\recon-v01\default\R31_19008-v01.ato"

ato = adjust_pulse(ato)

#Choose the RRNG file
with open(os.path.join(data[0], files[0]), 'r') as f:
    f_range = f
    ions, rrngs = read_rrng(f_range)

# Choose the frequency, pressure, and voltage files
with open(os.path.join(data[0], files[2]), 'r') as f:
    freq_file = f
    ato = ato_freqhist(ato, freq_file)
# freq_file = r"C:\Users\voer874\Documents\APT\R31_19008\R31_19008_FreqHist.csv"

ato = ato_time(ato) #include calculated time from frequency and pulse numbers

with open(os.path.join(data[0], files[3]), 'r') as f:
    pres_file = f
    ato = ato_pressure(ato, pres_file)

with open(os.path.join(data[0], files[4]), 'r') as f:
    volt_file = f
    ato = ato_voltage(ato, volt_file)

#insert file path for rrng file to get ranges here

#label the ions and return labeled dataframe and array with all ions
i_list, lpos = label_ions(ato, rrngs) 

output = data[1]

path = data[2]

ato.to_pickle(os.path.join(path, '%s_ato.pkl'%output))

lpos.to_pickle(os.path.join(path, '%s_lpos.pkl'%output))

with zipfile.ZipFile(os.path.join(path, '%s.zip'%output), 'w') as zipF:
    zipF.write(os.path.join(path, '%s_ato.pkl'%output), arcname='%s_ato.pkl'%output, compress_type=zipfile.ZIP_DEFLATED)
    zipF.write(os.path.join(path, '%s_lpos.pkl'%output), arcname='%s_lpos.pkl'%output, compress_type=zipfile.ZIP_DEFLATED)

os.remove(os.path.join(path, '%s_ato.pkl'%output))
os.remove(os.path.join(path, '%s_lpos.pkl'%output))
