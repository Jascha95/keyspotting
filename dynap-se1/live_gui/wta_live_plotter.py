#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:53:45 2021

@author: dzenn
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from threading import Thread

from time import sleep, time
from numba import njit

import numpy as np

import tkinter as tk

class WtaActivityPlot():
    
    def __init__(self, sink_node=None, refresh_rate=10, input_ids_groups=None, exc_cluster_size=None, TG=None):

        
        self.sink_node = sink_node
        self.refresh_rate = refresh_rate
        
        self.TG = TG
        self.mode = 'global_inh'
        
        if input_ids_groups:
            self.exc_ids = input_ids_groups[0]
            self.inh_ids = input_ids_groups[1]
            if len(input_ids_groups) == 3:
                self.mode='global_exc'
                self.global_exc_ids = input_ids_groups[2]
            self.raster_y_range = (self.exc_ids[0]-1, self.exc_ids[-1]+1)
        else:
            self.raster_y_range = None
            
        self.exc_cluster_size=exc_cluster_size
        
        self.main = tk.Tk()
        self.main.geometry("800x800+100+50")
        self.main.title("Activity gui")
        
        self.activity_dt = 0.5
        
        evts = self.sink_node.get_events()
        evts_n = np.array([[evt.timestamp, evt.neuron_id] for evt in evts])
        
        self.raster_x = np.arange(0,100,1)*1e-06
        self.raster_y = np.ones(100, dtype=int)
        
        self.start_timestamp = self.raster_x[0]
        self.start_time = time()
        # self.timestamp_delta = 
        
        
        self.raster_fig, axes = plt.subplots(2,2, figsize=(12,8), dpi = 100, gridspec_kw={'width_ratios': [2, 1]})
        self.raster_ax = axes[0][0]
        self.activity_ax = axes[0][1]
        self.instant_rates_ax = axes[1][0]
        self.instant_rates_inh_ax = axes[1][1]
        
                
        self.raster_plot, = self.raster_ax.plot(self.raster_x, self.raster_y, '|')
        self.raster_ax.set_xlabel("time (s)")
        self.raster_ax.set_ylabel("neuron_id")
        self.raster_ax.set_ylim((self.exc_ids[0]-1, self.exc_ids[-1]+1))
        
             
        self.activity_x, self.activity_y = get_activity(self.raster_x, self.raster_y, (self.exc_ids[0], self.exc_ids[-1]))
        
        
        self.activity_plot, = self.activity_ax.plot(self.activity_x, self.activity_y)
        self.activity_ax.set_xlabel("time (s)")
        self.activity_ax.set_ylabel("core activity")
        
        
        rates = get_neuron_rates(self.raster_x, self.raster_y, 256)
        self.instant_rates_input_plot = self.instant_rates_ax.bar(np.arange(len(TG.input_ids)*TG.exc_cluster_size)+TG.start_neuron, np.repeat(TG.input_rates, TG.exc_cluster_size),
                                                                  alpha=0.3)
        self.instant_rates_plot = self.instant_rates_ax.bar(self.exc_ids, rates[:len(self.exc_ids)])
        self.instant_rates_ax.set_xlabel("neuron id")
        self.instant_rates_ax.set_ylabel("firing rate, Hz")

        if self.mode=='global_exc':
            ids = np.concatenate((self.inh_ids, self.global_exc_ids))
        else:
            ids = self.inh_ids
        self.instant_rates_inh_plot = self.instant_rates_inh_ax.bar(ids, rates[ids])
        self.instant_rates_inh_ax.set_xlabel("neuron id")
        self.instant_rates_inh_ax.set_ylabel("firing rate, Hz")

        
        
        self.raster_fig.tight_layout()
        
        self.raster_tk = FigureCanvasTkAgg(self.raster_fig, self.main)
        self.raster_tk.get_tk_widget().pack()
            
            
        
        self.main.after(self.refresh_rate, self.update_plot)
        self.main.mainloop()
        
        
    def update_plot(self):
        
        evts = self.sink_node.get_events()
        
        if len(evts) != 0:
            evts_n = np.array([[evt.timestamp, evt.neuron_id] for evt in evts])
            
            # plt.plot(evts_n[:,0], evts_n[:,1], '.')
            # activity_x, activity_y = get_activity(self.activity_y[-1], evts_n[:,0]*1e-06, self.raster_x[-1],
            #                                       evts_n[-1,0]*1e-06, self.activity_dt)
            
            # self.activity_x = np.concatenate((self.activity_x, activity_x))
            # self.activity_y = np.concatenate((self.activity_y, activity_y))
            
            # select_mask = (self.activity_x[-1] - self.activity_x) < 5
            
            # self.activity_x = self.activity_x[select_mask]
            # self.activity_y = self.activity_y[select_mask]
            

            self.raster_x = np.concatenate((self.raster_x, evts_n[:,0]*1e-06))
            self.raster_y = np.concatenate((self.raster_y, evts_n[:,1]))
            
            select_mask = (self.raster_x[-1] - self.raster_x) < 5
            
            self.raster_x = self.raster_x[select_mask]
            self.raster_y = self.raster_y[select_mask]
            # self.x.append(self.x[-1] + 1)
            # self.y.append(random.random()*50)
            
            self.raster_plot.set_xdata(self.raster_x)
            self.raster_plot.set_ydata(self.raster_y)
            self.raster_ax.set_xlim((self.raster_x.min(), self.raster_x.max()))
            
            if len(self.activity_x):
                if self.raster_x[-1] - self.activity_x[-1] > self.activity_dt:
                    self.activity_x, self.activity_y = get_activity(self.raster_x, self.raster_y, (self.exc_ids[0], self.exc_ids[-1]), self.activity_dt)
            else:
                self.activity_x, self.activity_y = get_activity(self.raster_x, self.raster_y, (self.exc_ids[0], self.exc_ids[-1]), self.activity_dt)
            
            self.activity_plot.set_xdata(self.activity_x)
            self.activity_plot.set_ydata(self.activity_y)
            self.activity_ax.set_xlim((self.activity_x.min(), self.activity_x.max()))
            self.activity_ax.set_ylim((0, self.activity_y.max()+10))
            
            
            
            rates = get_neuron_rates(self.raster_x[(self.raster_x[-1] - self.raster_x) < 1], self.raster_y[(self.raster_x[-1] - self.raster_x) < 1],256)
            for rect, rate in zip(self.instant_rates_plot, rates[self.exc_ids]):
                rect.set_height(rate)
            self.instant_rates_ax.set_ylim((0, rates.max()+10))
            
            for rect, rate in zip(self.instant_rates_input_plot, np.repeat(self.TG.input_rates, self.TG.exc_cluster_size)):
                rect.set_height(rate)
            
            if self.mode=='global_exc':
                ids = np.concatenate((self.inh_ids, self.global_exc_ids))
            else:
                ids = self.inh_ids
            for rect, rate in zip(self.instant_rates_inh_plot, rates[ids]):
                rect.set_height(rate)
            self.instant_rates_inh_ax.set_ylim((0, rates.max()+10))
            
            if self.raster_y_range is None:
                self.raster_ax.set_ylim((self.raster_y.min(), self.raster_y.max()))
            else:
                self.raster_ax.set_ylim(self.raster_y_range)
            try:    
                self.raster_tk.draw()
            except Exception:
                print("(Drawing hiccup)")
            self.raster_tk.flush_events()
            
        # else:
            # print("No events to plot!")
            
            
        self.main.after(self.refresh_rate, self.update_plot)



# @njit
def get_neuron_rates(spiketimes, neuron_ids, num_neurons, time_window=1):
    rates = np.zeros(num_neurons)
    max_timestamp = spiketimes[-1]
    for spike_id in np.arange(len(spiketimes)):
        if max_timestamp-spiketimes[spike_id] < time_window:
            rates[neuron_ids[spike_id]] += 1
    
    return rates

# @njit        
def get_trace(last_activity_value, spiketimes, t_start, t_end, dt):
    
    x = []
    y = []
    
    t_current = t_start
    activity_current = last_activity_value
    activity_prev = last_activity_value
    spike_id=0
    spike_id_last=len(spiketimes)
    
    for t_current in np.arange(t_start+dt, t_end, dt):
        activity_current = relaxate(activity_prev, 1, dt)
        while(spike_id<spike_id_last and spiketimes[spike_id] < t_current):
            activity_current += 1
            spike_id += 1
        x.append(t_current)
        y.append(activity_current)
        activity_prev = activity_current
        
    return x, y


# @njit        
def get_activity(spiketimes, neuron_ids, mask_ids, time_window=0.1):
    
    x = []
    y = []
    
    t_start = spiketimes[0]
    t_end = spiketimes[-1]
    
    spike_id=0
    spike_id_last=len(spiketimes)
    
    for t_current in np.arange(t_start+time_window, t_end, time_window):
        activity_current = 0
        while(spike_id<spike_id_last and spiketimes[spike_id] < t_current):
            if (neuron_ids[spike_id] >= mask_ids[0] and neuron_ids[spike_id] < mask_ids[1]):
                activity_current += 1
            spike_id += 1
        x.append(t_current)
        y.append(activity_current/((mask_ids[1] - mask_ids[0])*time_window))
        
    return np.array(x), np.array(y)

# @njit         
def relaxate(A, tau, delta_t):
    return A*np.exp(-delta_t/tau)

# def plot_activity():
        
#     x = [1,2,3,4,5,6,7,8,9,10]
#     y = [30,32,34,32,33,31,29,32,35,45]
    
#     fig = plt.figure()        
#     p1, = plt.plot(x, y)
    
#     for i in range(20):
#         x = x[1:]
#         y = y[1:]
    
#         x.append(x[-1] + 1)
#         y.append(random.random()*50)
        
#         p1.set_xdata(x)
#         p1.set_ydata(y)
#         fig.canvas.draw()
#         fig.canvas.flush_events()
#         plt.xlim((x[0], x[-1]))
#         plt.ylim((min(y), max(y)))
        
#         sleep(0.5)
        
#         print(x)
        
def run_gui(sink_node, refresh_rate, input_ids_groups, TG):
    gui = WtaActivityPlot(sink_node, refresh_rate, input_ids_groups, TG=TG)
    
def run_plotting_thread(sink_node, refresh_rate, input_ids_groups=None, TG=None):
    
    import threading
    
    gui_thread = threading.Thread(target=run_gui, args=(sink_node, refresh_rate,input_ids_groups, TG, ))
    gui_thread.start()  
    
    
      

# if __name__ == '__main__':
    
    
#     # plot_activity()
#     # t1 = Thread(target = plot_activity)
#     # t1.start()
    
#     # run_plotting_thread()