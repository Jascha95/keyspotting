"""
author: julianb
date: 20.03.2020
email: julian.buechel@gmail.com
"""

import numpy as np
import matplotlib
# matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['lines.linewidth'] = 0.5
# matplotlib.rcParams['lines.markersize'] = 0.5
import matplotlib.pyplot as plt
import os
#from numba import njit
import time
from typing import Tuple

# - 25 x faster than without njit. One iteration takes about 25 micro seconds
# @njit
def find_rising_and_falling_points(voltage_smoothed : np.ndarray, t_min: float, t_max : float, dt : float) -> Tuple[np.ndarray,np.ndarray]:
    """
    :brief Given the membrane potential of a neuron, this method finds the times where a current is applied (rising times)
            and the points where the current induction stops (falling times).
    :param voltage_smoothed : np.ndarray : The recorded membrane potential, filtered with a moving average
    :param t_min : float : Smallest time value
    :param T_max : float : Biggest time value
    :param dt : float : Time step of the recordings
    :param threshold : float : Used to divide the signal into two classes
    :returns : (rising_times,falling_times) : Tuple(np.ndarray,np.ndarray) : The rising and falling times recorded 
    """

    # - Calculate a descent threshold based on the min and max values of the data
    threshold = (np.max(voltage_smoothed) + np.min(voltage_smoothed))/2

    above = voltage_smoothed > threshold
    # - Divide the membrane potential into regions above and below a threshold
    # - Set a minimum width for a plateau to count as one. This is important if the signal is very noisy
    min_plateau_width = int(0.01/dt)
    t_smoothed = np.linspace(t_min,t_max,len(voltage_smoothed))

    cur_min_value = np.inf
    cur_min_time = 0
    cur_min_idx = 0

    cur_max_value = 0
    cur_max_time = 0
    cur_max_idx = 0

    continuous_min = False
    continuous_max = False

    cur_width_min = 0
    cur_width_max = 0

    min_plateau = []
    max_plateau = []

    # - Find plateaus and remember the time, index and value of the maximum and minimum per plateau
    # - In theory, these values would be the falling and rising times already, but due to the noise
    # - the maximum or minimum could lie before the actual falling or rising point.
    for idx,is_above in enumerate(above):
        if(is_above):
            if(continuous_max and voltage_smoothed[idx] > cur_max_value):
                cur_max_value = voltage_smoothed[idx]
                cur_max_time = t_smoothed[idx]
                cur_max_idx = idx
                cur_width_max += 1
            elif(not continuous_max):
                cur_max_value = voltage_smoothed[idx]
                cur_max_time = t_smoothed[idx]
                cur_max_idx = idx
                cur_width_max += 1
                continuous_max = True
            else:
                cur_width_max += 1
            
            if(continuous_min):
                if(cur_width_min > min_plateau_width):
                    min_plateau.append((cur_min_value,cur_min_time,cur_min_idx))
                cur_min_time = 0
                cur_min_value = np.inf
                cur_min_idx = 0
                cur_width_min = 0
                continuous_min = False
            
        else:
            if(continuous_min and voltage_smoothed[idx] > cur_min_value):
                cur_min_value = voltage_smoothed[idx]
                cur_min_time = t_smoothed[idx]
                cur_min_idx = idx
                cur_width_min += 1
            elif(not continuous_min):
                cur_min_value = voltage_smoothed[idx]
                cur_min_time = t_smoothed[idx]
                cur_min_idx = idx
                cur_width_min += 1
                continuous_min = True
            else:
                cur_width_min += 1
            
            if(continuous_max):
                if(cur_width_max > min_plateau_width):
                    max_plateau.append((cur_max_value,cur_max_time,cur_max_idx))
                cur_max_time = 0
                cur_max_value = 0
                cur_max_idx = 0
                cur_width_max = 0
                continuous_max = False
            
    # - Given the min and max plateaus, we want to find the actual falling and rising time
    # - If we look ahead of the current max- or minimum for 6 ms and we see an increase or
    # - decrease by at least 10% of the total voltage range, we determine this point as
    # - a falling or rising point
    lookahead_threshold = 0.1*(max(voltage_smoothed)-min(voltage_smoothed))
    lookahead_step = int(0.006/dt)

    rising_times = []
    falling_times = []

    # - Get the rising times
    for min_value,min_time,idx in min_plateau:
        new_val = min_value
        new_time = min_time
        new_idx = idx

        if(new_idx+lookahead_step < len(voltage_smoothed)):
            diff_lookahead = voltage_smoothed[new_idx+lookahead_step] - new_val
            while(diff_lookahead < lookahead_threshold):
                new_idx += 1
                new_val = voltage_smoothed[new_idx]
                new_time = t_smoothed[new_idx]
                if(new_idx+lookahead_step < len(voltage_smoothed)):
                    diff_lookahead = voltage_smoothed[new_idx+lookahead_step] - new_val
                else:
                    break
        
        rising_times.append(new_time)

    # - Get the falling times
    for max_value,max_time,idx in max_plateau:
        new_val = max_value
        new_time = max_time
        new_idx = idx

        if(new_idx+lookahead_step < len(voltage_smoothed)):
            diff_lookahead = new_val - voltage_smoothed[new_idx+lookahead_step]
            while(diff_lookahead < lookahead_threshold):
                new_idx += 1
                new_val = voltage_smoothed[new_idx]
                new_time = t_smoothed[new_idx]
                if(new_idx+lookahead_step < len(voltage_smoothed)):
                    diff_lookahead = new_val - voltage_smoothed[new_idx+lookahead_step]
                else:
                    break
        
        falling_times.append(new_time)

    return (np.asarray(rising_times),np.asarray(falling_times))

    
# @njit
def get_falling_voltage_and_time(v : np.ndarray, t : np.ndarray, falling_times : np.ndarray, rising_times : np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    """
    :brief Need to extract a falling time suitable for calculating the time constant
            Loop through the falling times. If there is only one time, take this one.
            If there are multiple, take the one that ends at a rising time
    :param v : np.ndarray : Membrane potential over time
    :param t : np.ndarray : Time array
    :param falling_times : np.ndarray : The falling times obtained from the voltage trace
    :param rising_times : np.ndarray : The rising times obtained from the voltage trace
    :returns : Tuple[np.ndarray,np.ndarray] : Tuple consisting of membrane potential and time trace during fall
    """
    t_max = t[-1]
    t_min = t[0]
    if(len(falling_times) == 1):
        # - There is only one
        f_time_begin = falling_times[0]
        found_one = False
        for r_time in rising_times:
            if(f_time_begin < r_time and not found_one):
                f_time_end = r_time
                found_one = True
                break
        if(not found_one):
            f_time_end = t_max
    elif(len(falling_times) > 1):
        # - 1) Check if there is one in the middle
        total_times = np.hstack((rising_times,falling_times))
        total_times = np.sort(total_times)
        found_middle = False
        for i in range(1,len(total_times)-1):
            for f_time in falling_times:
                if(f_time == total_times[i]):
                    # - Found one in the middle
                    f_time_begin = f_time
                    f_time_end = total_times[i+1]
                    found_middle = True
                    break
            if(found_middle):
                break
        
        if(not found_middle):
            # - Take first one
            f_time_begin = falling_times[0]
            found_one = False
            for r_time in rising_times:
                if(f_time_begin < r_time and not found_one):
                    f_time_end = r_time
                    found_one = True
                    break
            if(not found_one):
                f_time_end = t_max
    else:
        print("No falling times!")

    falling_voltage = v[(t > f_time_begin) & (t < f_time_end)]
    falling_time = t[(t > f_time_begin) & (t < f_time_end)]

    return (falling_voltage,falling_time)

# @njit
def get_tau(falling_voltage : np.ndarray, falling_time : np.ndarray):
    """
    :brief This function fits an exponential of the form V(t) = A*exp(-t/tau)+B
            to the falling membrane potential
    :params : falling_voltage : np.ndarray : Membrane potential during the fall
    :params : falling_time : np.ndarray : Array of the times during the fall
    :returns : tau : float : Time constant of the neuron
    """
    # - Fit exponential to get time constant
    duration = falling_time[-1]-falling_time[0]
    t_fit = np.linspace(0,duration,len(falling_time))
    # - Fit exponential function of type f(t) = A*exp(-t/tau)+B
    B = np.min(falling_voltage)
    A = np.max(falling_voltage) - B
    eps = 0.000001
    tau = np.median(np.divide(-t_fit,(np.log((falling_voltage - B + eps)/A))))
    return tau


def get_falling_times(neuron_trace : np.ndarray, dt : float, ma_filter : np.ndarray, debug : int = 0):
    """
    :brief Given a time trace of a neurons membrane potential, this method computes the falling times
            meaning the times after a subthreshold current was applied to the neuron
    :param neuron_trace : np.ndarray : This two dimensional array holds the times (in s) in the first column and the membrane
            membrane potentials in the second
    :param : float : dt : dt of the time recording in seconds. If None, will be calculated from the time trace
    :param : np.ndarray : ma_filter : Filter  for the moving average
    :returns np.ndarray of size one dimension containing the times where the membrane potential is falling 
    """
    t = neuron_trace[:,0]
    t_min = t[0]
    t_max = t[-1]

    # - Get the voltage
    v = neuron_trace[:,1]

    # - Smooth the voltage trace using moving average
    voltage_smoothed = np.convolve(v, ma_filter, mode='valid')

    return find_rising_and_falling_points(voltage_smoothed, t_min, t_max, dt)


def get_tau_core(neuron_traces : np.ndarray, dt : float, ma_filter : np.ndarray, debug : int):
    # - Skip the first 20 ms
    N_skip = int(0.02/dt)

    # - Store the computes taus
    taus = np.empty((neuron_traces.shape[0],))
    if(debug > 1):
        fig = plt.figure(figsize=(7,5))

    # - Loop over all neuron traces
    for i in range(neuron_traces.shape[0]):
        if(debug > 1):
            print(i)
        if(i == 1):
            t0 = time.time()
        neuron_trace = neuron_traces[i,:,:]

        N = neuron_trace.shape[0]
        neuron_trace = neuron_trace[N_skip:N-N_skip,:]
        v = neuron_trace[:,1]
        t = neuron_trace[:,0]
        t_min = t[0]
        t_max = t[-1]

        # - Obtain the times of the rising and falling points
        rising_times, falling_times = get_falling_times(neuron_trace, dt=0.0002, ma_filter=ma_filter, debug=DEBUG)

        # - Add a constant offset
        rising_times += 0.008
        falling_times += 0.008

        assert(len(falling_times)>0), "Length is zero"

        # - Extract one array of a falling time to calculate the time constant 
        falling_voltage, falling_time = get_falling_voltage_and_time(v, t, falling_times, rising_times)

        # - Fit exponential of the form V(t) = A*exp(-t/tau)+B
        tau = get_tau(falling_voltage, falling_time)
        taus[i] = tau

        # - Plot the results
        if(debug > 1):
            is_rising=True
            if(falling_times[0] < rising_times[0]):
                is_rising = False
            
            r_c = 0
            f_c = 0

            plt.clf()
            ax1 = fig.add_subplot(311)
            ax1.set_xlim([t_min-0.01,t_max+0.01])
            for i in range(len(rising_times)+len(falling_times)):
                if(is_rising):
                    rise_time = rising_times[r_c]
                    try:
                        until = falling_times[f_c]
                    except:
                        until = t_max
                    r_c += 1
                    is_rising = False
                    # - Plot the line
                    to_plot = v[(t >= rise_time) & (t < until)]
                    l1 = ax1.plot(np.linspace(rise_time,until,len(to_plot)),to_plot, color="C2")
                else:
                    fall_time = falling_times[f_c]
                    try:
                        until = rising_times[r_c]
                    except:
                        until = t_max
                    f_c += 1
                    is_rising = True
                    # - Plot the line
                    to_plot = v[(t >= fall_time) & (t < until)]
                    l2 = ax1.plot(np.linspace(fall_time,until,len(to_plot)),to_plot, color="C4")

            lines = [l1[0],l2[0]]
            ax1.legend(lines, ["Rising","Falling"])
            
            ax2 = fig.add_subplot(312)
            ax2.set_xlim([t_min-0.01,t_max+0.01])
            ax2.plot(t,v)

            # - Calculate the fitted function
            duration = falling_time[-1]-falling_time[0]
            t_fit = np.linspace(0,duration,len(falling_time))
            B = np.min(falling_voltage)
            A = np.max(falling_voltage) - B
            f_fit = A*np.exp(-t_fit/tau) + B

            ax3 = fig.add_subplot(313)
            ax3.set_xlim([t_min-0.01,t_max+0.01])
            ax3.plot(falling_time,falling_voltage, color="C4")
            ax3.plot(falling_time, f_fit, color="C2")

            plt.tight_layout()

            plt.draw()
            plt.pause(2.0)

    if(debug > 1):
        plt.show()

    if(debug > 0):
        print("Time for one core %.4f s" % (time.time()-t0))
    
    return taus


if __name__ == '__main__':
    # - Start main script
    DEBUG = 2
    
    dt = 0.0002
    
    N_filter = int(0.002 / dt)
    ma_filter = np.ones((N_filter,))/N_filter
    
    # - Collect the paths that start with traces..
    traces_paths = []
    path = os.path.join(os.getcwd(),"Resources/")
    for i in os.listdir(path):
        if(os.path.isfile(os.path.join(path,i)) and 'traces' in i):
            traces_paths.append(os.path.join(path,i))
    
    taus = []
    
    for trace_path in traces_paths:
        neuron_traces = np.load(os.path.join(trace_path)) 
    
        taus_trace = get_tau_core(neuron_traces, dt, ma_filter, debug=DEBUG)
        taus.append(taus_trace)
    
    # - Load a sample file
    # neuron_traces = np.load(os.path.join(os.getcwd(), "Resources/traces12.npy"))
    # taus = get_tau_core(neuron_traces, dt, ma_filter, debug=DEBUG)
    
    names = []
    fig = plt.figure(figsize=(7,5))
    plt.xlim([0,0.06])
    for i,t in enumerate(taus):
        l = plt.hist(t, bins=50, density=True, histtype="step")
        names.append(("Core %d" % i))
    
    plt.legend(names)
    plt.show()
    
    # - Plot distribution of taus
