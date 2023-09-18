# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 01:34:38 2021

@author: gaomi
"""
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fftpack import fft, fftfreq, ifft
import pandas as pd
from copy import deepcopy
import glob
import os
import random
from scipy.signal import convolve, filtfilt
from scipy.signal.windows import hann
import math
import torch

def slice_data(hr, dp, target_lines= 15000, num_slicing=10, seed=10):
    random.seed(seed)
    if len(hr.shape) == 1:
        hr_sliced = torch.zeros(hr.shape[0]*num_slicing)
    elif len(hr.shape) == 2:
        if hr.shape[1] == 2:
            hr_sliced = torch.zeros(hr.shape[0]*num_slicing,2)
        else:
            ratio = dp.shape[2] // hr.shape[1]
            hr_sliced = torch.zeros(hr.shape[0]*num_slicing, target_lines//ratio)
    dp_sliced = torch.zeros(dp.shape[0]*num_slicing,dp.shape[1],target_lines)

    for i in range(hr.shape[0]):
        if len(hr.shape) == 1:
            hr_sliced[i*num_slicing:(i+1)*num_slicing] = hr[i]
        s_lists = random.sample([j for j in range(dp.shape[2]-target_lines)], num_slicing)
        for j in range(num_slicing):
            s = s_lists[j]
            dp_sliced[i*num_slicing+j,:,:] = dp[i,:,s:s+target_lines]
            if len(hr.shape) == 2:
                if hr.shape[1] == 2:
                    hr_sliced[i*num_slicing+j] = hr[i]
                else:
                    hr_sliced[i*num_slicing+j,:] = hr[i,s//ratio:(s+target_lines)//ratio]
    return hr_sliced, dp_sliced

def load_exp_data(file_dir1, file_dir2, num_slicing=10, dp_freq=1000, target_T=15):
    files = os.listdir(file_dir1)[1:]
    # print(files[0:10])
    bi_array = []
    bq_array = []
    hr_array = []
    rr_array = []
    for f in files:
        # print(f)
        df = pd.read_csv(os.path.join(file_dir1, f))
        hr_label = float(f.split('_')[4][2:])
        rr_label = float(f.split('_')[5][2:])
        if rr_label < 8 or rr_label > 40 or hr_label < 50 or hr_label > 150:
            continue
        bi = df.iloc[:, 0].values
        bq = df.iloc[:, 1].values
        target_lines = target_T*dp_freq
        s_lists = random.sample([i for i in range(len(bi)-target_lines)], num_slicing)
        for i in range(num_slicing):
            s = s_lists[i]
            bi_array.append(bi[s:s+target_lines])
            bq_array.append(bq[s:s+target_lines])
            hr_array.append(hr_label)
            rr_array.append(rr_label)
    
    files = os.listdir(file_dir2)
    for f in files:
        df = pd.read_csv(os.path.join(file_dir2, f))
        hr_label = float(f.split('_')[4][2:])*60
        rr_label = float(f.split('_')[5][2:])*60
        if rr_label < 8 or rr_label > 40 or hr_label < 50 or hr_label > 150:
            continue
        bi = df.iloc[:, 0].values
        bq = df.iloc[:, 1].values
        target_lines = target_T*dp_freq
        s_lists = random.sample([i for i in range(len(bi)-target_lines)], num_slicing)
        for i in range(num_slicing):
            s = s_lists[i]
            bi_array.append(bi[s:s+target_lines])
            bq_array.append(bq[s:s+target_lines])
            hr_array.append(hr_label)
            rr_array.append(rr_label)
    print(bi_array[:10], bq_array[:10])
    return np.array(bi_array), np.array(bq_array), np.array(hr_array), np.array(rr_array)

def load_exp_motion(file_dir1, file_dir2, num_slicing=10, dp_freq=1000, target_T=5):
    files = os.listdir(file_dir1)[1:]
    # print(files[0:10])
    bi_array = []
    bq_array = []
    action_map = {}
    action_array = []
    for f in files:
        # print(f)
        if '.csv' not in f:
            continue
        df = pd.read_csv(os.path.join(file_dir1, f))
        f = f.replace('.csv', '')
        action_label = f.split('_')[-1].split('.')[0].replace(')', '').replace('.csv', '')
        if 'Biopac' in action_label or 'pfh' in action_label:
            continue
        if action_label == 'LayDown' or action_label == 'Laydown':
            action_label = 'LieDown'
        if action_label == 'PseudoJuump' or action_label == 'Pseudojump':
            action_label = 'PseudoJump'
        if action_label == 'LeftSide':
            action_label = 'LieUp'
        if action_label not in action_map:
            action_map[action_label] = 0
        action_map[action_label] += 1
        bi = df.iloc[:, 0].values
        bq = df.iloc[:, 1].values
        target_lines = target_T*dp_freq
        # print(target_lines, len(df))
        if target_lines == len(df):
            bi_array.append(bi)
            bq_array.append(bq)
            action_array.append(action_label)
        else:
            s_lists = random.sample([i for i in range(len(bi)-target_lines)], num_slicing)
            for i in range(num_slicing):
                s = s_lists[i]
                bi_array.append(bi[s:s+target_lines])
                bq_array.append(bq[s:s+target_lines])
                action_array.append(action_label)
    
    files = os.listdir(file_dir2)
    for f in files:
        if '.csv' not in f:
            continue
        df = pd.read_csv(os.path.join(file_dir2, f))
        f = f.replace('.csv', '')
        action_label = f.split('_')[-1].split('.')[0].replace(')', '').replace('.csv', '')
        if 'Biopac' in action_label or 'pfh' in action_label:
            continue
        if action_label == 'LayDown' or action_label == 'Laydown':
            action_label = 'LieDown'
        if action_label == 'PseudoJuump' or action_label == 'Pseudojump':
            action_label = 'PseudoJump'
        if action_label == 'LeftSide':
            action_label = 'LieUp'
        if action_label not in action_map:
            action_map[action_label] = 0
        action_map[action_label] += 1
        bi = df.iloc[:, 0].values
        bq = df.iloc[:, 1].values
        target_lines = target_T*dp_freq
        # print(target_lines, len(df))
        if target_lines == len(df):
            bi_array.append(bi)
            bq_array.append(bq)
            action_array.append(action_label)
        else:
            s_lists = random.sample([i for i in range(len(bi)-target_lines)], num_slicing)
            for i in range(num_slicing):
                s = s_lists[i]
                bi_array.append(bi[s:s+target_lines])
                bq_array.append(bq[s:s+target_lines])
                action_array.append(action_label)
    # print(bi_array[:10], bq_array[:10])
    return np.array(bi_array), np.array(bq_array), np.array(action_array), action_map

def load_exp_data_comb(file_dir1, file_dir2, num_slicing=10, dp_freq=1000, target_T=15):
    files = os.listdir(file_dir1)[1:]
    # print(files[0:10])
    bi_array = []
    bq_array = []
    hr_array = []
    rr_array = []
    action_map = {}
    action_array = []
    for f in files:
        # print(f)
        df = pd.read_csv(os.path.join(file_dir1, f))
        f = f.replace('.csv', '')
        action_label = f.split('_')[-1].split('.')[0].replace(')', '').replace('.csv', '')
        if 'Biopac' in action_label or 'pfh' in action_label:
            continue
        if action_label == 'LayDown' or action_label == 'Laydown':
            action_label = 'LieDown'
        if action_label == 'PseudoJuump' or action_label == 'Pseudojump':
            action_label = 'PseudoJump'
        if action_label == 'LeftSide':
            action_label = 'LieUp'
        if action_label not in action_map:
            action_map[action_label] = 0
        action_map[action_label] += 1

        hr_label = float(f.split('_')[4][2:])
        rr_label = float(f.split('_')[5][2:])
        if rr_label < 8 or rr_label > 40 or hr_label < 50 or hr_label > 150:
            continue
        bi = df.iloc[:, 0].values
        bq = df.iloc[:, 1].values
        target_lines = target_T*dp_freq
        s_lists = random.sample([i for i in range(len(bi)-target_lines)], num_slicing)
        for i in range(num_slicing):
            s = s_lists[i]
            bi_array.append(bi[s:s+target_lines])
            bq_array.append(bq[s:s+target_lines])
            hr_array.append(hr_label)
            rr_array.append(rr_label)
            action_array.append(action_label)
    
    files = os.listdir(file_dir2)
    for f in files:
        df = pd.read_csv(os.path.join(file_dir2, f))
        f = f.replace('.csv', '')
        action_label = f.split('_')[-1].split('.')[0].replace(')', '').replace('.csv', '')
        if 'Biopac' in action_label or 'pfh' in action_label:
            continue
        if action_label == 'LayDown' or action_label == 'Laydown':
            action_label = 'LieDown'
        if action_label == 'PseudoJuump' or action_label == 'Pseudojump':
            action_label = 'PseudoJump'
        if action_label == 'LeftSide':
            action_label = 'LieUp'
        if action_label not in action_map:
            action_map[action_label] = 0
        action_map[action_label] += 1
        hr_label = float(f.split('_')[4][2:])*60
        rr_label = float(f.split('_')[5][2:])*60
        if rr_label < 8 or rr_label > 40 or hr_label < 50 or hr_label > 150:
            continue
        bi = df.iloc[:, 0].values
        bq = df.iloc[:, 1].values
        target_lines = target_T*dp_freq
        s_lists = random.sample([i for i in range(len(bi)-target_lines)], num_slicing)
        for i in range(num_slicing):
            s = s_lists[i]
            bi_array.append(bi[s:s+target_lines])
            bq_array.append(bq[s:s+target_lines])
            hr_array.append(hr_label)
            rr_array.append(rr_label)
            action_array.append(action_label)

    print(bi_array[:10], bq_array[:10])
    return np.array(bi_array), np.array(bq_array), np.array(hr_array), np.array(rr_array), np.array(action_array)

def load_exp_hrrr_xc(file_dir1, file_dir2, target_hr, num_slicing=10, dp_freq=1000, target_T=15):
    files = os.listdir(file_dir1)[1:]
    # print(files[0:10])
    bi_array = []
    bq_array = []
    hr_array = []
    rr_array = []

    omega1 = 30
    omega2 = 40
    dt = 1/dp_freq

    for f in files:
        # print(f)
        df = pd.read_csv(os.path.join(file_dir1, f))
        hr_label = float(f.split('_')[4][2:])
        rr_label = float(f.split('_')[5][2:])
        if rr_label < 8 or rr_label > 40 or hr_label < 50 or hr_label > 150:
            continue
        bi = df.iloc[:, 0].values
        bq = df.iloc[:, 1].values
        target_lines = target_T*dp_freq
        s_lists = random.sample([i for i in range(len(bi)-target_lines)], num_slicing)
        for i in range(num_slicing):
            s = s_lists[i]
            bi_sliced = bi[s:s+target_lines]
            bq_sliced = bq[s:s+target_lines]
            nxc_bi, nxc_bq, nxc_ms_bi, nxc_ms_bq = apply_cross_correlation(bi_sliced, bq_sliced, 
                                                    target_hr, dt, omega1, omega2)
            bi_array.append(nxc_bi)
            bq_array.append(nxc_bq)
            hr_array.append(hr_label)
            rr_array.append(rr_label)
    
    files = os.listdir(file_dir2)
    for f in files:
        df = pd.read_csv(os.path.join(file_dir2, f))
        hr_label = float(f.split('_')[4][2:])*60
        rr_label = float(f.split('_')[5][2:])*60
        if rr_label < 8 or rr_label > 40 or hr_label < 50 or hr_label > 150:
            continue
        bi = df.iloc[:, 0].values
        bq = df.iloc[:, 1].values
        target_lines = target_T*dp_freq
        s_lists = random.sample([i for i in range(len(bi)-target_lines)], num_slicing)
        for i in range(num_slicing):
            s = s_lists[i]
            bi_sliced = bi[s:s+target_lines]
            bq_sliced = bq[s:s+target_lines]
            nxc_bi, nxc_bq, nxc_ms_bi, nxc_ms_bq = apply_cross_correlation(bi_sliced, bq_sliced, 
                                                    target_hr, dt, omega1, omega2)
            bi_array.append(nxc_bi)
            bq_array.append(nxc_bq)
            hr_array.append(hr_label)
            rr_array.append(rr_label)

    # print(bi_sliced, nxc_ms_bi)
    # print(bq_sliced, nxc_ms_bq)
    # print(bi_array[:10], bq_array[:10])
    return np.array(bi_array), np.array(bq_array), np.array(hr_array), np.array(rr_array)

def load_data(filename1, filename2, data_type):
    if filename1 == filename2:
        f1 = open(filename1).read()
        content = f1.split('\n')
        #print('last')
        #print(content[-1])
        content.pop()
        data = []
        if data_type == 'data':
            for line in content:
                temp = line.split(',')
                vector = []
                for t in temp:
                    if t != '':
                        vector.append(float(t))
            #filter_data = lowpass_filter(fs, order, cutoff, vector).copy()
            #data.append(filter_data)
                data.append(vector)
        else:
            for line in content:
                if line != '':
                    data.append(float(line))
        print(len(data))
    else:
        f1 = open(filename1).read()
        f2 = open(filename2).read()
        content1 = f1.split('\n')
        content1.pop()
        content2 = f2.split('\n')
        content2.pop()
        #print(len(content))
        content = content1 + content2
        print(data_type)
        print(len(content))
        data = []
        if data_type == 'data':
            for line in content:
                temp = line.split(',')
                vector = []
                for t in temp:
                    if t != '':
                        vector.append(float(t))
                #filter_data = lowpass_filter(fs, order, cutoff, vector).copy()
                #data.append(filter_data)
                data.append(vector)
        else:
            for line in content:
                if line != '':
                    data.append(float(line))
    return data

def lowpass_filter(fs, order, cutoff, noise_data):
    nyq = fs*0.5
    b, a = butter(order, cutoff/nyq, btype='low')
    filter_data = filtfilt(b, a, noise_data)
    return filter_data
    
def plot_loss(loss):
    """plot perplexities"""
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.plot(loss)

def rolling_std_filter(input_data, k, std_thre):
    """

    Parameters
    ----------
    input_data : numpy array
        original data before denoising
    k : int
        half size of window
    std_thre : float
        threshold of std for removing spike

    Returns
    -------
    Denoised data

    """
    n = len(input_data)
    if type(input_data[0]) == str:
        rstd_data = np.array([float(i) for i in input_data])
    else:
        rstd_data = deepcopy(input_data)
    for i in range(n):
        curr_window = input_data[max(0, i-k): min(n, i+k+1)]
        if type(curr_window[0]) == str:
            curr_window = np.array([float(i) for i in curr_window])
        try:
            curr_mean = np.mean(curr_window)
            curr_std = np.mean(curr_window)
            idx = np.where(np.abs(curr_window - std_thre * curr_std))
            rstd_data[np.array(idx) + max(0, i - k - 1)] = curr_mean
        except:
            print(type(curr_window[0]))
            print(curr_window)
    return rstd_data

def median_filter(input_data, k, thre):
    """

    Parameters
    ----------
    input_data : numpy array
        original data before denoising
    k : int
        half size of window
    thre : float
        threshold of spike height for removing spike

    Returns
    -------
    Denoised data

    """
    n = len(input_data)
    mean_data = np.mean(input_data)
    idx = np.array(np.where(np.abs(input_data - mean_data) >= thre))[0]
    # print(idx)
    median_data = deepcopy(input_data)
    for i in idx:
        # print(i)
        low_b = max(0, i-k)
        up_b = min(n, i+k+1)
        median_data[i] = np.median(input_data[low_b:up_b])
    return median_data

def gauss_smoooth(input_data, k_gauss, fwhm, srate):
    n = len(input_data)
    gtime = 1000*np.array([i for i in range(-k_gauss, k_gauss + 1)]) / srate
    gauswin = np.exp(-(4*np.log(2)*gtime**2) / fwhm**2 )
    # print(np.sum(gauswin))
    gauswin = gauswin / np.sum(gauswin);
    
    gtime_half = 1000*np.array([i for i in range(-k_gauss//2, k_gauss//2 + 1)]) / srate
    gauss_half = np.exp(-(4*np.log(2)*gtime_half**2) / fwhm**2)
    gauss_half = gauss_half / np.sum(gauss_half);
    
    gauss_data = deepcopy(input_data)
    for i in range(k_gauss):
        gauss_data[i] = np.sum(input_data[i:i+k_gauss+1] * gauss_half)
    
    for i in range(k_gauss, n-k_gauss):
        gauss_data[i] = np.sum(input_data[i-k_gauss:i+k_gauss+1] * gauswin)
    
    for i in range(n-k_gauss, n):
        gauss_data[i] = np.sum(input_data[i-k_gauss:i+1] * gauss_half)
   
    return gauss_data

def fft_filter(input_data, srate, up_bound, low_bound):
    n = len(input_data)
    dt = 1/srate
    noisy_data = deepcopy(input_data)
    fft_data = fft(noisy_data)
    # if n%2 == 1:
    #     fft_data = fft_data[:(n+1)//2]
    # else:
    #     fft_data = fft_data[:n//2+1]
    origin_fft_freq = fftfreq(n, dt)
    pos_mask = np.where((np.abs(origin_fft_freq) < up_bound) & (origin_fft_freq > low_bound))
    fft_freq = origin_fft_freq[pos_mask]
    fft_power = np.abs(fft_data)**2
    fft_power = fft_power[pos_mask]
    plt.figure()
    plt.plot(fft_freq[0:], fft_power[0:])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Power')
    plt.close()      
    peak_freq = fft_freq[fft_power.argmax()]
    clean_fft_data = fft_data
    clean_fft_data[origin_fft_freq > peak_freq] = 0
    denoised_data = ifft(clean_fft_data, n)
    return peak_freq, denoised_data
    
    
def comb_denoise(time, input_data, k_rstd, std_thre, k_median, median_thre, k_gauss, fwhm, srate, freq_up_thre, freq_lw_thre):
    rstd_data = rolling_std_filter(input_data, k_rstd, std_thre)
    # print(input_data[:5])
    median_data = median_filter(rstd_data, k_median, median_thre)
    # print(input_data[:5])
    gauss_data = gauss_smoooth(median_data, k_gauss, fwhm, srate)
    # print(input_data[:5])
    freq_label, denoised_data = fft_filter(gauss_data, srate, freq_up_thre, freq_lw_thre)
    # plt.legend('input data', 'denoised data')
    return freq_label, denoised_data
    
def denoise_pipline(file_name):
    k_rstd = 15
    std_thre = 2
    k_median = 100
    rr_median_thre = 5
    hr_median_thre = 0.1
    k_gauss = 250
    fwhm = 250
    srate = 250
    hr_freq_thre = 3
    rr_freq_thre = 2
    
    data = pd.read_csv(file_name, header=None, names=['sec', 'RR', 'HR'], skiprows=9)
    time = data['sec']*60
    rr_df = data['RR']
    hr_df = data['HR']
    # print(rr_df.values)
    # print(hr_df.values)
    rr_freq, rr_denoised = comb_denoise(time.values[:7500], rr_df.values[:7500], k_rstd, std_thre, k_median, rr_median_thre, k_gauss, fwhm, srate, rr_freq_thre)
    hr_freq, hr_denoised = comb_denoise(time.values[:7500], hr_df.values[:7500], k_rstd, std_thre, k_median, hr_median_thre, k_gauss, fwhm, srate, hr_freq_thre)
    
    rr_freq_fft, fft_rr = fft_filter(rr_df.values[:7500], srate, rr_freq_thre)
    hr_freq_fft, fft_hr = fft_filter(hr_df.values[:7500], srate, hr_freq_thre)
    plt.figure()
    plt.plot(time.values[:7500], rr_df.values[:7500], '-b')
    plt.plot(time.values[:7500], rr_denoised[:7500], '-g')
    plt.xlabel('Time (s)')
    plt.legend(['Origin RR', 'Denoised RR'])
    plt.figure()
    plt.plot(time.values[:7500], hr_df.values[:7500], '-k')
    plt.plot(time.values[:7500], hr_denoised[:7500], '-r')
    plt.xlabel('Time (s)')
    plt.legend(['Origin HR', 'Denoised HR'])
    # plt.legend(['Origin RR', 'Denoised RR', 'Origin HR', 'Denoised HR'])
    return rr_freq, hr_freq

def visualize_resnet(model, input_tensor, use_cuda):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
    targets = [ClassifierOutputTarget(5)]    
    grayscale_cam = cam(input_tensor=input_tensor,targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    # visualization = show_cam_on_image(rgb_img, )

def single_hr_cycle(target_hr, dt, omega1, omega2):
    T = 60/target_hr
    t = np.linspace(0, T, int(T/dt), endpoint=False)
    omega1 /= T
    omega2 /= T
    gamma = 1
    b = T/2
    c = T*1e-3
    pulse = np.exp(-(t-b)**2./c)
    g_shape = np.cos(omega1*t + gamma*np.sin(omega2*t));
    w = g_shape * pulse
    # normalize
    w = w/max(abs(w));
    if abs(min(w)) < abs(max(w)):
       w = -w
    return t, w

def cross_correlation(filter, radar):
    # Filter signal+noise with the expected pulse-signal (without normal)
    cc_wt_normal = convolve(radar, filter,'same')

    # Uniform window for calculating local norm (average)
    temp_flt = np.ones(len(filter))
    # Calculate local norm of radar
    localnorms = convolve(radar**2, temp_flt, 'same')
    localnorms = np.sqrt(localnorms)

    # Local norm of embedded signal
    flt_sq = np.sqrt(np.sum(filter**2))

    # Prevent division by zero, force local norm of signal to be greater than 0
    e = 1e-9
    indsNull = np.where(localnorms < e)
    localnorms[indsNull] = e

    # Return normalized cross correlation
    normalized_cc = cc_wt_normal/localnorms/flt_sq
    return normalized_cc

def apply_cross_correlation(bi, bq, target_hr, dt, omega1, omega2, NZandFLTscale=1):
    my_flt = hann(math.ceil(200*NZandFLTscale))
    my_flt = my_flt/np.sum(my_flt)
    lpf2_bi = filtfilt(my_flt, 1, bi)
    lpf2_bq = filtfilt(my_flt, 1, bq)
    ms_bi = bi - lpf2_bi
    ms_bq = bq - lpf2_bq

    tTAR, wTAR = single_hr_cycle(target_hr, dt, omega1, omega2)
    TMPL = -wTAR[(len(wTAR)//2 - math.ceil(100*NZandFLTscale)):(len(wTAR)//2 + math.ceil(100*NZandFLTscale))]
    
    nxc_bi = cross_correlation(TMPL, bi)
    nxc_bq = cross_correlation(TMPL, bq)
    nxc_ms_bi = cross_correlation(TMPL, ms_bi)
    nxc_ms_bq = cross_correlation(TMPL, ms_bq)

    return nxc_bi, nxc_bq, nxc_ms_bi, nxc_ms_bq

if __name__ == '__main__':
    result = {}
    for bp_file in glob.glob('*Biopac*.csv'):
        print(bp_file)
        result[bp_file] = denoise_pipline(bp_file)
    
    print(result)
        
    