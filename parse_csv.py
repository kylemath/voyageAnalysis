import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def butterworth_bandpass(signal):
    # Define filter parameters
    order = 4
    lowcut = .5
    highcut = 5
    fs = 64

    # Create Butterworth filter
    b, a = butter(order, [lowcut, highcut], btype='bandpass', fs=fs)

    # Apply the filter to your signal
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal

def parse_csv(file_name):
    #open data csv
    df = pd.read_csv(file_name, header=0)

    #parse name
    name = file_name
    row_seperator = "_"
    value_seperator = "%"
    file_values = dict()
    for pair in name.split(row_seperator):
        try:
            key, value = pair.split(value_seperator)
            file_values[key] = value
        except Exception as e:
            pass #not a value line
    print(file_values)

    #get just rows of certain type
    eeg_lines = df[df['type'] == 'eeg']
    ppg_lines = df[df['type'] == 'ppg']

    #get just the data 
    eeg_channel_names = "e[1-5]_*"
    ppg_channel_names = "ppg[1-9]_*"
    eeg_data = eeg_lines.filter(df.filter(regex=(eeg_channel_names)))
    ppg_data = ppg_lines.filter(df.filter(regex=(ppg_channel_names)))

    #add on the timestamp column
    eeg_data["timestamp"] = eeg_lines["timestamp"]
    ppg_data["timestamp"] = ppg_lines["timestamp"]
    return eeg_data, ppg_data

#parse the data
eeg_data, ppg_data = parse_csv(sys.argv[1])

ppg_srate = 64
ppg_period = 1/ppg_srate #seconds
ppg_length = ppg_data.shape[0]
timestamps = list(np.arange(0, ppg_period*ppg_length, ppg_period))
ppg_data['timestamp'] = timestamps

## to take a subset of a dataset
# start = 600
# end = 650

# mask = (ppg_data["timestamp"] >= start) & (ppg_data["timestamp"] <= end)
# ppg_data = ppg_data[mask]

#plot the spectra data
chans = ['e1', 'e2', 'e3', 'e4']
freqs = list(range(0,125,4))

fig, ax = plt.subplots(1,4)
nColumns = 32
for ichan in range(4):
    for iwin in range(1000):
        ax[ichan].plot(freqs, eeg_data.iloc[iwin, ichan*nColumns:(ichan+1)*nColumns])
plt.show()



fig, ax = plt.subplots(7, 1)
ax[0].plot(ppg_data['timestamp'], ppg_data['ppg1_ambient'], label='Ambient', color='black')
ax[0].plot(ppg_data['timestamp'], ppg_data['ppg2_infrared'], label='Infrared', color='red')
# ppg_data['ppg3_red'][1000:].plot(label='Red', color='orange')
# adding title to the plot
ax[0].set_title('Muse S PPG Data')
# adding Label to the x-axis
ax[0].set_xlabel('Time')
# adding legend to the curve
ax[0].legend()


rawData = ppg_data['ppg1_ambient']
ax[0].plot(ppg_data['timestamp'], rawData, label='Ambient', color='black')

# butterworth filter
filterData = butterworth_bandpass(rawData)
# Compute the moving average
window_size = 100
smoothData = np.convolve(filterData, np.ones(window_size)/window_size, mode='valid')
temptime = ppg_data['timestamp'].iloc[int(window_size/2)-1:-int(window_size/2)]


ax[1].plot(ppg_data['timestamp'], filterData, label='Ambient', color='black')
ax1b = ax[1].twinx()
ax1b.plot(temptime, smoothData, label='Ambient', color='red')
ax1b.legend(['Smoothed'])

# Normalize to mean 0 std 1
normData = np.divide(np.subtract(smoothData, np.average(smoothData)), np.std(smoothData))
ax[2].plot(temptime, normData, label='Ambient', color='black')
ax2b = ax[2].twinx()

# Compute the moving average
window_size = 200
smoothData = np.convolve(normData, np.ones(window_size)/window_size, mode='valid')
temptime = temptime.iloc[int(window_size/2)-1:-int(window_size/2)]
ax2b.plot(temptime, smoothData, label='Ambient', color='red')
ax2b.legend(['Smoothed'])

# Derivative to find peaks
diffData = np.diff(smoothData)
ax[3].plot(temptime.iloc[:-1], diffData, label='Ambient', color='red')
signData = np.sign(diffData)
ax[4].plot(temptime.iloc[:-1], signData, label='Ambient', color='red')
signDiff = np.diff(signData)
finalTime = temptime.iloc[:-2]
ax[5].plot(finalTime, signDiff, label='Ambient', color='red')
ax5b = ax[5].twinx()

ax5b.plot(ppg_data['timestamp'], filterData)

peakInds = np.where(signDiff == 2.0)
timestamps_at_index = [finalTime.iloc[i] for i in peakInds]
interBeatIntervals = np.diff(timestamps_at_index)[0]

ax[6].plot(1/interBeatIntervals*60, color='green')
ax[6].set_ylabel('Heart Rate (bpm)')

plt.show()



# so the code itself for computing HRV: without the plotting
rawData = ppg_data['ppg1_ambient']
b, a = butter(4, [.5, 5], btype='bandpass', fs=64)
filterData = filtfilt(b, a, rawData)
window_size = 100
smoothDataA = np.convolve(filterData, np.ones(window_size)/window_size, mode='valid')
temptimeA = ppg_data['timestamp'].iloc[int(window_size/2)-1:-int(window_size/2)]
normData = np.divide(np.subtract(smoothDataA, np.average(smoothDataA)), np.std(smoothDataA))
window_size = 200
smoothDataB = np.convolve(normData, np.ones(window_size)/window_size, mode='valid')
temptimeB = temptimeA.iloc[int(window_size/2)-1:-int(window_size/2)]
diffData = np.diff(smoothDataB)
signData = np.sign(diffData)
signDiff = np.diff(signData)
finalTime = temptimeB.iloc[:-2]
peakInds = np.where(signDiff == 2.0)
timestamps_at_index = [finalTime.iloc[i] for i in peakInds]
interBeatIntervals = np.diff(timestamps_at_index)[0]
HeartRates = 1/interBeatIntervals*60
HRV = np.std(interBeatIntervals*1000)
print('Heart Rate Variability:', np.round(HRV,4))

plt.show()





