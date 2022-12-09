import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file into a pandas dataframe
# The first row of the file is assumed to be the header row
# Use the values in the header row as the column names
df = pd.read_csv("VoyageRecording.csv")

# Create a new dataframe containing only the rows where the 'type' column is equal to 'eeg'
new_df = df[df['type'] == 'eeg']

# Remove the specified columns from the new dataframe
new_df = new_df.drop(columns=['gyroX', 'gyroY', 'gyroZ', 'accelerometerX', 'accelerometerY', 'accelerometerZ', 'ppg1_ambient', 'ppg2_infrared', 'ppg3_red'])
print(new_df.columns)
print(new_df.iloc[1,36:68])



# Create a line plot of the timestamp and eeg1_0 columns
chans = ['e1', 'e2', 'e3', 'e4']
freqs = list(range(0,125,4))

fig, ax = plt.subplots(1,4)
for iwin in range(1000):
    ax[0].plot(freqs, new_df.iloc[iwin, 3:35])
    ax[1].plot(freqs, new_df.iloc[iwin, 35:67])
    ax[2].plot(freqs, new_df.iloc[iwin, 67:99])
    ax[3].plot(freqs, new_df.iloc[iwin, 99:131])

plt.show()


# fig, ax = plt.subplots(1, 4)
# for ichan, chan in enumerate(chans):
#     for freq in freqs:
#         if freq == 0:
#             name = chan + '_' + str(freq)
#             ax[ichan].plot(new_df['timestamp'], new_df[name])

# # # Show the plot
# plt.show()