## **Notebook and functions for non linear EEG analysis**

Electrodes time series can be studied in a higher dimensional phase space through time or space embedding.

This allows several metrics to be calculated, that can be helpful to characterize the dinamics of the signals under different stimuli or experimental conditions.

**How to use**

Install required packages:
```
  -python = 3.12
  -matplotlib
  -scipy
  -scikit.learn
  -pip:
    -teaspoon
    -mne
```
Set your own workflow path changing the `path` variables in the files.

From backward masking data: 

- [Results](/BW_CS) for the [correlation sum script](/correlation.py)
- [Graphs](/averageCsum.png) for averaged results between conditions
