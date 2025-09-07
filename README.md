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

**Dataset info and observables**

Set dataset information and labels in init.py[\init.py] file.

This is a central hub for every string and information which is dataset specific. All of this info is stored in a dictionary called by other scripts with the ```get_maind()``` function.

**Convert and save dataset to MNE evoked file format**

For use across different datasets the data has to be converted using the toMNE.py[\toMNE.py] script.

Data can be saved using two methods:

```avg_trials = True``` : Save data averaged across repeated trials;


```avg_trials = False``` : Save data keeping each repeated trial.


