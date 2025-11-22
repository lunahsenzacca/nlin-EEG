## **Notebook and functions for non linear EEG analysis**

Electrodes time series can be studied in a higher dimensional phase space through time or space embedding.

This allows several metrics to be calculated, that can be helpful to characterize the dinamics of the signals under different stimuli or experimental conditions.

# **How to use**

Install the required packages in a conda enviroment:
```
  -python = 3.12
  -numpy
  -matplotlib
  -pandas
  -scipy
  -cython
  -tqdm
  -pip:
    -teaspoon
    -mne
    -inquirer (Only for menu.py script)
```

### **Dataset info and observables**

Set dataset information and labels in [init.py](/init.py) file.

This is a central hub for every string and information which is dataset specific. All of this info is stored in a dictionary called by other scripts with the ```get_maind()``` function.

### **Convert and save dataset to MNE file format**

For use across different datasets the data has to be converted using the [toMNE.py](/makedata/toMNE.py) script.

    -```mne.Evoked``` for data averaged across different trials or single trial data (e.g. resting state data);
    -```mne.Epochs``` for multiple trial data

### **Modules**

After data conversion standard [modules](/modules) can be launched by running their main function in the appropriate conda enviroment.

```
    cd <<nlin-EEG-FOLDER>>
    python -m modules.<<MODULE_NAME>>
```


