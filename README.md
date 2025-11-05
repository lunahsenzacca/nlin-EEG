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
```

### **Dataset info and observables**

Set dataset information and labels in [init.py](/init.py) file.

This is a central hub for every string and information which is dataset specific. All of this info is stored in a dictionary called by other scripts with the ```get_maind()``` function.

### **Convert and save dataset to MNE evoked file format**

For use across different datasets the data has to be converted using the [toMNE.py](/toMNE.py) script.

Data can be saved using two methods:

```avg_trials = True``` : Save time series averaged across same condition trials;

```avg_trials = False``` : Save time series for each trial.

### **Compute observables**

After data conversion the following observables can be computed running a ```.py``` script in the appropriate conda enviroment:


*  $S_{m}\left(\Delta_{ij}||i-j| = \delta t\right)$ : Spacetime Separation Plot for different embedding dimensions $m$ and realtive times $\delta t$;

$\rightarrow$ ```python -m separation```

*  $R_{m}\left(r|i,j\right)$ : Recurrence Plot for different embedding dimensions $m$ and scales $r$;

$\rightarrow$ ```python -m recurrence```

*  $C_{m}(r)$ : Correlation Sum for different embedding dimensions $m$ and scales $r$;

$\rightarrow$ ```python -m corrsum```

*  $\nu_{m}(r)$ : Correlation Exponent for different embedding dimensions $m$ and scales $r$ deriving results from $C(m,r)$;

$\rightarrow$ ```python -m correxp```

*  $\lambda_{m}$ : Largest Lyapunov Exponent for different embedding dimensions $m$.

$\rightarrow\$ ```python -m llyap```


