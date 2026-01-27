## **Notebook and functions for non linear EEG analysis**

Electrodes time series can be studied in a higher dimensional phase space through time or space embedding.

This allows several metrics to be calculated, that can be helpful to characterize the dinamics of the signals under different stimuli or experimental conditions.

For this reason this package(?) includes modules aimed at streamlining the computation and analysis of these observables. Nothing special about this, just a glorified wrapper.

# **How to use**

In order to increase portability and ease of installation this project runs in its own environment taking advantage of devenv and poetry libraries.

## **Linux install script**

### Requirements

* ```nix``` package manager
* ```devenv``` for developement environment deployment
* ```direnv``` for automatic activation behaviour in the shell
* A running graphical session (Will implement in devenv environment at some point)

### Instructions
  
[ADD LINUX INSTALL SCRIPT]

## **WSL2 install script**

### Requirements

* An up-to-date Windows 11 installation
* WSL2 with NixOS (see [here](https://nix-community.github.io/NixOS-WSL/install.html))

### Instructions

1. Download the install script [nixosWSL.sh](/nixosWSL/nixosWSL.sh)
2. Open NixOS-WSL and enter the following commands:
   ```
      $ cd </path/to/nixosWSL.sh>
      $ ./nixosWSL.sh
   ```
   ```</path/to/nixosWSL.sh>``` is the location of the downloaded script, on WSL2 it will probably be something like ```/mnt/c/Users/<your_username>/Downloads```.

### **Dataset info and observables**

Set dataset information and labels in [init.py](/init.py) file.

This is a central hub for every string and information which is dataset specific. All of this info is stored in a dictionary called by other scripts with the ```get_maind()``` function.

### **Convert and save dataset to MNE file format**

For use across different datasets the data has to be converted using the [toMNE.py](/makedata/toMNE.py) script.

* ```mne.Evoked``` for data averaged across different trials or single trial data (e.g. resting state data);
* ```mne.Epochs``` for multiple trial data.

### **Modules**

After data conversion standard [modules](/modules) can be launched by running their main function in the devenv shell:

```
    $ cd $HOME/nlin-EEG/Compute
    $ python -m modules.<MODULE>
```

### **Speaking of glorified wrappers...**

The [menu.py](/menu.py) coupled with the appropriate graphical backend allows to launch modules and plot results, all within the CLI. Just type ```menu``` in the devenv shell.

