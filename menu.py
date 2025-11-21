# Master script for post-processing analysis

# For I/O of modules
import os
import json

# Inquirer library for menu selection
import inquirer as inq

# Our Very Big Dictionary of standar options and paths
from init import get_maind

maind = get_maind()

## Validators for subject and condition custom entries
def list_validator(ilist: list):

    def validator(answers, current):

        list_current = current.split(',')
        
        for element in list_current:
            if element not in ilist:
                raise inq.errors.ValidationError('', reason =f'{element} not in dataset!')

        return True

    return validator

## Validator for time window values
def window_validator(window: list):

    def validator(answers,current):
        
        window_current = [float(i) for i in current.split(',')]

        #print(window_current,type(window_current))

        if window_current[0]>window_current[1]:
            raise inq.errors.ValidationError('', reason ='t_min greater than t_max!')

        for w in window_current:

            if w > window[1] or w < window[0]:
                raise inq.errors.ValidationError('', reason = f't = {w}s outside dataset!')

        return True

    return validator

## Inputs

## Module wide dataset inputs
def m_input(n: int, exp_name = 'noise'):

    sub_list = maind[exp_name]['subIDs']
    conditions = maind[exp_name]['conditions'].keys()
    pois = maind[exp_name]['pois']
    window = maind[exp_name]['window']

    input = [

        [
            inq.List('exp_name',
                     message = 'Choose a preprocessed dataset to work with',
                     choices = [str(i) for i in maind['exp_lb'].keys()]
            )
        ],[
            inq.List('sub_opt',
                     message = 'Subjects to include',
                     choices = [f'All ({len(sub_list)} subjects)', 'Type...']
            )
        ],[
            inq.Text('sub_list',
                     message = 'Type desired subject IDs separated by \',\'',
                     validate = list_validator(sub_list)
            )
        ],[
            inq.List('cond_opt',
                     message = 'Conditions to include',
                     choices = [f'All ({len(conditions)} conditions)', 'Check...']
            )
        ],[
            inq.Checkbox('conditions',
                     message = 'Select conditions [->]',
                     choices = conditions
            )
        ],[
            inq.List('pois_opt',
                     message = 'Channels to include',
                     choices = [f'All ({len(pois)} channels)', 'Check...', 'Type...']
            )
        ],[
            inq.Checkbox('pois',
                     message = 'Select channels [->]',
                     choices = pois
            )
        ],[
            inq.Text('pois',
                     message = 'Type desired Channels names separated by \',\'',
                     validate = list_validator(pois)
            )
        ],[
            inq.List('window_opt',
                     message = 'Choose time window [t_min,t_max]s',
                     choices = [f'Full lenght: [{window[0]},{window[1]}]s','Type...']
            )
        ],[
            inq.Text('window',
                     message = 'Type t_min and t_max [s] separated by \',\'',
                     validate = window_validator(window)
            )
        ],[
            inq.Text('clst_lb',
                     message = 'Name this selection',
            )
        ],[
            inq.Confirm('avg_trials',
                     message = 'Average trial data before computation?',
                     default = True
            )
        ],[
            inq.List('obs_name',
                     message = 'Choose module to run',
                     choices = [str(i) for i in maind['obs_lb'].keys()]
            )
        ]
    ]
    return input[n]

# Choose to use defaults parameters for module printing it's values
def d_input(obs_name: str):
    
    with open(f'./modules/defaults/{obs_name}.json', 'r') as f:
        d = json.load(f)

    print('The selected module has the following default parameters:\n')

    for key in d:
        print(key, d[key])
    print('')

    input = [ 
        inq.List('default_opt',
                message = 'Select what to do',
                choices = ['Use default','Type...']
        )
    ]

    return input

# Script launch input with overwriting check
def l_input(exist: bool):

    warn = ''

    if exist == True:

        warn = ' (  Overwriting!)'

    input = [
            inq.List('mode_opt',
                    message = 'That\'s it! Pleace select one of the followings',
                    choices = [f'Compute{warn}', f'Compute & Plot{warn}','Just quit...']
            )
    ]

    return input

# Initial selection, choose between launch and plot
def cmode():

    input = [
            inq.List('mode',
                    message = 'Hello! Please select a mode',
                    choices = ['Launch module', 'Plot results','QuickPlt','Just quit...']
            )
    ]

    mode = inq.prompt(input)['mode']

    return mode

#Launch new computation
def launch():

    print('\nHELLO I\'M A LAUNCH SCRIPT!\n')
    
    ## Prompt for preprocessed dataset name, subjects, conditions, channels and time window

    # Get exp_name
    exp_name = inq.prompt(m_input(0))['exp_name']
    
    # Get sub_list
    sub_opt = inq.prompt(m_input(1, exp_name ))['sub_opt']

    # Opt for a subset
    if 'All' not in sub_opt:
       
        sub_list = list(inq.prompt(m_input(2, exp_name ))['sub_list'].split(','))

    else:

        sub_list = maind[exp_name]['subIDs']

    # Get conditions
    cond_opt = inq.prompt(m_input(3, exp_name ))['cond_opt']

    # Opt for a subset
    if 'All' not in cond_opt:
       
        c = inq.prompt(m_input(4, exp_name ))['conditions']
        conditions = [maind[exp_name]['conditions'][i] for i in c]

    else:

        conditions = list(maind[exp_name]['conditions'].values())

    # Get pois
    pois_opt = inq.prompt(m_input(5, exp_name))['pois_opt']

    # Opt for a subset
    if 'Ch' in pois_opt:
       
        pois = inq.prompt(m_input(6, exp_name ))['pois']

    elif 'Ty' in pois_opt:
        
        c = inq.prompt(m_input(7, exp_name ))['pois'].split(',')
        pois = list(c)

    else:

        pois = maind[exp_name]['pois']

    # Get time window
    window_opt = inq.prompt(m_input(8, exp_name ))['window_opt']

    # Opt for a another
    if 'Ty' in window_opt:

        c = inq.prompt(m_input(9, exp_name))['window'].split(',')
        window = [float(i) for i in c]

    else:

        window = maind[exp_name]['window']

    clst_lb = inq.prompt(m_input(10, exp_name))['clst_lb']

    print('')

    ## Prompt for trial averaging

    avg_trials = inq.prompt(m_input(11, exp_name))['avg_trials']

    ## Make a dictionary for this infos

    info = {
        'exp_name': exp_name,
        'sub_list': sub_list,
        'conditions': conditions,
        'ch_list': pois,
        'window': window,
        'clst_lb': clst_lb,
        'avg_trials': avg_trials,
    }

    ## Prompt for module

    obs_name = inq.prompt(m_input(12))['obs_name']

    default_opt = inq.prompt(d_input(obs_name))['default_opt']

    if 'Ty' in default_opt:
        ## ADD PROMPTING FOR PARAMETERS
        parameters = {}
    else:
        with open(f'./modules/defaults/{obs_name}.json', 'r') as f:
            parameters = json.load(f)

    # Add calc_lb to info as wall
    info['calc_lb'] = parameters['calc_lb']

    ## Prompt for compute or plot or both
    
    '''ADD OVERWRITE CHECK'''

    exist = False

    mode_opt = inq.prompt(l_input(exist))['mode_opt']

    plot_opt = False

    if 'quit' in mode_opt:

        return

    elif 'plot' in mode_opt:

        plot_opt = True

    ## Dump options in .tmp folder for execution
    
    # Experiment info
    with open('./.tmp/last.json', 'w') as f:
        json.dump(info, f, indent = 2)

    # Script parameters
    with open(f'./.tmp/modules/{obs_name}.json', 'w') as f:
        json.dump(parameters, f, indent = 2)
    
    cmd = f'python -m modules.{obs_name}'

    os.system(cmd)

    ## Launch compiled plotting

    if plot_opt == True:
    
        cmd = 'python -m plot'

        os.system(cmd)
    
    return

# Plot some results
def plot():

    print('\n\tPLOTTING\n')
    
    ## Prompt for preprocessed dataset name, and clust_lb of saved data

    # Get exp_name
    exp_name = inq.prompt(m_input(0))['exp_name']

    # Get clust_lb [ADD DEFAULT COLLECTION OF CLUSTERS]
    #clust_lb = inq.prompt(c_input)['clust_lb']
    
    ## Prompt for trial averaging

    avg_trials = inq.prompt(m_input(10, exp_name))['avg_trials']

    ## Prompt for module

    obs_name = inq.prompt(m_input(11))['obs_name']

    ## Prompt for existing calculations for selected

    '''ADD LIST SELECTION'''
    
    ## Print default parameters and prompt for changing

    '''ADD PARAMETER INPUTS'''

    ## PLACEHOLDER, WILL LAUNCH PLOTTING WITH SIMPLE_PLOT
    cmd = 'python -m t_plotting'

    os.system(cmd)

    ## Launch compiled plotting    
    #from plotting import simple_plot
    
    #simple_plot
    
    return

if __name__ == '__main__':

    with open('logo.txt', 'r') as file:
        logo = file.read()

    print('\n\n\n',logo,'\n\n\n')

    mode = cmode()

    if 'Lau' in mode:
        launch()
    elif 'Plo' in mode:
        plot()
    elif 'Quic' in mode:
        os.system('python -m t_plotting')
