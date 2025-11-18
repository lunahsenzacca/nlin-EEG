# Master script for post-processing analysis

# For I/O of modules
import os

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

    ## Prompt for trial averaging

    avg_trials = inq.prompt(m_input(10, exp_name))['avg_trials']

    ## Prompt for module

    obs_name = inq.prompt(m_input(11))['obs_name']
    
    ## Prompt for module parameters
    
    ## Prompt for results plotting and saving after computation

    ## Launch compiled script
    
    cmd = f'python -m modules.{obs_name}'

    os.system(cmd)
    
    ## Launch compiled plotting

    #if plot_opt == True:
    
    #    from plotting import simpleplot
    
    #print(exp_name, sub_list, conditions, pois, window, avg_trials)

    return

if __name__ == '__main__':

    launch()

