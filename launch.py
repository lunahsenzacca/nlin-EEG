# Master script for post-processing analysis

# Inquirer library for menu selection
import inquirer as inq

# Our Very Big Dictionary of standar options and paths
from init import get_maind

maind = get_maind()

## Validators for subject and condition custom entries
def make_validator(ilist: list):

    def validator(answers, current):

        list_current = current.split(',')
        
        for element in list_current:
            if element not in ilist:
                raise inq.errors.ValidationError('', reason =f'{element} not in dataset!')

        return True

    return validator

## Inputs
def m_input(n: int, exp_name = 'noise'):

    sub_list = maind[exp_name]['subIDs']
    conditions = maind[exp_name]['conditions'].keys()
    pois = maind[exp_name]['pois']

    input = [

        [
            inq.List('exp_name',
                     message = 'Choose a preprocessed dataset to work with',
                     choices = maind['exp_lb'].keys()
            )
        ],[
            inq.List('sub_opt',
                     message = 'Subjects to include',
                     choices = [f'All ({len(sub_list)} subjects)', 'Type...']
            )
        ],[
            inq.Text('sub_list',
                     message = 'Type desired subject IDs separated by \',\'',
                     validate = make_validator(sub_list)
            )
        ],[
            inq.List('cond_opt',
                     message = 'Conditions to include',
                     choices = [f'All ({len(conditions)} conditions)', 'Check...']
            )
        ],[
            inq.Checkbox('conditions',
                         message = 'Select conditions [right]',
                         choices = conditions
            )
        ],[
            inq.List('pois_opt',
                     message = 'Channels to include',
                     choices = [f'All ({len(pois)} conditions)', 'Check...', 'Type...']
            )
        ],[
            inq.Checkbox('pois',
                         message = 'Select pois [right]',
                         choices = pois
            )
        ],[
            inq.Text('pois',
                     message = 'Type desired Channels names separated by \',\'',
                     validate = make_validator(pois)
            )
        ]
    ]

    return input[n]


def launch():

    print('\nHELLO I\'M A LAUNCH SCRIPT!\n')
    
    ## Prompt for preprocessed dataset

    # Get exp_name
    exp_name = inq.prompt(m_input(0))['exp_name']
    
    # Get sub_list
    sub_opt = inq.prompt(m_input(1, exp_name = exp_name))['sub_opt']

    # Opt for a subset
    if 'All' not in sub_opt:
       
        sub_list = list(inq.prompt(m_input(2, exp_name = exp_name))['sub_list'].split(','))

    else:

        sub_list = maind[exp_name]['subIDs']

    # Get conditions
    cond_opt = inq.prompt(m_input(3, exp_name = exp_name))['cond_opt']

    # Opt for a subset
    if 'All' not in cond_opt:
       
        c = inq.prompt(m_input(4, exp_name = exp_name))['conditions']
        conditions = [maind[exp_name]['conditions'][i] for i in c]

    else:

        conditions = maind[exp_name]['conditions'].values()

    # Get pois
    pois_opt = inq.prompt(m_input(5, exp_name = exp_name))['pois_opt']

    # Opt for a subset
    if 'Ch' in pois_opt:
       
        pois = inq.prompt(m_input(6, exp_name = exp_name))['pois']

    elif 'Ty' in pois_opt:

        pois = list(inq.prompt(m_input(7, exp_name = exp_name))['pois'].split(','))

    else:

        pois = maind[exp_name]['pois']

    ## Prompt for parameters
    
    ## Prompt for results plotting after computation

    ## Launch compiled script
    
    print(exp_name, sub_opt, sub_list, conditions, pois)

    return

if __name__ == '__main__':

    launch()

