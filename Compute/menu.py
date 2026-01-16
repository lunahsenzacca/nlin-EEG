# Master script for post-processing analysis

# For I/O of modules
import os
import json

# Inquirer library for menu selection
import inquirer as inq

from pprint import pp

from core import obs_path

# Our Very Big Dictionary of standar options and paths
from init import get_maind

maind = get_maind()

## Validators for subject and condition custom entries
def list_validator(ilist: list):

    def validator(answers, current):

        if '#' in current:
            tuple_current = []
            tuple_str = current.split('#')
            for tuple in tuple_str:
                tuple_current.append(tuple.split(','))
        else:
            tuple_current = [current.split(',')]

        for list_current in tuple_current:
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

# exp_name input
def i_exp_name():

    input = [
        inq.List('exp_name',
                 message = 'Choose a preprocessed dataset to work with',
                 choices = [str(i) for i in maind['exp_lb'].keys()]
                )

    ]

    exp_name = inq.prompt(input)['exp_name']

    return exp_name

# sub_list input
def i_sub_list(exp_name: str):

    sub_list = maind[exp_name]['subIDs']

    inputs = [
        [
            inq.List('sub_opt',
                     message = 'Subjects to include',
                     choices = [f'All ({len(sub_list)} subjects)', 'Type...']
            )
        ],[
            inq.Text('sub_list',
                     message = 'Type desired subject IDs separated by \',\'',
                     validate = list_validator(sub_list)
            )
        ]       
    ]

    sub_opt = inq.prompt(inputs[0])['sub_opt']

    if 'All' not in sub_opt: 
        
        c = inq.prompt(inputs[1])['sub_list'].split(',')
        sub_list = list(c)
        print('')

    return sub_list

# conditions input
def i_conditions(exp_name: str):

    conditions = maind[exp_name]['conditions'].keys()


    inputs = [
        [
            inq.List('cond_opt',
                     message = 'Conditions to include',
                     choices = [f'All ({len(conditions)} conditions)', 'Check...']
            )
        ],[
            inq.Checkbox('conditions',
                     message = 'Select conditions [Right] and confirm [Enter]',
                     choices = conditions
            )
        ]
    ]

    cond_opt = inq.prompt(inputs[0])['cond_opt']

    if 'All' not in cond_opt:
       
        c = inq.prompt(inputs[1])['conditions']
        conditions = [maind[exp_name]['conditions'][i] for i in c]

    else:

        conditions = list(maind[exp_name]['conditions'].values())

    return conditions

# ch_list input
def i_channels(exp_name: str):

    pois = maind[exp_name]['pois']

    if os.path.isfile(path = './.tmp/clst.json') == True:

        with open('./.tmp/clst.json', 'r') as f:
            saved_clst = json.load(f)

        opt = [f'All ({len(pois)} channels)', f'Saved Clusters ({len(saved_clst)})', 'Check...', 'Type...']

    else:
        saved_clst = {}
        opt = [f'All ({len(pois)} channels)', 'Check...', 'Type...']

    inputs = [
        [
            inq.List('ch_list_opt',
                     message = 'Channels to include',
                     choices = opt
            )
        ],[
            inq.Checkbox('ch_list',
                     message = 'Select channels [Right] and confirm [Enter]',
                     choices = pois
            )
        ],[
            inq.Text('ch_list',
                     message = 'Type desired Clusters separated by \'#\' and individual Channels by \',\'',
                     validate = list_validator(pois)
            )
        ],[
            inq.List('clst_lb',
                     message = 'Select one of the following',
                     choices = list(saved_clst.keys()),
            )
        ],[
            inq.Text('clst_lb',
                     message = 'Name this group',
            )
        ],[
            inq.Confirm('ch_list_sv', 
                     message = 'Save this cluster selection?',
            )
        ]
    ]

    pois_opt = inq.prompt(inputs[0])['ch_list_opt']

    if 'Ch' in pois_opt:

        ch_list = inq.prompt(inputs[1])['ch_list']

    elif 'Ty' in pois_opt:

        ch_str = inq.prompt(inputs[2])['ch_list']

        if '#' in ch_str:
            ch_list = []
            cl_str = ch_str.split('#')
            for cl in cl_str:
                ch_list.append(cl.split(','))
        else:
            ch_list = ch_str.split(',')
        print('')

    elif 'Sav' in pois_opt:

        clst_lb = inq.prompt(inputs[3])['clst_lb']
        ch_list = saved_clst[clst_lb]

    else:

        clst_lb = 'all'
        ch_list = pois

    if 'Ch' in pois_opt or 'Ty' in pois_opt:

        clst_lb = inq.prompt(inputs[4])['clst_lb']

        print('')

        ch_list_sv = inq.prompt(inputs[5])['ch_list_sv']

        print('')

        if ch_list_sv == True:

            saved_clst[clst_lb] = ch_list

            with open('./.tmp/clst.json', 'w') as f:
                json.dump(saved_clst, f, indent = 2)

    return ch_list, clst_lb

# window input
def i_window(exp_name: str):

    window = maind[exp_name]['window']

    inputs = [
        [
            inq.List('window_opt',
                     message = 'Choose time window [t_min,t_max]s',
                     choices = [f'Full lenght: [{window[0]},{window[1]}]s','Type...']
            )
        ],[
            inq.Text('window',
                     message = 'Type t_min and t_max [s] separated by \',\'',
                     validate = window_validator(window)
            )
        ]
    ]

    window_opt = inq.prompt(inputs[0])['window_opt']

    if 'Ty' in window_opt:

        c = inq.prompt(inputs[1])['window'].split(',')
        window = [float(i) for i in c]

    else:

        window = None


    return window

# avg_trials input
def i_avg_trials():

    input = [
        inq.Confirm('avg_trials',
                    message = 'Average trial data before computation?',
                    default = True
        )
    ]

    avg_trials = inq.prompt(input)['avg_trials']

    print('')

    return avg_trials

# obs_name input
def i_obs_name():

    input = [
        inq.List('obs_name',
                 message = 'Choose module to run',
                 choices = [str(i) for i in maind['obs_lb'].keys()]
        )
    ]

    obs_name = inq.prompt(input)['obs_name']

    return obs_name

# Choose to use defaults parameters for module printing it's values
def i_parameters(obs_name: str):
    
    with open(f'./modules/defaults/{obs_name}.json', 'r') as f:
        d = json.load(f)

        f.seek(0)
        txt = f.read()

    if len(d) > 1:
        print('The selected module has the following default parameters:\n')

        for key in list(d.keys())[1:]:
            pp({key: d[key]}, width = 10)
        print('')

    else:

        print('The selected module has no extra parameters, but you can still choose to type a calculation label\n')

    input = [
        inq.List('default_opt',
                message = 'Select what to do',
                choices = ['Use defaults','Change...']
        )
    ]

    default_opt = inq.prompt(input)['default_opt']

    if 'Ch' in default_opt:

        import tempfile
        import subprocess

        temporary = tempfile.NamedTemporaryFile(mode='w+t', prefix = f'tmp_{obs_name}', suffix = ".json", delete=True, dir = ".tmp")

        n = temporary.name

        with open(n, 'a') as f:
            f.write(txt)

        temporary.close

        subprocess.call(['micro', n])

        with open(n, 'r') as f:
            parameters = json.load(f)

    else:
        with open(f'./modules/defaults/{obs_name}.json', 'r') as f:
            parameters = json.load(f)

    return parameters

# Script launch input with overwriting check
def i_launch_opt(info: dict):

    path = obs_path(exp_name = info['exp_name'], obs_name = info['obs_name'], clst_lb = info['clst_lb'], avg_trials = info['avg_trials'], calc_lb = info['calc_lb'])

    exist = os.path.isdir(path)

    warn = ''

    if exist == True:

        warn = ' (  Overwriting!)'

    input = [
        inq.List('mode_opt',
                message = 'That\'s it! Pleace select one of the followings',
                choices = [f'Compute{warn}', f'Compute & Plot{warn}','Just quit...']
        )
    ]

    mode_opt = inq.prompt(input)['mode_opt']

    quit_opt = False
    plot_opt = False

    if 'quit' in mode_opt:

        quit_opt = True

    elif 'Plot' in mode_opt:

        plot_opt = True

    return plot_opt, quit_opt

# Plot options input
def i_plot_opt():

    defaults = os.listdir('plotting/modules')

    saved = []
    for root, dirs, files in os.walk('../Cargo/results', topdown = False):
        for name in files:
            if name == 'info.json':

                relpath = os.path.relpath(root, start = '../Cargo/results')
                saved.append(relpath)

    for i, d in enumerate(defaults):
        defaults[i] = d.split('.')[0]

    if len(saved) == 0:
        opt = ['Edit plotting defaults...']
    else:
        opt = [f'Choose from saved... ({len(saved)} found)', 'Edit plotting defaults...']

    if os.path.isfile('.tmp/info.json') == True:

        opt = ['Last computation'] + opt

    inputs = [
        [
            inq.List('plot_opt',
                    message = 'What do you want to plot?',
                    choices = opt,
            )
        ],[
            inq.List('saved_opt',
                    message = 'Found the following saved results',
                    choices = saved,
            )
        ],[
            inq.List('default_opt',
                    message = 'Choose defaults to edit',
                    choices = ['basic'] + defaults,
            )
        ],[
            inq.Confirm('plot_def',
                    message = 'Edit plotting options? (Without changing defaults)',
            )
        ]
    ]

    return inputs

# Initial selection, choose between launch and plot
def cmode():

    input = [
        inq.List('mode',
                message = 'Hello! Please select a mode',
                choices = ['Launch module', 'Plot results','Relaunch last selection','Just quit...']
        )
    ]

    mode = inq.prompt(input)['mode']

    return mode

#Launch new computation
def launch():

    ## Prompt for preprocessed dataset name, subjects, conditions, channels and time window

    # Get exp_name
    exp_name = i_exp_name()

    ## Prompt for subjects
    sub_list = i_sub_list(exp_name)

    ## Prompt for conditions
    conditions = i_conditions(exp_name)

    ## Prompt for channels
    ch_list, clst_lb = i_channels(exp_name)

    ## Prompt for time window
    window = i_window(exp_name)

    ## Prompt for trial averaging
    avg_trials = i_avg_trials()

    ## Prompt for module
    obs_name = i_obs_name()

    ## Prompt for module parameters
    parameters = i_parameters(obs_name)
    calc_lb = parameters['calc_lb']

    ## Make a dictionary for run info

    info = {
        'exp_name': exp_name,
        'sub_list': sub_list,
        'conditions': conditions,
        'ch_list': ch_list,
        'window': window,
        'clst_lb': clst_lb,
        'avg_trials': avg_trials,
        'obs_name': obs_name,
        'calc_lb': calc_lb
    }

    ## Prompt for compute or plot or both
    plot_opt, quit_opt = i_launch_opt(info)
    if quit_opt == True:
        keep_open = 1
        return keep_open

    # Add plot opt to info for relaunch option
    info['plot_opt'] = plot_opt

    # Set extra instructions for plotting to false FOR NOW
    info['extra_instructions'] = False

    ## Dump choices in .tmp folder for execution
    os.makedirs('.tmp/modules/', exist_ok = True)

    # Experiment info
    with open('.tmp/info.json', 'w') as f, open('.tmp/last.json') as l:
        json.dump(info, f, indent = 2)
        json.dump(info, l, indent = 2)

    # Script parameters
    with open(f'.tmp/modules/{obs_name}.json', 'w') as f:
        json.dump(parameters, f, indent = 2)

    cmd = f'python -m modules.{obs_name}'

    os.system(cmd)

    ## Launch compiled plotting

    if plot_opt == True:

        cmd = 'python -m plotting.plot'

        os.system(cmd)

    return 0

# Plot some results
def plot():

    # Import plotting options menu
    input = i_plot_opt()

    plot_opt = inq.prompt(input[0])['plot_opt']

    info_path = ''

    if 'Las' in plot_opt:

        info_file = '.tmp/info.json'

    elif 'Choo' in plot_opt:

        keep_choosing = 0
        while keep_choosing == 0:

            info_path = '../Cargo/results/' + inq.prompt(input[1])['saved_opt']
            info_file = info_path + '/info.json'

            with open(info_path + '/variables.json', 'r') as f:
                variables = json.load(f)

            print('     CALCULATION INFO:\n')

            print(f'Module: {variables['obs_name']}')
            if variables['calc_lb'] is not None:
                print(f'Calculation: {variables['calc_lb']}')

            print('')

            for key in list(variables.keys())[2:]:  

                pp({key: variables[key]}, depth = 1, width = 30, compact = True)

            print('')

            keep_choosing = not inq.confirm('Choose another?')

            print('')

    elif 'Edi' in plot_opt:

        import subprocess

        keep_editing = 0
        while keep_editing == 0:

            default_opt = inq.prompt(input[2])['default_opt']

            if default_opt != 'basic':

                default_opt = 'modules/' + default_opt

            subprocess.call(['micro', f'plotting/{default_opt}.json'])

            keep_editing = not inq.confirm('Edit another?', default = True)

            print('')

    if 'Edi' not in plot_opt:

        with open(info_file, 'r') as f:
            info = json.load(f)

        plot_def = inq.prompt(input[3])['plot_def']

        if plot_def == True:

            import subprocess

            info['extra_instructions'] = True

            with open('plotting/basic.json', 'r') as f:

                instructions = json.load(f)

            with open(f'plotting/modules/{info['obs_name'] + '.json'}', 'r') as f:

                obs_instructions = json.load(f)

            for key in obs_instructions.keys():

                instructions[key] = obs_instructions[key]

            with open('.tmp/extra_instructions.json', 'w') as f:

                json.dump(instructions, f, indent = 2)

            subprocess.call(['micro','.tmp/extra_instructions.json'])

        else:

            info['extra_instructions'] = False

        with open('.tmp/info.json', 'w') as f:
            json.dump(info, f, indent = 2)

        cmd = 'python -m plotting.plot'

        os.system(cmd)

    return

if __name__ == '__main__':

    with open('logo.txt', 'r') as file:
        logo = file.read()

    print('\n\n\n',logo,'\n\n\n')

    keep_open = 0
    while keep_open == 0:
        mode = cmode()

        if 'Lau' in mode:
            keep_open = launch()
        elif 'Plo' in mode:
            plot()
        elif 'Relau' in mode:
            # Experiment info
            with open('./.tmp/last.json', 'r') as f:
                info = json.load(f)

            cmd = f'python -m modules.{info['obs_name']}'

            os.system(cmd)

            ## Launch compiled plotting
            if info['plot_opt'] == True:

                cmd = 'python -m plotting.plot'

                os.system(cmd)

        keep_open = not inq.confirm('Keep messing around?')

        print('')
