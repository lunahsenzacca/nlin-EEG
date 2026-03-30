with open('.tmp/last.json','r') as f:

    info = json.load(f)

path = c.obs_path(exp_name = info['exp_name'],
                  obs_name = info['obs_name'],
                  clst_lb = info['clst_lb'],
                  avg_trials = info['avg_trials'],
                  calc_lb = info['calc_lb'])

path = path.replace('../Cargo/results/','')

RES, info = load(info['obs_name'], path = path.replace('../Cargo/results/',''))
