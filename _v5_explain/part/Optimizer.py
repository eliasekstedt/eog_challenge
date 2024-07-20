
from datetime import datetime
import pandas as pd
import time

from util.tools_filing import file_it

from skopt import gp_minimize
from skopt.space import Real

class Optimizer:
    def __init__(self, logpath, setup, path_model, hparam, augmentation, device):
        self.device = device
        self.logpath = logpath
        self.logfilepath = f'{self.logpath}gp_log.txt'
        self.setup = setup
        self.path_model = path_model
        self.hparam = hparam
        self.augmentation = augmentation
        self.rank = None

    def objective(self, param):
        def make_line():
            message = f'{score}'
            for p in param:
                message = message + f'\t{p}'
            message = message + f'\t{toc-tic}'
            return message

        for i, key in enumerate(self.setup['key_for_opt']):
            self.augmentation[key] = param[i]
        runtag = f'opt{str(datetime.now())[8:10]}'
        for key in self.hparam.keys():
            runtag += f'_{self.hparam[key]}'

        from part.Workflow import Workflow
        workflow = Workflow(
            path=self.path_model,
            hparam=self.hparam,
            augmentation=self.augmentation,
            tag=runtag,
            device=self.device,
            )
        workflow.initiate_run()
        workflow.get_image_samples()
        workflow.load_data()
        workflow.load_model()
        tic = time.perf_counter()
        workflow.learn_parameters()
        toc = time.perf_counter()
        workflow.evaluate()
        score = workflow.evaluator.score
        file_it(self.logfilepath, make_line())
        self.log_along()
        return score

    def optimize(self):
        def create_header():
            header = f'score'
            for key in self.setup['key_for_opt']:
                header = header + f'\t{key}'
            return header + '\ttime'

        file_it(self.logfilepath, create_header())
        #space = [Real(self.setup['bound'][0], self.setup['bound'][1], name='P.A.R.A.M')]
        space = self.setup['bounds']
        result = gp_minimize(
            self.objective,
            space,
            n_calls=self.setup['n_calls'],
            n_initial_points=min(3, self.setup['n_calls']),
            acq_func='EI',
            x0=None,
            y0=None,
            )
        
        best_param = result.x[0]
        print(f'best param: {best_param}')
        print(f'all?\n{result.x}')
    
    def log_along(self):
        df = pd.read_csv(self.logfilepath, sep='\t')
        df.sort_values('score', inplace=True)
        print(f'call: {len(df)}/{self.setup["n_calls"]}')
        self.rank = df