
from datetime import datetime
import pandas as pd
import time

from Universal.Tools import file_it

from skopt import gp_minimize
from skopt.space import Real

class Optimizer:
    def __init__(self, logpath, setup, path, hparam):
        self.logpath = logpath
        self.logfilepath = f'{self.logpath}gp_log.txt'
        self.setup = setup
        self.path = path
        self.hparam = hparam
        self.rank = None

    def objective(self, param):
        self.hparam[self.setup['hpq']] = param[0]
        runtag = f'WF1_{str(datetime.now())[8:10]}'
        runtag = runtag + '_gp_cr'
        from WF1_classifier.Flow import Workflow
        workflow = Workflow(path=self.path, hparam=self.hparam, tag=runtag)
        workflow.load_data()
        workflow.initiate_run()
        tic = time.perf_counter()
        workflow.learn_parameters()
        toc = time.perf_counter()
        workflow.evaluate()
        cm = workflow.evaluator.cmatrix
        accuracy = (cm[0,0] + cm[1,1])/cm.sum()
        file_it(self.logfilepath, f'{accuracy}\t{param[0]}\t{round(toc-tic)}')
        self.plot_along()
        return 1-accuracy

    def optimize(self):
        file_it(self.logfilepath, f'accuracy\tparam\ttime')
        space = [Real(self.setup['bound'][0], self.setup['bound'][1], name='P.A.R.A.M')]
        result = gp_minimize(self.objective, space, n_calls=self.setup['n_calls'], acq_func='EI', n_random_starts=min(5, self.setup['n_calls']))
        best_param = result.x[0]
        print(f'best param: {best_param}')
    
    def plot_along(self):
        df = pd.read_csv(self.logfilepath, sep='\t')
        df.sort_values('accuracy', inplace=True)
        from WF1_classifier.parts.Tools import gp_plot
        gp_plot(df, self.setup['bound'], self.logpath)
        print(f'call: {len(df)}/{self.setup["n_calls"]}')
        self.rank = df
