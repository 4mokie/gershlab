import qcodes as qc
from qcodes.dataset.experiment_container import (Experiment,
                                                 load_last_experiment,
                                                 new_experiment)
from qcodes.dataset.database import initialise_database
from qcodes.dataset.measurements import Measurement
from qcodes.instrument.parameter import Parameter

import numpy as np
import time

from si_prefix import si_format as SI

from tqdm import tqdm, tqdm_notebook

from  gershlab.JJ_data_processing import pbi, bpbi

class QCmeas():
    
    def __init__(self, sample, tools = [] ,  folder = r'..\_expdata'): 
        
        self.tools = tools
        self.sample = sample
        self.folder = folder
        
#         self.setup = lambda : None
        
        self.db_connect()
        
        
    def db_connect(self):
        
        sample = self.sample 
        folder = self.folder 
        qc.config["core"]["db_location"] = folder +'\Experiments_{}.db'.format(sample)
            
    def pbi(self, idx, **kwargs):
        
        self.db_connect()
        ax = pbi( idx, **kwargs)
        
        return ax
    
    def bpbi(self, ids, **kwargs):
        
        self.db_connect()
        ax = bpbi( ids, **kwargs)
        
        return ax

    def set_meas(self, dep, fast_indep, slow_indep = None):
        """
           Regestration of the parameters and setting up before/after run procedure

           args:
                dep: InstParameter to be regestered as dependent var 
                fast_indep: InstParameter to be regestered as fast independent var 
                slow_indep: InstParameter to be regestered as second independent var (if any) 
                setup: Procedure to be done before run
                cleanup: Procedure to be done after run

            returns:
            meas: Measurement() object

        """ 
        
        try:
            setup = self.setup
        except AttributeError:
            print('Please define setup procedure')
            
        try:
            cleanup = self.setup
        except AttributeError:
            print('Please define setup procedure')
            
            
        meas = Measurement()
        meas.register_parameter( fast_indep )  # register the fast independent parameter

        if slow_indep is not None:
            meas.register_parameter( slow_indep )  # register the first independent parameter
            meas.register_parameter( dep , setpoints = ( slow_indep,  fast_indep ) ) 
        else:
            meas.register_parameter( dep, setpoints = ( fast_indep , ))

            
            
        meas.add_before_run(setup, args=())    
        meas.add_after_run(cleanup, args=())    
        meas.write_period = 2

        return meas


    def name_exp(self, exp_type = '', **kwargs):
        """
            Set name of the experiment
            args:
                ext_type[str]:   any label for experiment
                **kwars[eg Bfield = 50e-6]    Dict of the parameter name and its value to add 
            returns:
                new_experimenrt object

        """
        
        self.db_connect()
        
        name = '{:s}_'.format(exp_type)
        for var , val in kwargs.items():
            
            
            name += '__{}= {}'.format( var,  eng(val) )

        sample_name = "{}".format(self.sample)

        return new_experiment( name = name,
                               sample_name = sample_name )
    
    
    def tool_status(self, which = 'all'):
        
        if which == 'all':
            
            which_keys = self.tools.keys()

        else :
             which_keys = which
        
        readings = {key : SI(self.tools[key].get())+ self.tools[key].unit 
                    for key in which_keys }

        return readings
    
    
    def make_label(self):
        label = ''
        
        readings = self.tool_status()
        
        for key, val in readings.items():
            
            if key in ['B', 'T', 'cos']:
                label += str(key) + f' = {val} '
            
        return label
        
    
    
    def time_scan (self, device, dur = 1, dt= 0.1):
   
    
        t = Parameter(name = 'Time', label = 'Time', unit = 's')
        
        meas = self.set_meas(device, t)
        self.name_exp( exp_type = 'Time scan')

        t_list = np.arange(0 ,dur, dt)
        tt_list = tqdm_notebook(t_list)

        with meas.run() as datasaver:

            for _t in tt_list:

                time.sleep(dt)

                dp = device.get()

                res = [( t, _t ), ( device, dp  )]
                datasaver.add_result(*res) 

        return datasaver.run_id 
    
    
    def mock_meas(self, y : tuple, x : tuple, x1 = None , label = 'description'):
        
        """
        Add x,y data to the qcodes database.
        
        Args:
            y, x: tuples with the first element being the parameter name
                and the second element is the corresponding array of points.
            x1  : optional, slow variable for 3d data, not emplemented

        """
        
        if x1 is not None:
            print ('2d writing is not emplemented yet') 
        
        self.setup   = lambda: None
        self.cleanup = lambda: None
    
        xdev, xdat = x
        ydev, ydat = y
        
        meas = self.set_meas(ydev, xdev)
        self.name_exp( exp_type = label)

        with meas.run() as datasaver:

            for _x, _y in zip(xdat, ydat):

                res = [( xdev, _x ), ( ydev, _y  )]
                datasaver.add_result(*res) 

        return datasaver.run_id 

        
