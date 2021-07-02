import qcodes as qc
from qcodes.dataset.experiment_container import new_experiment
# from qcodes.dataset.database import initialise_database
from qcodes.dataset.measurements import Measurement
from qcodes.instrument.parameter import Parameter
from qcodes.dataset.data_set import load_by_run_spec
import numpy as np
import time, math


import matplotlib.pyplot as plt

from si_prefix import si_format as SI

from tqdm import tqdm_notebook

from gershlab.JJ_data_processing import pbi, bpbi, xy_by_id

from collections import Iterable


class QCmeas():
    """
    This is a general class of qcodes measurements (set of tools + measured data), which is used for both MW and DC measurements 

    Args:

        sample: string - sample name, used for locating database file
        tools: dict of Instrument.Parameters, each is supposed to suppots set() and get() functions 
        folder: string containing the path of the folder with all db's
        

    """

    def __init__(self, sample, tools=[], folder=r'..\_expdata'):

        self.tools = tools
        self.sample = sample
        self.folder = folder

        self.db_connect()

    def db_connect(self):
        """
        Call it to make sure that currently we are reading from folder\Experiments_{sample}.db file'
        """

        sample = self.sample
        folder = self.folder
        qc.config["core"]["db_location"] = folder + '\Experiments_{}.db'.format(sample)

    def xy_by_id(self, idx):
        
        """
        Returns two arrays of x and y values of idx runid 
        """


        self.db_connect()
        out = xy_by_id(idx)

        return out


    def xyz_by_id(self, idx):
        
        """
        Returns three arrays of x, y and z values of idx runid. Works only with complete 3d scans  
        """
        
        ## TODO: make it working for unfinished scans
        
        x_raw, y_raw, z_raw = self.xy_by_id(idx)

        x = np.unique( x_raw )
        y = np.unique( y_raw )

        N_x = len( x )
        N_y = len( y )
        
        z = np.reshape( z_raw, (N_x, N_y) )
        return x, y, z

    def pbi(self, idx, isBatch=False, **kwargs):
        
        """
        Plots data from idx runids on the same plot
        
        Args:
            
            idx: int or list of ints corrresponded to runids to plot 
            isBatch: flag showing whether runid contains batch meas, meaning list_of_ids vs list_of_param
            kwargs: keyword arguments for ax.plot() 
            
        """
        self.db_connect()

        if isinstance(idx, Iterable):
            ax = bpbi(idx, **kwargs)
            return ax
         
        if isBatch == True:
            param, ids = self.xy_by_id(idx)
            ax = self.bpbi(ids, **kwargs)

            for idx, p in zip(ids, param):
                print(f'idx : {idx}  param : {p}', "\t")

        else:
            ax = pbi(idx, **kwargs)

        return ax

    def mpbi(self, id_list, titles =[], **kwargs):
        
        """
        Plots data from idx runids on the multiple plots in grid
        
        Args:
            
            id_list:  list of ints corrresponded to runids to plot 
            titles: list of strings of the same size as id_list, contains subplot titles
            kwargs: keyword arguments for ax.plot() 
            
        """
   
        fig=plt.figure(figsize = (10,10), dpi= 80, facecolor='w', edgecolor='k')
        plt.tight_layout()
        
        n = len(id_list)
        print(id_list)
        col = 2
        
        axs = []
        #gs= GridSpec (n//col , col )
    
        for i, _id in enumerate(id_list):

            ax = fig.add_subplot( math.ceil(n/col), col, i+1)
            self.pbi(_id ,axes = ax,**kwargs)
            
            if titles != []:
                ax.set_title(titles[i])
            axs += [ax]
        plt.tight_layout()
    
        return axs

    def find_in_batch(self, idx, value):
        """
        Looks for id in the batch_meas (i.e. runids vs params) with param most close to value 
        

            
        """        
        
        self.db_connect()
        param, ids = self.xy_by_id(idx)

        index = np.argmin(abs(param - value))

        return ids[index], param[index]

    def get_name(self, ids):
        
        """
       Returns list of exp names for list of ids 
            
        """        

        self.db_connect()

        if not isinstance(ids, Iterable):
            ids = [ids]

        names = []
        for idx in ids:
            names.append(str(idx) + ' ' + load_by_run_spec(captured_run_id=idx).exp_name)

        if len(names) == 1:
            names = names[0]

        return names

    def set_meas(self, dep, fast_indep, slow_indep=None):
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
            cleanup = self.cleanup
        except AttributeError:
            print('Please define setup procedure')

        meas = Measurement()
        meas.register_parameter(fast_indep)  # register the fast independent parameter

        if slow_indep is not None:
            meas.register_parameter(slow_indep)  # register the first independent parameter
            meas.register_parameter(dep, setpoints=(slow_indep, fast_indep))
        else:
            meas.register_parameter(dep, setpoints=(fast_indep,))

        meas.add_before_run(setup, args=())
        meas.add_after_run(cleanup, args=())
        meas.write_period = 2

        return meas

    def name_exp(self, exp_type='', **kwargs):
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
        for var, val in kwargs.items():
            name += '__{}= {}'.format(var, SI(val))

        sample_name = "{}".format(self.sample)

        return new_experiment(name=name,
                              sample_name=sample_name)

    def set_state(self, settings):
        """
            Setting of certain values to the Instrument parameters
            args:
                **kwargs:   dict of InstNicknames and values to be set

        """
        if settings != {}:
            for var, val in settings.items():
                self.tools[var].set(val)

    def tool_status(self, which='all'):
        
        """
        Measures (gets) the current value of the tools in self.tools dict
        
        
        Args:
            
            which: str or list of str, indicating which tools parameters to get, 'all' or according to the list 

            
        """

        if which == 'all':

            which_keys = self.tools.keys()

        else:
            which_keys = which
            
        # for key in which_keys:
        #     print(key, self.tools[key].get())
        
        readings = {key: SI(self.tools[key].get()) + self.tools[key].unit
                    for key in which_keys}

        return readings

    def make_label(self):
        label = ''

        readings = self.tool_status()

        for key, val in readings.items():

            if key in ['B', 'T', 'cos']:
                label += str(key) + f' = {val} '

        return label

    def time_scan(self, device, dur=1, dt=0.1):
        """
        PPerforms time measurement, calling device.get() with delay dt, repeating N = dur/dt times
        
 
            
        """

        start = time.time()

        t = Parameter(name='Time', label='Time', unit='s')

        meas = self.set_meas(device, t)
        self.name_exp(exp_type='Time scan')

        t_list = np.arange(0, dur, dt)
        tt_list = tqdm_notebook(t_list)

        with meas.run() as datasaver:
            for _t in tt_list:
                time.sleep(dt)

                dp = device.get()

                current = time.time()
                res = [(t, current - start), (device, dp)]
                datasaver.add_result(*res)

        return datasaver.run_id

    def mock_meas(self, y: tuple, x: tuple, x1=None, label='description'):

        """
        Add x,y data to the qcodes database.

        Args:
            y, x: tuples with the first element being the parameter name
                and the second element is the corresponding array of points.
            x1  : optional, slow variable for 3d data, not emplemented

            label: name of the experiment

        """


        self.setup = lambda: None
        self.cleanup = lambda: None

        xdev, xdat = x
        ydev, ydat = y

        if x1 is not None:
            x1dev, x1dat = x1
            meas = self.set_meas(ydev, xdev, x1dev)
        else:
            meas = self.set_meas(ydev, xdev)
        self.name_exp(exp_type=label)

        with meas.run() as datasaver:

            if x1 is None:
                for _x, _y in zip(xdat, ydat):
                    res = [(xdev, _x), (ydev, _y)]
                    datasaver.add_result(*res)
            else:
                Xdat, X1dat = np.meshgrid(xdat, ydat)
                Xdat = Xdat.flatten()
                X1dat = X1dat.flatten()
                Ydat = ydat.flatten()
                for _x, _x1,  _y in zip(Xdat, X1dat, Ydat):
                    res = [(xdev, _x), (x1dev, _x1), (ydev, _y)]
                    datasaver.add_result(*res)
                
        return datasaver.run_id
