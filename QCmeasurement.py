import qcodes as qc
from qcodes.dataset.experiment_container import new_experiment
# from qcodes.dataset.database import initialise_database
from qcodes.dataset.measurements import Measurement
from qcodes.instrument.parameter import Parameter
from qcodes.dataset.data_set import load_by_run_spec
import numpy as np
import time, math
import pandas as pd
from sklearn.cluster import KMeans

from Qextraction import fitlor

import matplotlib.pyplot as plt

from si_prefix import si_format as SI

from tqdm import tqdm_notebook

from gershlab.JJ_data_processing import pbi, bpbi, xy_by_id

from collections import Iterable


class QCmeas():

    def __init__(self, sample, tools=[], folder=r'..\_expdata'):

        self.tools = tools
        self.sample = sample
        self.folder = folder

        self.db_connect()

    def db_connect(self):

        sample = self.sample
        folder = self.folder
        qc.config["core"]["db_location"] = folder + '\Experiments_{}.db'.format(sample)

    def xy_by_id(self, idx):

        self.db_connect()
        out = xy_by_id(idx)

        return out


    def xyz_by_id(self, idx):
        x_raw, y_raw, z_raw = self.xy_by_id(idx)

        x = np.unique( x_raw )
        y = np.unique( y_raw )

        N_x = len( x )
        N_y = len( y )
        
        z = np.reshape( z_raw, (N_x, N_y) )
        return x, y, z

    def pbi(self, idx, isBatch=False, **kwargs):

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
        self.db_connect()
        param, ids = self.xy_by_id(idx)

        index = np.argmin(abs(param - value))

        return ids[index], param[index]

    def get_name(self, ids):

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
    
    ###Added by plamen 07/01/2021 for IQ blob discrimination
    
    def IQ(self,idx,alpha=0.2,kmeans=True,n_blobs=2,**kwargs):
        _, Is, _, Qs = self.xy_by_id(idx)
    
        if kmeans==True:
            fig, ax = plt.subplots(2,figsize=(5,7))
    
            X=np.array(list(zip(Is*1e3,Qs*1e3)))
            kmeans = KMeans(n_clusters=n_blobs, random_state=0).fit(X)
            labels=kmeans.labels_
    
            centers=kmeans.cluster_centers_
            
            df=pd.DataFrame(np.array([Is*1e3,Qs*1e3,labels]).T)
    
            for i in range(n_blobs):
                ax[0].plot(df[df[2]==i][0],df[df[2]==i][1],'.',color='C'+str(i),alpha=alpha)
    
            ax[1].plot(labels,'.',alpha=0.2)
    
            ax[0].plot(centers[:,0],centers[:,1],'x',color='r')
            ax[0].set_xlabel('I, mV')
            ax[0].set_ylabel('Q, mV')
            ax[1].set_ylabel('Blob')
            ax[1].set_yticks([0,1])
            ax[1].set_xlabel('Time step (~0.1s)')
        else:
            fig,ax=plt.subplots()
            ax.plot(Is*1e3, Qs*1e3, '.', alpha = alpha)
            ax.set_xlabel('I, mV')
            ax.set_ylabel('Q, mV')

    def Qfac(self,idx):
        
    
        nu = self.xy_by_id(idx)[0]/1e9
    #    y = 10**((bf.xy_by_id(idx)[1]-background)/10.0)
        y = 10**((self.xy_by_id(idx)[1])/10.0)
        
        fig,ax = plt.subplots(1,1,figsize = (8,5))
        ax.plot(nu, y)
        p = fitlor(nu,y,showfit = True)
        nu = p[2]
      #  Q = p[2]/2/p[3]
        kappa_t = 2*p[3]
        kappa_c = np.sqrt(p[1])*p[3]
        kappa_in = kappa_t-2*kappa_c
        Q_c = nu/kappa_c
        Q_in = nu/kappa_in
        ax.axvline(p[2])
        ax.axvline(p[2]-p[3])
        return print ("Coupling  Q = ", np.round(Q_c),"Internal  Q = ", np.round(Q_in))