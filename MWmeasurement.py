from gershlab.QCmeasurement import QCmeas
from gershlab.meas_util import *

from si_prefix import si_format as SI
from tqdm import tqdm_notebook
from qcodes.instrument.parameter import Parameter

import numpy as np
import time

class MWmeas(QCmeas):
    def __init__(self, sample, tools=[], folder=r'..\_expdata'):

        super().__init__(sample, tools, folder)

    def S21_scan(self, ydevice, fast, slow=None, Navg = 1, N_card_avg = 4000, presets={}, label=''):
        """
            Do scan through list of values(x_list) of any InstrParameter(x_var), meas any (y_var) adta and save to datasaver

            args:
                datasaver: Datasaver to save results of the measurment
                y_var: InstParameter to be measured (tested only on S21.ampl and S21/phase)
                x_var: InstParameter to be scan (indep var)
                x_list: list of values for be scan through
                **kwargs: dict of the InstParameters and its values to add to datasaver (have to be defined in set_meas)
        """
        # out = []
        iqmixer = self.tools['iqmixer']
        
        xdevice = self.tools.get(fast[0],fast[0]) 
        x_list  = fast[1]

        xunit = xdevice.unit
        xlabel = xdevice.label
        tx1_list = [0]

        meas = self.set_meas(ydevice, xdevice)

        if slow is not None:
            x1device = self.tools.get(slow[0], slow[0])
            x1_list  = slow[1]
            x1unit   = x1device.unit
            
            x1_min = np.min(x1_list)
            x1_max = np.max(x1_list)
            tx1_list = tqdm_notebook(x1_list, 
                                     desc=f'{SI(x1_min)} to { SI(x1_max)} {x1unit} scan')
            
            meas = self.set_meas(ydevice, xdevice, x1device)

        x_min = np.min(x_list)
        x_max = np.max(x_list)
        # print(x_min, x_max)

        self.set_state(presets)
        self.name_exp(exp_type=label+str(self.tool_status(presets.keys())))

        with meas.run() as datasaver:
            
            self.set_state(presets)
            time.sleep(1)
            
            with iqmixer.ats.get_prepared(N_pts=8192, N_avg=N_card_avg):
           # with iqmixer.ats.get_prepared(N_pts=2048, N_avg=N_card_avg):

                for x1 in tx1_list:
                    if slow is not None:
                        x1device.set(x1)
                     #   time.sleep(.300)
                        time.sleep(.0100)

                    tx_list = tqdm_notebook(x_list, desc=f'{xlabel} = {SI(x_min)} to {SI(x_max)} {xunit} scan',
                                            leave=False)
                    for x in tx_list:

                        xdevice.set(x)
                        time.sleep(.010)
                        tx_list.set_description('{} @ {}{}'.format(xlabel, SI(x), xunit))

                        
                        S21 = []
                        for _ in range(Navg):
                            iqmixer.ats.start_capturing()
                            S21 += [ydevice.get()]

                        # out.append(S21/Navg)

                        res = [(xdevice, x), (ydevice, np.mean(S21, axis = 0))]
                        if slow is not None:
                            res = [(x1device, x1)] + res

                        datasaver.add_result(*res)

        return datasaver.run_id
    
    
    def single_pump_sweep(self, ydevice, fpump_list, Nsw ):
        
        rnd = np.random.random
        buf = np.zeros((len(fpump_list), Nsw))
        iqmixer = self.tools['iqmixer']
        
        with iqmixer.ats.get_prepared(N_pts = 8192, N_avg = 400):
            tNsw = tqdm_notebook(range(Nsw), desc = 'pump scan',  leave = False)                
            # tNsw.set_description(f'pump scan @ {SI()}Hz'.format( eng(f_target) ))
            
            for j in tNsw:
    
                time.sleep(rnd()*0.5 + .100)
                
                for i,fpump in enumerate(fpump_list):
                    self.tools['fpump'].set( fpump )
                    time.sleep(0.003)
    
                    iqmixer.ats.start_capturing() 
                    
                    try:
                        S21 = ydevice.get()    
                    except RuntimeError:
                        print(-1-i)
                        break
                    
                    
                    buf[i,j] = S21
    
        # print(buf)
        scan_avg = np.mean(buf, axis = 0 )
        # 
        # result = buf / Nsw
        result = np.mean(buf - scan_avg, axis = 1)
        return result
    
    def pump_sweep_ftargs(self, ydevice, fpump_list, Nsw ,slow, ftarg_list,
                   presets = {}, label = ''):
        
        xdevice = self.tools['fpump']
        
        if slow[0] == 'time':
            start = time.time()

            t = Parameter(name='Time', label='Time', unit='s')
            
            x1device = t
            
            self.tools['time'] = t
            
            x1_list  = slow[1]
            
            tx1_list = tqdm_notebook(x1_list)
            
            x1device.set = lambda dt :  time.sleep (x1_list[1] -  x1_list[0]  )
            x1device.get = lambda :  time.time() - start
            
        else:    
           
            x1device = self.tools.get(slow[0], slow[0])
            x1_list  = slow[1]
            tx1_list = tqdm_notebook(x1_list)
                
        meas = self.set_meas(ydevice, xdevice, x1device)
        self.set_state(presets)
        self.name_exp(exp_type=label+str(self.tool_status(presets.keys())))

        with meas.run() as datasaver:
            
            for i, x1 in enumerate(tx1_list):
                self.set_state(presets)
                self.set_state({slow[0] : x1,
                                'fprobe': ftarg_list[i]})
                
                print(f'frequency is set to {ftarg_list[i]}')
                                
                S21vspump = self.single_pump_sweep(ydevice, fpump_list, Nsw)                
        
                x1value = x1device.get()
                for j,fpump in enumerate(fpump_list):
                    datasaver.add_result(( x1device, x1value),
                                          ( xdevice, fpump ),
                                          ( ydevice, S21vspump[j] ))
        
    def make_ftargs_by_id(self, idx, p_axis, p_list, shift_from_min = 0):
        
        
        data = self.xyz_by_id(idx)
        amps = data[2]
        params = data[p_axis ]
        fprobes = data[(p_axis + 1) % 2 ]
                
        ftargs = []

        for p in p_list:
            
            i = np.argmin(abs(params - p))
            amp = amps[i]
            j_min = np.argmin(amp)
                        
            ftargs += [fprobes[j_min] - shift_from_min]
            
        return np.array(ftargs)
    
    def make_probe_list(self, res, span, N_pts):
        """
        Making linspace array around given point
        
        args:
            res: central point
            span: span around central point
            N_pts: number of points
           
        """ 
        return np.linspace(res - span/2, res + span/2, N_pts)