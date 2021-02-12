import qcodes as qc
from qcodes.dataset.experiment_container import (Experiment,
                                                 load_last_experiment,
                                                 new_experiment)
from qcodes.dataset.database import initialise_database
from qcodes.dataset.measurements import Measurement

import numpy as np
from gershlab.QCmeasurement import *
from gershlab.meas_util import *
from tqdm import tqdm, tqdm_notebook

from collections.abc import Iterable


class JJmeas(QCmeas):

    def __init__(self, sample, tools=[], folder=r'..\_expdata'):
        super().__init__(sample, tools, folder)

    def set_param(self, **kwparams):

        for k, v in kwparams:
            self.k = v
            self.exps.k = v

    def stabilize_I(self, amp):

        I = self.tools['I']

        x = np.arange(100)
        i_s = amp * np.exp(- x / 20) * np.cos(x / 3)

        for i in i_s:
            I.set(i)
            time.sleep(0.1)

    def IVC_cust(self, i_list, Ioff=0, Vthr=1, dt=0.1, N_avg=1, label=''):

        I = self.tools['I']
        V = self.tools['V']

        V.meas_Voff()
        Voff = V.Voff

        self.stabilize_I(amp=np.max(i_list))

        meas = self.set_meas(V, I)

        ti_list = tqdm_notebook(i_list, leave=False)

        if label == '':
            label = self.make_label()

        self.name_exp(exp_type=label)
        with meas.run() as datasaver:
            for i in ti_list:

                I.set(i + Ioff)
                time.sleep(dt)

                is_vs = [[I.get(), V.get()] for _ in range(N_avg)]

                ir, v = np.mean(is_vs, axis=0)
                res = [(I, ir - Ioff), (V, v - Voff)]

                ti_list.set_description('I = {}A'.format(SI(ir)))

                if abs(v - Voff) > Vthr:
                    datasaver.add_result((I, np.nan), (V, np.nan))
                    break

                datasaver.add_result(*res)

        self.stabilize_I(amp=i)
        run_id = datasaver.run_id
        self.last_runid = run_id

        return run_id

    def IVC_udu(self, amp, stp, **kwargs):

        i_list = udu_list(amp, stp)
        run_id = self.IVC_cust(i_list, **kwargs)

        return run_id

    def IVC_fwd(self, amp, stp, **kwargs):

        i_list = np.concatenate([np.linspace(0, amp, int(amp / stp)),
                                 np.linspace(amp, 0, int(amp / stp / 10) + 1),
                                 np.linspace(0, -amp, int(amp / stp)),
                                 np.linspace(-amp, 0, int(amp / stp / 10) + 1)])

        run_id = self.IVC_cust(i_list, **kwargs)
        return run_id

    def meas_hist(self, N, i_list, dt=0, Vthr=30e-6, label=''):

        I = self.tools['I']
        V = self.tools['V']

        V.meas_Voff()
        Voff = V.Voff

        Isw = Parameter(name='Swithching current', label='Swithching current',
                        unit='A')

        t = Parameter(name='Time', label='Time',
                      unit='s')

        meas = self.set_meas(Isw, t)

        if label == '':
            label = 'hist ' + self.make_label()

        self.name_exp(exp_type=label)

        to = time.time()

        ti_list = tqdm_notebook(i_list, leave=False)
        with meas.run() as datasaver:

            V = Voff

            for i in ti_list:

                if (V - Voff) < Vthr:
                    break

                time.sleep(dt)
                I.set(i)
                ti_list.set_description('I = {}A'.format(SI(i)))

                V = V.get()

            res = [(Isw, i), (t, time.time() - t0)]
            datasaver.add_result(*res)

        run_id = datasaver.run_id
        return run_id

    def cos_to_B(self, cos):

        for frust in ['ZF', 'FF']:
            if not hasattr(self, frust):
                raise Exception(f'Please indicate value of {frust}!')

        ZF = self.ZF
        FF = self.FF

        return np.arccos(cos) * (2 * (FF - ZF) / np.pi) + ZF

    def B_to_cos(self, B):

        for frust in ['ZF', 'FF']:
            if not hasattr(self, frust):
                raise Exception(f'Please indicate value of {frust}!')

        ZF = self.ZF
        FF = self.FF

        return np.cos(np.pi / 2 * (B - ZF) / (FF - ZF))

    def Bscan_f(self, f, B_list=None):

        B = self.tools['B']
        idx = Parameter(name='runid', label='#')

        meas = self.set_meas(idx, B)

        if label == '':
            label = 'Bscan ids list ' + self.make_label()

        self.name_exp(exp_type=label)

        tB_list = tqdm_notebook(B_list)

        with meas.run() as datasaver:

            for b in tB_list:
                B.set(b)
                tB_list.set_description('B = {}A'.format(SI(b)))

                yield self

                res = [(runid, idx), (B, b)]
                datasaver.add_result(*res)

    def Bscan(self, B_list=None, cos_list=None):

        B = self.tools['B']
        T = self.tools['T']

        if B_list is None and cos_list is None:
            raise Exception('Please specify either B or cos list!')

        elif B_list is not None and cos_list is not None:
            raise Exception('Please choose either B or cos list!')

        elif cos_list is not None:
            B_list = self.cos_to_B(cos_list)

        tB_list = tqdm_notebook(B_list)

        for b in tB_list:
            B.set(b)
            tB_list.set_description('B = {}A'.format(SI(b)))

            yield self

            runids.append(self.last_runid)

        B.set(0)

    def Isw_by_id(self, ids, fullIVC=True, **kwargs):

        self.db_connect()

        if fullIVC:
            k = 1
        else:
            k = 2

        if not isinstance(ids, Iterable):
            ids = [ids]

        Ics = np.array([k * extract_Isw_R0_by_id(idx, **kwargs)[0] for idx in ids])

        if len(Ics) == 1:
            Ics = Ics[0]

        return Ics

    def Tscan(self, T_list):

        htr = self.tools['htr']
        T8 = self.tools['T']

        tolerT8 = 0.02
        chkrepeat = 20
        chkperiod_sec = 2

        tT_list = tqdm_notebook(T_list)
        for t in tT_list:

            htr.set(t)
            print('ramping T8 to {}K...'.format(SI(t)))
            time.sleep(30)

            count_T = 0
            while count_T < chkrepeat:
                T_now = T8.get()
                if (1 - tolerT8) * t <= T_now <= (1 + tolerT8) * t:
                    count_T += 1

                    time.sleep(chkperiod_sec)
                    print('{}'.format(SI(T_now), end=" "))
                elif count_T >= 1:
                    count_T -= 1

            print('T is set')
            tT_list.set_description('T = {}K'.format(SI(t)))

            yield self
