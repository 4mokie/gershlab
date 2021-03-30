from gershlab.QCmeasurement import *
from gershlab.meas_util import *

from si_prefix import si_format as SI
from tqdm import tqdm_notebook


class MWmeas(QCmeas):
    def __init__(self, sample, tools=[], folder=r'..\_expdata'):

        super().__init__(sample, tools, folder)

    def S21_scan(self, ydevice, fast, slow=None, presets={}, label=''):
        """
            Do scan through list of values(x_list) of any InstrParameter(x_var), meas any (y_var) adta and save to datasaver

            args:
                datasaver: Datasaver to save results of the measurment
                y_var: InstParameter to be measured (tested only on S21.ampl and S21/phase)
                x_var: InstParameter to be scan (indep var)
                x_list: list of values for be scan through
                **kwargs: dict of the InstParameters and its values to add to datasaver (have to be defined in set_meas)
        """
        out = []
        iqmixer = self.tools['iqmixer']
        xdevice = fast[0]
        x_list = fast[1]

        xunit = xdevice.unit
        xlabel = xdevice.label
        x1_list = [0]

        meas = self.set_meas(ydevice, xdevice)

        if slow is not None:
            x1device = slow[0]
            x1_list = slow[1]

            meas = self.set_meas(ydevice, xdevice, x1device)

        x_min = np.min(x_list)
        x_max = np.max(x_list)

        tx_list = tqdm_notebook(x_list, desc='{} to {} {} scan'.format(SI(x_min), SI(x_max), xunit),
                                leave=False)

        self.name_exp(exp_type=label)

        with meas.run() as datasaver:
            # self.set_state(presets)
            with iqmixer.ats.get_prepared(N_pts=8192, N_avg=4000):

                for x1 in x1_list:
                    if slow is not None:
                        x1device.set(x1)

                    for x in tx_list:

                        xdevice.set(x)
                        tx_list.set_description('{} @ {}{}'.format(xlabel, SI(x), xunit))

                        iqmixer.ats.start_capturing()
                        S21 = ydevice.get()

                        out.append(S21)

                        res = [(xdevice, x), (ydevice, S21)]
                        if slow is not None:
                            res = [(x1device, x1)] + res

                        datasaver.add_result(*res)

        return datasaver.run_id
