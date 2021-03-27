# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 10:28:59 2018

@author: kalas
"""

import numpy as np
import os
from scipy.interpolate import UnivariateSpline
import pandas as pd
pd.set_eng_float_format(accuracy=1, use_eng_prefix=True)

##import gershlab.JJformulas as jjf

import OriginExt as O
import matplotlib.pyplot as plt

from IPython.core.display import display, HTML


# import qcodes as qc
# from qcodes.dataset.database import initialise_database
from qcodes.dataset.plotting import plot_by_id, get_data_by_id


from si_prefix import si_format as SI

from scipy.constants import k, h, e

import lmfit as lmf

kB = k
γ = 0.57721566
RQ = h / 4 / e ** 2
Φ0 = h / 2 / e


def update_df(df, index, valdict):
    for k, v in valdict.items():
        df.loc[index, k] = v

    calc_params = calc_jj_param(df)
    df = df.assign(**calc_params)

    return df


def calc_jj_param(df, params=None):
    default_params = ['Rn', 'Ro', 'Ej', 'Ec', 'Iab', 'wp']

    Rn = df['Rn_tot'] / df['N']
    Ro = df['Ro_tot'] / df['N']

    A = df['Ajj']

    Cjj = jjf.Cjj(A)
    Ec = jjf.Ec(Cjj) / 2

    Ej = jjf.EJ_AB(Rn)
    Iab = jjf.Ic(Rn)

    wp = jjf.wpK(Ej, Ec)

    if params is None:
        params = default_params

    calc_params = {k: v for k, v in locals().items() if k in params}

    return calc_params


def show_df(df, sort=None, find=None, which='all', status='all'):
    def make_link(dev_name):

        folder = dev_name.split('N')[0]

        for root, dirs, _ in os.walk('..'):  # goes 2 dir up and walk down until meet the chain folder
            if '!_chains' in dirs:
                path = root + "/!_chains/{}/{}_logbook.ipynb".format(folder, dev_name)

        return '<a target="_blank" href={}>{}</a>'.format(path, dev_name)

    def link_if_measd(df):

        which = df['status'].str.startswith('measd')

        meas_devs = df.loc[which].index

        rename_dict = {dev: make_link(dev) for dev in meas_devs}
        df = df.rename(index=rename_dict)

        return df

    measd = df['status'].str.startswith('measd')

    cdf = df.copy()

    if status is 'measd':
        cdf = cdf[measd]
    elif status is 'fab':
        cdf = cdf[not measd]

    cdf = cdf.sort_index(axis=0)

    cdf = link_if_measd(cdf)

    cdf = cdf.sort_index(axis=1)

    if sort is not None:
        cdf = cdf.sort_values(by=[sort])

    if find is not None:
        cdf = cdf.filter(like=find, axis='index')

    if which is not 'all':
        cdf = cdf[which]

    display(HTML(cdf.to_html(escape=False, classes='table table-hover', header="true")))


#     return cdf


def read_opj_data(cols: tuple, preprint: bool = True):
    app = O.ApplicationSI()
    #     app.Visible = app.MAINWND_SHOW

    app.BeginSession()

    outputData = app.GetWorksheet(app.ActivePage.Name)
    all_data = np.array(outputData, dtype=object)

    sel_data = tuple(all_data[i, :] for i in cols)

    if preprint:
        if len(cols) != 2:
            print('Plotting of multi-d data is not emplemented yet')
        else:
            fig, ax = plt.subplots()
            ax.plot(sel_data[0], sel_data[1])

    app.EndSession()

    return sel_data


def pbi(idx, **kwargs):
    if 'marker' not in kwargs.keys():
        kwargs['marker'] = 'o'

    if 'ls' not in kwargs.keys():
        kwargs['ls'] = 'None'

    if 'interactive' in kwargs.keys():
        interactive = kwargs.pop('interactive')
    else:
        interactive = False

    axes, _ = plot_by_id(idx, **kwargs)

    ax = axes[0]

    if interactive:
        connect_interactive(ax)

    return ax


def batch_plot_by_id(ids, axes=None, labels=None, **kwargs):
    if axes is None:
        fig, axes = plt.subplots()

    if 'marker' not in kwargs.keys():
        kwargs['marker'] = 'o'

    if 'ls' not in kwargs.keys():
        kwargs['ls'] = 'None'

    for i, idx in enumerate(ids):
        if labels is not None:
            label = labels[i]
        else:
            label = ''

        plot_by_id(idx, axes=axes, label=label, **kwargs)

    axes.legend()
    return axes


bpbi = batch_plot_by_id


def connect_interactive(ax):
    global x_clk, y_clk, fitline, text

    def find_nearest(valx, valy, ax):

        linex = ax.lines[-1].get_xdata()
        liney = ax.lines[-1].get_ydata()

        ind = np.argmin((linex - valx) ** 2 + (liney - valy) ** 2)

        x = linex[ind]
        y = liney[ind]

        return x, y

    def onclick(event):
        global x_clk, y_clk, fitline, text

        try:
            for f in fitline:
                f.remove()
            text.set_text('')

        except NameError:
            fitline = None

        x_clk, y_clk = find_nearest(event.xdata, event.ydata, ax)

    def offclick(event):
        global deltaX, deltaY, fitline, text

        x_offclk, y_offclk = find_nearest(event.xdata, event.ydata, ax)

        deltaX = x_offclk - x_clk
        deltaY = y_offclk - y_clk

        dYdX = deltaY / deltaX

        tx = ' dx = {} \n dy = {} \n dy/dx = {}'.format(SI(deltaX), SI(deltaY), SI(dYdX))

        fline, = ax.plot([x_offclk, x_clk], [y_offclk, y_clk], '--k', alpha=1)
        dot1, = ax.plot([x_clk], [y_clk], 'xr', alpha=1)
        dot2, = ax.plot([x_offclk], [y_offclk], 'xr', alpha=1)

        fitline = [fline, dot1, dot2]

        text.set_text(tx)

    text = ax.text(0.05, 0.85, "", bbox=dict(facecolor='white', alpha=0.5), transform=ax.transAxes)

    fig = ax.figure
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('button_release_event', offclick)


def xy_by_id(idx):
    alldata = get_data_by_id(idx)

    out = []
    for dd in alldata:
        for d in dd:
            out.append(d['data'])

    return out


def cut_dxdy(vA0, vB0, dx, dy):
    ind1 = np.where(np.abs(vA0) < dx)
    vA1, vB1 = vA0[ind1], vB0[ind1]

    ind2 = np.where(np.abs(vB1) < dy)
    vA, vB = vA1[ind2], vB1[ind2]

    return vA, vB


def eng_string(x, sig_figs=3, si=True):
    if isinstance(x, str) or x != x:
        return x

    x = float(x)
    sign = ''
    if x < 0:
        x = -x
        sign = '-'
    if x == 0:
        exp = 0
        exp3 = 0
        x3 = 0
    elif abs(x) < 1 and abs(x) > 1e-3:
        exp = 0
        exp3 = 0
        x3 = x
    else:
        exp = int(math.floor(math.log10(x)))
        exp3 = exp - (exp % 3)
        x3 = x / (10 ** exp3)
        x3 = round(x3, -int(math.floor(math.log10(x3)) - (sig_figs - 1)))
        if x3 == int(x3):  # prevent from displaying .0
            x3 = int(x3)

    if si and exp3 >= -24 and exp3 <= 24 and exp3 != 0:
        exp3_text = 'yzafpnum kMGTPEZY'[exp3 // 3 + 8]
    elif exp3 == 0:
        exp3_text = ''
    else:
        exp3_text = 'e%s' % exp3

    return ('%s%s%s') % (sign, x3, exp3_text)


def extract_Isw_R0_by_id(idx, fullIVC=True, dy=50e-6, yoff=0):
    Is, Vs = xy_by_id(idx)

    Vs -= yoff

    Is, Vs = cut_dxdy(Is, Vs, dx=250e-9, dy=dy)

    return extract_Isw_R0(Is, Vs, fullIVC)


def extract_Isw_R0(Is, Vs, fullIVC):
    l = len(Is)
    if l == 0:
        Isw, R0 = np.nan, np.nan
        return Isw, (R0, np.nan)

    if fullIVC:
        Isw = (np.max(Is) - np.min(Is)) / 2
    else:
        Isw = np.max(Is)

    order = Is.argsort()
    Is, Vs = Is[order], Vs[order]

    fit = lambda x, R0, b: x * R0 + b
    fitmodel = lmf.Model(fit)

    R0_guess = np.mean(np.diff(Vs) / np.diff(Is))

    fitparams = fitmodel.make_params(R0=R0_guess, b=0)
    fitmodel.set_param_hint('R0', min=0)

    if fullIVC:
        Is, Vs = Is[l // 5:-l // 5], Vs[l // 5:-l // 5]
    else:
        Is, Vs = Is[:-l // 5], Vs[:-l // 5]

    try:

        result = fitmodel.fit(Vs, fitparams, x=Is, nan_policy='omit')
        R0 = result.params['R0'].value
        errR0 = result.params['R0'].stderr
    except:
        R0 = np.nan
        errR0 = np.nan
    if R0 < 0 or R0 < errR0:
        R0 = np.nan
        errR0 = np.nan

    return Isw, (R0, errR0)


def fit_hist(Isws, EJ=4, Ec=0.05, dIdt=30e-9, bins=21, verbose=False):
    def wpK(EjK, EcK):
        return np.sqrt(8 * EjK * EcK)

    eps = .001

    Isw = Isws

    counts, Ibins = np.histogram(Isw, bins=bins)
    dI = np.mean(np.diff(Ibins))

    SP = np.cumsum(counts) / len(Isw)
    Gamma = np.array([np.log((1 - SP[i]) / (1 - SP[i + 1])) for i in range(len(SP) - 1)]) * dIdt / dI

    Ic = 2 * pi * EJ / Φ0 * kB
    Ic0 = 1.5 * np.max(Isw)
    Ib = Ibins[:-2]

    while abs((Ic - Ic0) / Ic) > eps:
        Ic = Ic0
        wa = wpK(EJ, Ec) * kB / hbar / 2 / pi * (1 - (Ib / Ic) ** 2) ** 0.25

        coeff = (-np.log(2 * pi * Gamma / wa)) ** (2 / 3)
        i = np.isfinite(coeff)

        a, b = np.polyfit(Ib[i][5:], coeff[i][5:], 1)

        Ic0 = -b / a
        Teff = -1 / kB * Φ0 / 2 / pi * 4 * np.sqrt(2) / 3 / np.sqrt(b) / a

    if verbose:
        fix, ax = plt.subplots()
        ax.plot(Ib[i], coeff[i], 'o')
        ax.plot(Ib[i], a * Ib[i] + b)

    return Ic0, Teff


def LV2df(fpath, skiprows=22, header=1, Tstr='Tsample'):
    df = pd.read_csv(filepath_or_buffer=fpath, skiprows=skiprows,
                     header=header, delimiter='\t',
                     usecols=['Tsample', 'IG', 'IS', 'V1', 'SD1', 'V2', 'SD2'])[1:].astype(float)
    dfT = df[Tstr]  # .to_numpy()
    dfB = df['IG']  # .to_numpy()
    dfI = df['IS']  # .to_numpy()
    dfV = df['V1']  # .to_numpy()

    return df, dfI, dfV, dfB, dfT


