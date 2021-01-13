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

import gershlab.JJformulas as jjf

import OriginExt as O
import matplotlib.pyplot as plt

from IPython.core.display import display, HTML


import qcodes as qc
from qcodes.dataset.database import initialise_database
from qcodes.dataset.plotting import plot_by_id, get_data_by_id


from si_prefix import si_format as SI



def update_df(df, index, valdict):
    
    for k, v in valdict.items():
        
        df.loc[index, k] = v
        
    calc_params = calc_jj_param(df)
    df = df.assign(**calc_params)
        
    return df


def calc_jj_param(df, params = None):
    
    default_params = ['Rn', 'Ro', 'Ej', 'Ec', 'Iab' , 'wp']
    
    Rn  = df['Rn_tot']/df['N']
    Ro  = df['Ro_tot']/df['N']
    
    A   = df['Ajj']
    
    Cjj = jjf.Cjj( A )
    Ec  = jjf.Ec(Cjj)/2
    
    Ej  =  jjf.EJ_AB( Rn )
    Iab =  jjf.Ic( Rn )
    
    wp  = jjf.wpK(Ej, Ec)
    
    if params is None:
        params = default_params
    
    calc_params = { k:v for k,v in locals().items() if k in params }
    
    return calc_params


    
def show_df(df, sort = None, find = None, which = 'all'):

    def make_link(dev_name):

        folder = dev_name.split('N')[0]

#         path = "./{}/{}_logbook.ipynb".format(folder,dev_name)
        path = "../{}/{}_logbook.ipynb".format(folder,dev_name)

        return '<a target="_blank" href={}>{}</a>'.format(path,dev_name)


    def link_if_measd(df):
        
        which = df['status'].str.startswith('measd')

        meas_devs = df.loc[which].index
        
        rename_dict = { dev : make_link(dev) for dev in meas_devs }
        df = df.rename( index  = rename_dict )
        
        return df


    
    cdf = df.copy()
    
    cdf = cdf.sort_index(axis = 0)
    
    cdf = link_if_measd(cdf)
    
    cdf = cdf.sort_index(axis = 1)

    if sort is not None:
        cdf = cdf.sort_values(by = [sort])
        
    if find is not None:
        cdf = cdf.filter(like = find, axis = 'index')
        
    if which is not 'all':
        cdf = cdf[which]
    
    display(HTML(cdf.to_html(escape = False, classes='table table-hover', header="true")))


def read_opj_data(cols : tuple, preprint : bool = True):
    app = O.ApplicationSI()
    app.BeginSession()
    
    outputData = app.GetWorksheet(app.ActivePage.Name)
    all_data = np.array(outputData, dtype=object)
    
    sel_data = tuple( all_data[i,:] for i in cols )
    
    if preprint:
        if len(cols) != 2 :
            print('Plotting of multi-d data is not emplemented yet')
        else:
            fig, ax = plt.subplots()
            ax.plot(sel_data[0], sel_data[1])
            
            
    app.EndSession()
            
    return sel_data 


def pbi(idx, **kwargs):
    
    if 'marker'  not in kwargs.keys():
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


def batch_plot_by_id(ids, ax = None, labels = None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    if 'marker'  not in kwargs.keys():
        kwargs['marker'] = 'o'
        
    if 'ls' not in kwargs.keys():
        kwargs['ls'] = 'None'
        
    for i, idx in enumerate(ids):
        if labels is not None:
            label = labels[i]
        else:
            label = ''

        plot_by_id(idx, axes = ax, label = label, **kwargs)
        
    ax.legend()
    return ax

bpbi = batch_plot_by_id


def connect_interactive(ax):
    global x_clk, y_clk, fitline , text


    def find_nearest(xval, yval, ax):

        linex = ax.lines[-1].get_xdata()    
        liney = ax.lines[-1].get_ydata()

        ind = np.argmin ( (liney - yval)**2 + (linex - xval)**2 )
        
        print(ind)

        x = linex[ind]
        y = liney[ind]

        return  x, y

    def onclick(event):
        global x_clk, y_clk, fitline, text 

        try: 
            for f in fitline:
                f.remove()
            text.set_text('')

        except NameError:
            fitline = None

        x_clk, y_clk = find_nearest( event.xdata, event.ydata, ax )

    def offclick(event):
        global deltaX, deltaY, fitline, text

        x_offclk, y_offclk = find_nearest(  event.xdata, event.ydata, ax )

        deltaX = x_offclk -  x_clk
        deltaY = y_offclk -  y_clk

        dYdX = deltaY/deltaX 

        tx = ' dx = {} \n dy = {} \n dy/dx = {}'.format(SI(deltaX), SI(deltaY),SI(dYdX) )

        fline, = ax.plot([ x_offclk , x_clk], [y_offclk ,  y_clk], '--k', alpha = 1)
        dot1,  =ax. plot([  x_clk], [ y_clk], 'xr', alpha = 1)
        dot2,  =ax. plot([  x_offclk], [y_offclk], 'xr', alpha = 1)

        fitline = [fline, dot1, dot2]
        
        text.set_text(tx)
    

    text=ax.text(0.05, 0.85, "", bbox = dict(facecolor='white', alpha=0.5), transform=ax.transAxes)
    
    fig = ax.figure
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('button_release_event', offclick)


    
def xy_by_id(idx):
    alldata = get_data_by_id(idx)
    
    x = alldata[0][0]['data']
    y = alldata[0][1]['data']
    
    return x,y


def extract_Isw_R0_by_id (idx, dy = 50e-6, yoff = 0):

    Is,Vs = xy_by_id(idx)
    
    Vs -= yoff

    Is,Vs = cut_dxdy(Is, Vs, dx = 250e-9 ,dy = dy)    
    return extract_Isw_R0 (Is,Vs)

def cut_dxdy(vA0, vB0, dx,dy):
    
    ind1 = np.where(np.abs(vA0) < dx )
    vA1, vB1 = vA0[ind1], vB0[ind1]

    ind2 = np.where(np.abs(vB1) < dy )
    vA, vB = vA1[ind2], vB1[ind2]

    return vA, vB


def offsetRemove(X,Y, offX, offY):
    
    return X - offX, Y - offY 


def eng_string( x, sig_figs=3, si=True):
    
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
    elif abs (x) < 1 and abs (x) > 1e-3  :
        exp = 0
        exp3 = 0
        x3 = x
    else:
        exp = int(math.floor(math.log10( x )))
        exp3 = exp - ( exp % 3)
        x3 = x / ( 10 ** exp3)
        x3 = round( x3, -int( math.floor(math.log10( x3 )) - (sig_figs-1)) )
        if x3 == int(x3): # prevent from displaying .0
            x3 = int(x3)

    if si and exp3 >= -24 and exp3 <= 24 and exp3 != 0:
        exp3_text = 'yzafpnum kMGTPEZY'[ exp3 // 3 + 8]
    elif exp3 == 0:
        exp3_text = ''
    else:
        exp3_text = 'e%s' % exp3

    return ( '%s%s%s') % ( sign, x3, exp3_text)

def extract_Isw_R0 (Is,Vs):
    
        if len( Is )== 0 or len( Vs )== 0 :
            Isw, R0 = np.nan, np.nan
            
            print('no points in cut range')
            return Isw, R0
        
        Isw = ( np.max(Is) - np.min(Is) ) /2
        
        order = Is.argsort()
        
        Is, Vs = Is[order], Vs[order]
        
        n_sl = np.where( (Is > 0)  ) and np.where (Is < 300e-12)

        if len( Vs[n_sl] )== 0 :
            R0 = np.nan
            return Isw, R0
        
#         R0, b = np.polyfit (  Is[n_sl] , Vs[n_sl], 1 )
        R0 = np.mean(np.diff(Vs[n_sl])) / np.mean(np.diff(Is[n_sl]))        
        
        if R0 < 0:
            R0 = np.nan
        return Isw, R0
