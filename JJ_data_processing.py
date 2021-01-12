# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 10:28:59 2018

@author: kalas
"""

import numpy as np
import os

import gershlab.JJformulas as jjf

import OriginExt as O
import matplotlib.pyplot as plt

from IPython.core.display import display, HTML


import qcodes as qc
from qcodes.dataset.database import initialise_database
from qcodes.dataset.plotting import plot_by_id, get_data_by_id
from scipy.interpolate import UnivariateSpline

from si_prefix import si_format as SI
import pandas as pd
pd.set_eng_float_format(accuracy=1, use_eng_prefix=True)


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
#     return cdf


def read_opj_data(cols : tuple, preprint : bool = True):
    app = O.ApplicationSI()
#     app.Visible = app.MAINWND_SHOW
    
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
        
#         n = len(Is)
#         n_min, n_max = np.int(n/3), np.int(2*n/3)
#         n_sl = slice(n_min, n_max)

        n_sl = np.where( (Is > 0)  ) and np.where (Is < 300e-12)
#         n_sl = np.where( abs(Is) < 200e-12 ) 
        
        if len( Vs[n_sl] )== 0 :
            R0 = np.nan
            return Isw, R0
        
#         R0, b = np.polyfit (  Is[n_sl] , Vs[n_sl], 1 )
        R0 = np.mean(np.diff(Vs[n_sl])) / np.mean(np.diff(Is[n_sl]))        
        
        if R0 < 0:
            R0 = np.nan
#         R0 = np.mean(np.diff(Vs[n_sl])) / np.mean(np.diff(Is[n_sl]))
        
        return Isw, R0

def load_by_key(exp, key, val):
        
    ind =   np.argmin ( np.abs( exp[key] - val ))
    return ind
    
def plot_by_key(exp, key, val, ax = None,**kw):
   
    ind =   np.argmin ( np.abs( exp[key] - val ))
    
    I, V = exp['Is'][ind], exp['Vs'][ind]
    
#     I = I - V/1.3e9

    if ax == None:
        fig, ax = plt.subplots()
        
    ax.plot( I, V, 'o', label = 'T = {:2.0f} mK, {} = {:1.2f} '.format( exp['T']/1e-3, key, exp[key][ind] ) , **kw)
    ax.legend()   
    
    return I, V
    
    
    
    
    
    
    
def get_R0(x,y, x0 = 0, VERBOSE = False):
    
    if len(y) < 5:
        return np.nan
    
    sort_ind = np.argsort(x)
    x, y = x[sort_ind], y[sort_ind]

    _, unique_ind = np.unique(x, return_index=True)
    x, y = x[unique_ind], y[unique_ind]

    x,y = remove_jumps(x,y)

    spl = UnivariateSpline(x, y)
    spl.set_smoothing_factor(0.5)
    
    diff = np.diff( spl(x) )/ np.diff( x )
    
    i_x0 = np.argmin( abs( x - x0) )
    
    R0 = diff[i_x0] 
    
    if VERBOSE:
        fig, ax = plt.subplots()
        
        ax.plot(x,y, 'o', ls = '')
        ax.plot(x, spl(x))
        
        ax2 = ax.twinx()
        ax2.plot(x[:-1], diff, c = 'k')
    
    return R0
    
    




def V_func(I,V, val):
    out = []
    for x in np.nditer(val):
        out = np.append (out,  V[np.argmin(abs(I-x))])
    return out



def remove_jumps(x,y):
    
    
    if len(y) < 3:
        return x,y
    
    y_out = []
    i_off = 1
    Voff = 0
    for i in range(len(y) ):

        steps = abs(np.diff(y))

        if i > i_off and i < len(y) - 2 :

            avg_diff = np.mean(steps[:i-1])
            if steps[i-1] > 10* avg_diff:
                Voff -= steps[i-1]



        y_out.append(y[i]+Voff)


    y = np.array(y_out)
    x = x
    return x,y



def XYEqSp(Xarr, Yarr, step):
    outX = []
    outY = []

    if len(Xarr) == 0 :
        outX, outY = 0, 0
    else:    
        n = int((np.max(Xarr) - np.min(Xarr)) // step)    

        for i in range(n):
            outX = np.append( outX, V_func(Xarr, Xarr, np.min(Xarr) + i*step)  )
            outY = np.append( outY, V_func(Xarr, Yarr, np.min(Xarr) + i*step)  )

    return outX, outY


    
