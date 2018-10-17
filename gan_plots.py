import pylab
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import interp1d
import random


def set_plot_environ(facecolor='#ffffff', grid_color=0.):
    plt.rcParams['font.sans-serif'] = ['Calibri', 'Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']
    plt.rcParams['axes.facecolor'] = facecolor
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 26
    plt.rcParams['axes.titlesize'] = 28
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['axes.titlepad'] = 12
    plt.rcParams['axes.edgecolor'] = '#cccccc'  

    plt.grid(color=(grid_color, grid_color, grid_color, 0.05), linestyle='--', linewidth=0.2)


def dist_chart(elements, x_label, y_label, title, f_name,
               colors=('#362cb2ff', '#c32148ff', '#fd5e0fff', '#228b22ff', '#C1A004ff', '#b710aaff'), bins=50,
               color_start=0, directories=None, scaleX=None):

    set_plot_environ()
    if scaleX is not None:
        pylab.xlim(scaleX)

    for i, e in enumerate(elements):
        plt_data = e['data']

        clr = colors[i + color_start]

        _ = sns.distplot(plt_data, color=clr, kde_kws={"lw": 6, }, label=e['label'], bins=bins)
        if scaleX is not None:
            pylab.xlim(scaleX)

    legnd = plt.legend(fontsize=24, loc=2)
    for text in legnd.get_texts():
        plt.setp(text, color='b')

    ax = plt.gca()
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:0.1f}'.format(x) for x in vals])


    pylab.xlabel(x_label)
    pylab.ylabel(y_label)
    if title is not None:
        pylab.title(title)
    figure = plt.gcf()
    height = 10
    aspect = 1920. / 1080.
    figure.set_size_inches(aspect * height, height)

    pylab.savefig(directories[0] + f_name, format='png', bbox_inches='tight', pad_inches=0.5, dpi=300)
    if len(directories) > 1:
        pylab.savefig(directories[1] + f_name, format='png', bbox_inches='tight', pad_inches=0.5, dpi=30)
    plt.close()


def cumul_return_plot(data_sets, f_name=None, dir_name='../images/pngs/', labels=['High volatility', 'Low volatility'],
                      legend_loc=2):
    set_plot_environ(facecolor='#000000', grid_color=1.)
    colors = ('#FF00FF66', '#7DF9FF66')

    smooth_points = 500
    for d, mat in enumerate(data_sets):
        x = np.arange(0, mat.shape[1])
        x_new = np.linspace(x.min(), x.max(), smooth_points)

        mat += 1.
        mat_cr = np.cumprod(mat, axis=1)

        first_col = mat_cr[:, 0]
        temp = mat_cr.T / first_col.T
        mat_cr_scaled = temp.T

        mat_cr_scaled *= 100.

        for y_ct, y in enumerate(mat_cr_scaled):
            f = interp1d(x, y, kind='quadratic')
            y_smooth = f(x_new)
            lbl = labels[d] if y_ct == 0 else None
            sns.lineplot(data=y_smooth, ci=None, color=colors[d], label=lbl)

    pylab.xlabel('52-week simulated time period')
    pylab.ylabel('Simulated cumulative returns')

    legnd = plt.legend(fontsize=24, loc=legend_loc)
    for text in legnd.get_texts():
        plt.setp(text, color='w')

    pylab.xlim([0, smooth_points])
    figure = plt.gcf()  
    height = 10
    aspect = 1920. / 1080.
    figure.set_size_inches(aspect * height, height)
    if f_name is not None:
        pylab.savefig(f'{dir_name}{f_name}', format='png', bbox_inches='tight', pad_inches=0.5, dpi=300)
        plt.close()
    else:
        pylab.show()


def scatter(elements, x_label='x label', y_label='y label', title='title', save_file=None, show=False,
            directories=None):
    set_plot_environ()

    pylab.xlabel(x_label)
    pylab.ylabel(y_label)
    if title is not None:
        pylab.title(title)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    for i, e in enumerate(elements):
        sns.regplot(x=e['x'], y=e['y'], fit_reg=False, scatter=True, label=e['label'], scatter_kws={
            'color': e['color'], 'alpha': e['alpha'], 's': e['size'], 'edgecolors': '#000000ff'
            })

    if elements[0]['label'] is not None:
        legnd = plt.legend(fontsize=24, loc=3)
        for text in legnd.get_texts():
            plt.setp(text, color='b')

    figure = plt.gcf() 
    height = 10
    aspect = 1920. / 1080.
    figure.set_size_inches(aspect * height, height)

    if save_file is not None:
        pylab.savefig(directories[0] + save_file, format='png', bbox_inches='tight', pad_inches=0.5, dpi=300)
        pylab.savefig(directories[1] + save_file, format='png', bbox_inches='tight', pad_inches=0.5, dpi=30)

    if show:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        pylab.show()

    plt.close()
