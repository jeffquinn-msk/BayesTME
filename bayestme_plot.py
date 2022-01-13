import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def st_plot(data, pos, cmap='BuPu', v_min=None, v_max=None, norm=None, layout='s', unit_dist=10, x_y_swap=False, invert=[0, 0], 
            name='st_plot', subtitles=None, fontsize=20, show=True, save=False):
    if x_y_swap:
        pos = pos[::-1]
    if len(data.shape) > 1:
        n_plots = data.shape[0]
    else:
        n_plots = 1
        data = data[None]
    if not isinstance(v_min, (list, np.ndarray)):
        v_min = [v_min for i in range(n_plots)]
    if not isinstance(v_max, (list, np.ndarray)):
        v_max = [v_max for i in range(n_plots)]
    if not isinstance(subtitles, (list, np.ndarray)):
        subtitles = [subtitles] + [None for i in range(n_plots-1)]
    subplots_adj = 1/n_plots
    x_axis_distance = pos[0].max() - pos[0].min() + 2
    y_axis_distance = pos[1].max() - pos[1].min() + 2
    dpi = plt.rcParams["figure.dpi"]
    text_width = np.sqrt(plt.rcParams['ytick.labelsize']) 
    text_ratio = text_width / (x_axis_distance*unit_dist + text_width)
    st_ratio = (subplots_adj - (text_ratio+0.05)) / subplots_adj
    if layout == 's':
        scatter_size = unit_dist**2
        fig_width = (x_axis_distance*unit_dist / dpi) / st_ratio
        fig_height = y_axis_distance*unit_dist / dpi
    elif layout == 'H':
        scatter_size = (2*unit_dist)**2
        fig_width = x_axis_distance*np.sqrt(3)*unit_dist/dpi / st_ratio
        fig_height = y_axis_distance*unit_dist/dpi
    fig = plt.figure(figsize=(fig_width*n_plots, fig_height))
    for i in range(n_plots):
        stbox = [0+i*subplots_adj, 0, st_ratio*subplots_adj, 1]
        cbbox = [(st_ratio+0.01+i)*subplots_adj, 0.05, 0.04*subplots_adj, 0.9]
        stframe = plt.axes(stbox)
        cbframe = plt.axes(cbbox)
        img = stframe.scatter(pos[0], pos[1], c=data[i], cmap=cmap, s=scatter_size, vmin=v_min[i], vmax=v_max[i], norm=norm, marker=layout, linewidths=0)
        stframe.set_xlim(pos[0].min()-1, pos[0].max()+1)
        stframe.set_ylim(pos[1].min()-1, pos[1].max()+1)
        stframe.axis('off')
        if subtitles[i]:
            stframe.set_title(subtitles[i])
        if invert[0]:
            stframe.invert_xaxis()
        if invert[1]:
            stframe.invert_yaxis()
        plt.colorbar(img, cax=cbframe)
    if save:
        print('Plot saved in {}'.format(save))
        plt.savefig(save+'{}.pdf'.format(name))
    if show:
        plt.show()
        return fig
    plt.close()

def plot_spots(ax, data, pos, rgb=np.array([1, 0, 0]), cmap=None, discrete_cmap=None, s=15, v_min=None, v_max=None, invert_x=True, invert_y=True, norm=None):
    if cmap is None and discrete_cmap is None:
        n_nodes = data.shape[0]
        plot_color = np.zeros((n_nodes, 4))
        plot_color[:, :3] = rgb[None]
        plot_color[:, 3] = data/data.max()
        ax.scatter(pos[0], pos[1], color=np.array([0, 0, 0, 0.02]), edgecolors=None, linewidths=None, s=s)
        img = ax.scatter(pos[0], pos[1], color=plot_color, edgecolors=None, linewidths=0, s=s)
    elif discrete_cmap is not None:
        for i, value in enumerate(np.unique(data)):
            idx = np.argwhere(data == value).flatten()
            img = ax.scatter(pos[0, idx], pos[1, idx], color=discrete_cmap[value], edgecolors=None, linewidths=0, s=s)
    else:
        img = ax.scatter(pos[0], pos[1], c=data, cmap=cmap, s=s, vmin=v_min, vmax=v_max, norm=norm)
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()
    return img


def scatter_pie(dist, pos, size, ax=None):
    cumsum = np.cumsum(dist)
    cumsum = cumsum/ cumsum[-1]
    pie = [0] + cumsum.tolist()
    x_pos, y_pos = pos
    for i, (r1, r2) in enumerate(zip(pie[:-1], pie[1:])):
        if r2 - r1 > 0:
            angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
            x = [0] + np.cos(angles).tolist()
            y = [0] + np.sin(angles).tolist()
            xy = np.column_stack([x, y])
            ax.scatter(x_pos, y_pos, marker=xy, s=size, facecolor='C{}'.format(i))
    return ax










