import matplotlib as mpl
import numpy as np
import seaborn as sns


def plot_custom_stripplot(ax, data, labels, showmedians=False, showquartiles=False, show_whiskers=False, colors=None,
                           *, no_trim=False, only_trim_up=False, use_log_scale=False, stripplot_kw={}):
    # See https://matplotlib.org/stable/gallery/statistics/customized_violin.html#sphx-glr-gallery-statistics-customized-violin-py
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    data = [np.array(sorted(x)) for x in data]

    assert not (no_trim and only_trim_up)
    if no_trim:
        whiskers_min = [np.nanmin(x) for x in data]
        whiskers_max = [np.nanmax(x) for x in data]
        medians = [np.nanmedian(x) for x in data]
        quartile1=None
        quartile3=None

        data = [x[~np.isnan(x)] for x in data]
    elif only_trim_up:
        quartile1, medians, quartile3 = np.nanpercentile(data, [25, 50, 75], axis=1)
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
        whiskers_min = [np.nanmin(x) for x in data]

        # Filtering equivalent to boxplot, see https://matplotlib.org/stable/gallery/statistics/boxplot_vs_violin.html
        data = [x[np.logical_and(x >= whiskers_min[i], x <= whiskers_max[i])] for i, x in enumerate(data)]
    else:
        quartile1, medians, quartile3 = np.nanpercentile(data, [25, 50, 75], axis=1)
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

        # Filtering equivalent to boxplot, see https://matplotlib.org/stable/gallery/statistics/boxplot_vs_violin.html
        data = [x[np.logical_and(x >= whiskers_min[i], x <= whiskers_max[i])] for i, x in enumerate(data)]

    if use_log_scale:
        data = [np.log10(x) for x in data]
        whiskers_min = np.log10(whiskers_min)
        whiskers_max = np.log10(whiskers_max)
        medians = np.log10(medians)
        if quartile1 is not None:
            quartile1 = np.log10(quartile1)
        if quartile3 is not None:
            quartile3 = np.log10(quartile3)

#     for i, x in enumerate(data):
#         sns.stripplot(
#             ax=ax,
#             data=x,
# #             x=i,
#             c=colors[i],
#         )
    sns.stripplot(
        ax=ax,
        data=data,
        palette=colors,
        jitter=0.4,
        **stripplot_kw,
    )

    if use_log_scale:
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
        ymin, ymax = ax.get_ylim()
        tick_range = np.arange(np.floor(ymin), ymax)
        ax.yaxis.set_ticks(tick_range)
        ax.yaxis.set_ticks([np.log10(x) for p in tick_range for x in np.linspace(10 ** p, 10 ** (p + 1), 10)], minor=True)


    inds = np.arange(0, len(medians))
    if showmedians:
        ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    if showquartiles:
        ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    if show_whiskers:
        ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)


def plot_cust_boxplot(ax, box_data, labels, *, colors, meanline=False, showmeans=False, showfliers=False):
    bplot = ax.boxplot(box_data,
                       positions=np.arange(len(box_data)),
                       labels=labels,
                       meanline=meanline, showmeans=showmeans,
                       whis=99, showfliers=showfliers,
                       notch=False)

    for i, box in enumerate(bplot['boxes']):
        box.set_color(colors[i])
        box.set_linewidth(2)
    for i, box in enumerate(bplot['medians']):
        box.set_color('k')
        box.set_linewidth(1)
    for i, box in enumerate(bplot['means']):
        box.set_color(colors[i])
        box.set_linewidth(2)
