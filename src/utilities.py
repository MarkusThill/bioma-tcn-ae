import os
import numpy
import tensorflow
import matplotlib.pyplot as plt

def select_gpus(gpu_list):
    if type(gpu_list) != list:
        gpu_list = [gpu_list]
    sel_gpus = ",".join(str(gpu) for gpu in gpu_list)
    os.environ["CUDA_VISIBLE_DEVICES"] = sel_gpus
    print("selected GPUs:", sel_gpus)
    
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    tensorflow.config.experimental.set_memory_growth(gpus[0], True)
    
def slide_window(df, window_length, verbose = 1):
    orig_TS_list = []
    X_list = []
    series = df.copy()
    for i in series.columns.values: # loop through all input dimensions
        s = series[i]
        s2 = roll_fast(s.values, window_length)
        X_list.append(s2)
    X = numpy.dstack((X_list))
    if verbose > 2:
        print("X.shape:", X.shape)
    return X
    
def roll_fast(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# Computes the squared Mahalanobis distance of each data point (row)
# to the center of the distribution, described by cov and mu. 
# (or any other point mu).
# If the parameters cov and mu are left empty, then this function 
# will compute them based on the data X.
def mahalanobis_distance(X, cov=None, mu=None):
    if mu is None:
        mu = numpy.mean(X, axis=0)
    if cov is None:
        cov = numpy.cov(X, rowvar = False)
    try:
        inv_cov = numpy.linalg.inv(cov)
    except numpy.linalg.LinAlgError as err:
        print("Error, probably singular matrix!")
        inv_cov = numpy.eye(cov.shape[0])
    
    X_diff_mu = X - mu
    M = numpy.apply_along_axis(lambda x: 
                    numpy.matmul(numpy.matmul(x, inv_cov), x.T) ,1 , X_diff_mu)
    return M


def get_anomaly_windows(is_anomaly):
    # add a zero at the beginning and end of the sequence and look for the edges of the anomaly windows
    edges = numpy.diff(numpy.concatenate([[0],is_anomaly,[0]])).nonzero()[0]
    return edges.reshape((-1,2)) + numpy.array([0,-1])


def plot_results(data, anomaly_score, pl_range = None, plot_signal = False, plot_anomaly_score = True):
    #anomaly_score = results["anomaly_score"]
    series = data["series"] 
    extend_window = 0
    my_alpha = 0.15
    cols = ["value"]
    if pl_range is None:
        pl_range = (0,series.shape[0])
        extend_window = 10 # extend anomaly window, just to see something in the plot
        my_alpha = 0.4
    plt.figure(figsize=(25,8))
    if plot_signal:
        plt.plot(series[cols].values, zorder=1)
        plt.ylim((series[cols].values.min(),series[cols].values.max()));
    if plot_anomaly_score:
        plt.plot(anomaly_score, 'b-', zorder=2)

    real_anoms = get_anomaly_windows(data['is_anomaly'])
    
    for i in real_anoms:
        plt.axvspan(i[0]-extend_window,i[1]+extend_window, ymin=0.0, ymax=50, alpha=my_alpha, color='red')
        
    ignorable_win = get_anomaly_windows(data['is_ignoreable'])
    for i in ignorable_win:
        plt.axvspan(i[0],i[1], ymin=0.0, ymax=50, alpha=my_alpha, color='yellow')
        
    # Choose the threshold as the smallest possible value that would not produce a false positive
    anoms = ((data["is_anomaly"]==1) | data["is_ignoreable"])
    extd = sorted(list(set(numpy.where(anoms)[0]) | set(numpy.where(anoms)[0] + 250) | set(numpy.where(anoms)[0] - 250)))
    idx = numpy.array(extd)
    idx = idx[(idx>=0) & (idx<anomaly_score.shape[0])]
    ignore = (numpy.ones(anomaly_score.shape[0]) == 1)
    ignore[idx] = False
    artifical_threshold = anomaly_score[ignore].max()
    if plot_anomaly_score:
        plt.axhline(y=artifical_threshold, xmin=0.0, xmax=650000, color='r')

    plt.xlim(pl_range);
    plt.draw()