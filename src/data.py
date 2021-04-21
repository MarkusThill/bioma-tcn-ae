import numpy
import pandas
import utilities
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class Data:    
    def __init__(self,
                 data_folder = "../data/MGAB/",
                 series_length = 100000,
                 num_anomalies = 10,
                 min_anomaly_distance = 2000,
                 window_length = 1050, 
                 window_stride = 5,
                 error_window_length = 128,
                 input_columns = ["value"],
                 scale_method = "StandardScaler", # [None, "MinMaxScaler", "StandardScaler"]
                 training_split = 1.0
                ):
        
        self.data_folder = data_folder
        self.error_window_length = error_window_length
        self.series_length = series_length
        self.num_anomalies = num_anomalies
        self.min_anomaly_distance = min_anomaly_distance
        self.window_length = window_length
        self.window_stride = window_stride
        self.input_columns = input_columns
        self.scale_method = scale_method
        self.training_split = training_split
        
    def create_chaotic_time_series(self, ts_id, verbose=0, return_plots_data=False):
        # parameters
        # mg_tau = 18
        # mg_n= 10.0
        # mg_beta = .25
        # mg_gamma = .1
        # mg_history = .9
        # step_size= 1
        series = numpy.load(self.data_folder + "/mg" + str(ts_id) + ".npy")
        seed = ts_id # for the moment like this

        numpy.random.seed(seed)
        anomaly_positions = numpy.random.randint((self.series_length-self.min_anomaly_distance*self.num_anomalies)/self.num_anomalies, size = self.num_anomalies)+self.min_anomaly_distance
        anomaly_positions = int(0.95*self.series_length) - numpy.cumsum(anomaly_positions)
        anomaly_positions.sort()

        new_series, plots = self.set_anomalies(series = series, anomalies_idx = anomaly_positions, verbose = verbose)
        if verbose > 1: 
            new_series.head()
            print("new_series.shape", new_series.shape)

        new_series["value"] = new_series["value"] + numpy.random.uniform(low=-0.01, high=0.01, size=new_series["value"].shape) # random uniform
        numpy.random.seed(None) # reset the seed, otherwise also other components like Keras (TF) use this seed
        if return_plots_data:
            return new_series.iloc[0:self.series_length], plots
        return new_series.iloc[0:self.series_length]
    
    
    def set_anomalies(self, series = numpy.array([1,2,3]), anomalies_idx = [], verbose = 0):
        min_length_cut = 100 # It does not make sense to make a trivial cut (e.g. cut segment of length 1)
        max_sgmt = 100 # maximum segment length in which we want to find the closest value
        anomaly_window = 200 # window size of anomaly window which we put around the anomaly
        order_derivative = 3 # until which derivative do we want to compare the similarity of the points (0->only value, 1-> value and 1st derivative, ...)

        real_anomalies_idx = list()
        anomalies_idx = numpy.sort(anomalies_idx)

        plots = []
        for ad_idx in anomalies_idx:
            gradients = list()
            gradients.append(series)
            for i in range(order_derivative):
                gradients.append(numpy.gradient(gradients[-1], axis=-1))
            if len(series.shape) > 1:
                all_grads = numpy.hstack(gradients).T
            else:
                all_grads = numpy.stack(gradients)
            #print("all_grads.shape", all_grads.shape)

            if verbose > 2:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(20,6))
                plt.plot(all_grads[0][ad_idx-50:ad_idx+50], label='mg1')
                plt.plot(all_grads[1][ad_idx-50:ad_idx+50], label='d/dx mg1')
                plt.plot(all_grads[2][ad_idx-50:ad_idx+50], label='d^2 / dx^2 mg1')
                plt.legend()

            mod_window_mean = all_grads.mean(axis=-1, keepdims=True)
            mod_window_std = all_grads.std(axis=-1, keepdims=True)


            # xxxxxxxxxxx [window1] xxxxx [window2] xxxxxxxxxxx
            mod_window1 = all_grads[:, ad_idx:ad_idx + max_sgmt]
            mod_window2 = all_grads[:, ad_idx + max_sgmt + min_length_cut: ad_idx + min_length_cut + 2*max_sgmt]

            # for both window, measure all pairwise similarities for all points (euclidian distance)
            similarities = numpy.sqrt((numpy.apply_along_axis(lambda x: mod_window1 - x[:, numpy.newaxis], axis=0, arr=mod_window2)**2).sum(axis=0))

            best_points = numpy.argwhere(similarities == numpy.min(similarities))[0] # (index first window, index 2nd window)

            idx_first = best_points[0] + ad_idx
            idx_second = best_points[1] +  ad_idx + max_sgmt + min_length_cut

            # Cut out a segment
            new_series = numpy.concatenate( [series[:idx_first], series[idx_second:]], axis=0 )
            idx_anomaly = idx_first
            real_anomalies_idx.append(idx_anomaly)
            if verbose > 1:
                print("First cut:", idx_first)
                print("Second cut:", idx_second)
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10,6))
                plt.plot(series, label='original')
                plt.plot(new_series, label='manipulated')

                plt.axvline(x=idx_first, color='r', alpha=0.9, linewidth=.8)
                plt.legend(loc="best")
                plt.xlabel("time / s")
                plt.ylabel("y(t)")
                plt.xlim((idx_first-50,idx_first+300))

            plots.append({"original": series.copy(), "manipulated": new_series.copy(), "idx_cut_first": idx_first, "idx_cut_second": idx_second})

            series = new_series

        if verbose > 0:
            print("All anomalies:", real_anomalies_idx)

        col = ["value"]
        if len(series.shape) > 1:
            col = ["value" + str(i+1) for i in range(series.shape[-1])] 
        series = pandas.DataFrame(series,  columns = col)
        series["is_anomaly"] = 0
        series["is_ignored"] = 0

        # Ignore anomalies in the very beginning and end
        series.loc[0:256, ("is_ignored")] = 1
        #series.loc[series.index[-512:-256], ("is_ignored")] = 1
        #series.iloc[-5120:-2560, series.columns.get_loc('is_ignored')] = 1

        # set anomaly windows
        for i in real_anomalies_idx:
            series.loc[i - anomaly_window:i+anomaly_window, ("is_anomaly")] = 1

        return series, plots
    
    
    def build_data(self, ts_id, verbose = 0):
        series = self.create_chaotic_time_series(ts_id, verbose=verbose, return_plots_data=False)

        if verbose > 2:
            print("CSV Column names:", series.columns.values)
            print("First few rows:", series.head(10))
            
        signal = self.extract_signal(series=series, input_columns = self.input_columns, verbose = verbose)

        if self.scale_method is not None:
            signal = self.scale_data(signal, scale_method = self.scale_method, verbose = verbose)

        X = numpy.array([])
        X = utilities.slide_window(signal, self.window_length, verbose = verbose)

        Y = X # otherwise assume that we have a simple autoencoding task....

        X_full = X

        # If the training sequences are long, it does not make sense to move the sliding window with a stride of 1
        # since the training would take forever and the results would probably not be much better...
        X = X[::self.window_stride]
        Y = Y[::self.window_stride]

        anomaly_labels = series["is_anomaly"][self.error_window_length-1:]

        if ("is_ignored" in series):
            is_ignored = series["is_ignored"][self.error_window_length-1:]
            is_sigChange = 0
            if ("is_signalChange" in series):
                is_sigChange = series["is_signalChange"][self.error_window_length-1:]
            is_ignoreable = numpy.logical_and(numpy.logical_or(is_ignored,  is_sigChange), numpy.logical_not(anomaly_labels)).values
            is_ignoreable[-256:] = True
        else:
            is_ignoreable = numpy.zeros(anomaly_labels.values.shape[0])

        ret = self.split_training_data(X, Y, is_ignoreable, anomaly_labels.values, self.training_split)
        ret["series"] = series
        ret["scaled_series"] = signal
        ret["X_full"] = X_full # Whole matrix X with stride=1
        return ret
    
    
    def extract_signal(self, series, input_columns = ["value"], verbose = 1):
        series = series.copy()
        series = series[list(input_columns)]
        series = series.reset_index(drop=True)
        series = series.apply(axis=0, func=pandas.to_numeric)
        series = series.values
        if verbose > 2:
            print("series.shape:", series.shape)
        series = series + 0.0 # just to make it float
        return pandas.DataFrame(series, columns = input_columns)
    
    def scale_data(self, df, scale_method = "StandardScaler", verbose = 1):
        scaler = MinMaxScaler() if scale_method == "MinMaxScaler" else StandardScaler()
        scaled = scaler.fit_transform(df)
        series = pandas.DataFrame(scaled, columns=df.columns.values)
        if verbose > 2:
            import matplotlib
            print(series.head(10))
            matplotlib.pyplot.figure(figsize=(20,6))
            matplotlib.pyplot.plot(series.values[64000:70000]) 
            matplotlib.pyplot.show()
        return series
    
    def split_training_data(self,X, Y, is_ignoreable, is_anomaly, training_split, verbose = 1):
        # Training split
        nrow = round(training_split * X.shape[0])
        train_X = X[:nrow]
        test_X = X[nrow:]
        train_Y = Y[:nrow]
        train_is_anomaly = is_anomaly[:nrow]
        test_Y = Y[nrow:]
        test_is_anomaly = is_anomaly[:nrow]
        #if is_ignoreable is not None:
        train_ignore = is_ignoreable[:nrow]
        test_ignore = is_ignoreable[nrow:]
        if verbose > 2:
            print("split_training_data()")
            print("train_X.shape", train_X.shape)
            print("train_Y.shape", train_Y.shape)
            print("test_X.shape", test_X.shape)
            print("test_Y.shape", test_Y.shape)
            print("train_ignore.shape", train_ignore.shape)
            print("test_ignore.shape", test_ignore.shape)
        return dict({"X":X,"Y":Y,"is_ignoreable":is_ignoreable,"is_anomaly":is_anomaly,
                     "train_X" : train_X, "train_Y": train_Y, "test_X":test_X, "test_Y":test_Y, 
                     "train_ignore": train_ignore, "test_ignore": test_ignore, "train_is_anomaly": train_is_anomaly, "test_is_anomaly": test_is_anomaly})