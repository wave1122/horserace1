""" Calculate out-of-sample MAD, MAE, RMSE, R^2 """
import os

# Use CPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['NUMEXPR_MAX_THREADS'] = '30'

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels import robust
from statistics import median
import sys
import gc
import math
from varname import nameof

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import datetime
import time
import psutil
import multiprocessing as multi
from itertools import product, repeat
from functools import partial

path_midas = "e:\\Seafile\\Library\\Copy\\Py-project\\midas\\"
sys.path.append(path_midas)

from mix import mix_freq, mix_freq2
from adl import estimate, forecast, midas_adl, rmse, estimate2,forecast2, midas_adl2

##### Set the current working directory
path="e:\\Copy\\SCRIPTS\\Forecast_Covid_Recovery\\Code\\"
os.chdir(path)
start_time = time.time()

##### Import algorithms
from my_algorithms_v2 import multivar_lhOLS, multivar_lhOLSPC, Regularized_Reg, prophetf, LSTMf, ANNf, CNNf, XGBf, GBMf, RFf

##### Calculate rolling-window OoS R^2 for 'tau'-steps ahead forecasts
def OoS_R_sq(sstart: int, df: pd.DataFrame, freq: str, tau: int, batch_size: int, num_epochs: int, wsize: int, T1: int, 
                                                                                                                                                        ylag: int, num_PCs: int, use_model, n_jobs = 1) -> pd.DataFrame:
    """ 
        INPUT:
        'sstart': sub-sample starting point
        'df': a pandas dataframe
        'fred': a frequency of type 'string (e.g., '3M', '1D', '1Y', or '1Q' etc.)
        'tau': forecast horizon
        'batch_size': batch size used for the Stochastic Gradient Descent
        'num_epochs': number of epochs used for the Stochastic Gradient Descent
        'wsize': window size
        'T1': size of a sub-sample
        'ylag': AR lag of the dependent variable 
       'num_PCs': number of principal components used for PCA
       OUTPUT:
        OoS median deviance, OoS MAE, OoS, RMSE, OoS R^2, in-sample RMSE, in-sample MAE, median optimal hyperparameters
    """
    assert (T1 > wsize+tau), "size of a subsample must be much greater than the window size!"
    assert (sstart+T1 <= len(df) ), "end point of the last subsample must be less than or equal to size of the dataframe!"

    df = df.iloc[sstart:(sstart+T1), :].copy()
    # get data for the dependant variable and predictors
    if ylag > 0:
        for i in np.arange(1, ylag+1):
            df[f'ylag{i}'] = df.iloc[:, 1].shift(i)
        df.dropna(inplace = True)
    # print( df.iloc[0, 0] )

    R = np.array(df.values[:, 1], dtype='float64')
    X = np.array(df.values[:, 2:], dtype='float64')
    dim = X.shape[1]

    rmse = 0
    var = 0
    mae = 0
    list_err = []
    rmse_in_ls, mae_in_ls, list_opt_params_vlues = [], [], []
    for s in np.arange(T1-wsize-tau-ylag):
        # estimate a long-run regression model and make a 'tau'-steps ahead forecast
        if use_model == 'multivar_lhOLS':
            r_forecast, rmse_in, mae_in, opt_params = multivar_lhOLS(R[s:(s+wsize+1)].reshape(-1, 1), X[s:(s+wsize+1), :].reshape(-1, dim), tau)
            list_opt_params_keys = [k for k in opt_params.keys()]
            list_opt_params_vlues.append( list( opt_params.values() ) )
            rmse_in_ls.append(rmse_in)
            mae_in_ls.append(mae_in)
        elif use_model == 'multivar_lhOLSPC':
            r_forecast, rmse_in, mae_in, opt_params = multivar_lhOLSPC(R[s:(s+wsize+1)].reshape(-1, 1), X[s:(s+wsize+1), :].reshape(-1, dim), num_PCs, tau)
            list_opt_params_keys = [k for k in opt_params.keys()]
            list_opt_params_vlues.append( list( opt_params.values() ) )
            rmse_in_ls.append(rmse_in)
            mae_in_ls.append(mae_in)
        elif use_model == 'lasso':
            r_forecast, rmse_in, mae_in, opt_params = Regularized_Reg(R[s:(s+wsize+1)].reshape(-1, 1), X[s:(s+wsize+1), :].reshape(-1, dim), tau, \
                                                                                                                                                                                                            use_model = 'lasso', n_jobs = n_jobs)
            list_opt_params_keys = [k for k in opt_params.keys()]
            list_opt_params_vlues.append( list( opt_params.values() ) )
            rmse_in_ls.append(rmse_in)
            mae_in_ls.append(mae_in)
        elif use_model == 'ridge':
            r_forecast, rmse_in, mae_in, opt_params = Regularized_Reg(R[s:(s+wsize+1)].reshape(-1, 1), X[s:(s+wsize+1), :].reshape(-1, dim), tau, use_model = 'ridge')
            list_opt_params_keys = [k for k in opt_params.keys()]
            list_opt_params_vlues.append( list( opt_params.values() ) )
            rmse_in_ls.append(rmse_in)
            mae_in_ls.append(mae_in)
        elif use_model == 'prophetf':
            r_forecast, rmse_in, mae_in, opt_params = prophetf(R[s:(s+wsize+1)].reshape(-1, 1), X[s:(s+wsize+1), :].reshape(-1, dim), df.iloc[0, 0], freq, tau)
            list_opt_params_keys = [k for k in opt_params.keys()]
            list_opt_params_vlues.append( list( opt_params.values() ) )
            rmse_in_ls.append(rmse_in)
            mae_in_ls.append(mae_in)
        elif use_model == 'LSTMf':
            r_forecast, rmse_in, mae_in, opt_params = LSTMf(R[s:(s+wsize+1)].reshape(-1, 1), X[s:(s+wsize+1), :].reshape(-1, dim), tau, batch_size, \
                                                                                                                                                                                                                    num_epochs, n_jobs = n_jobs)
            list_opt_params_keys = [k for k in opt_params.keys()]
            list_opt_params_vlues.append( list( opt_params.values() ) )
            rmse_in_ls.append(rmse_in)
            mae_in_ls.append(mae_in)
        elif use_model == 'ANNf':
            r_forecast, rmse_in, mae_in, opt_params = ANNf(R[s:(s+wsize+1)].reshape(-1, 1), X[s:(s+wsize+1), :].reshape(-1, dim), tau, batch_size, \
                                                                                                                                                                                                                    num_epochs, n_jobs = n_jobs)
            list_opt_params_keys = [k for k in opt_params.keys()]
            list_opt_params_vlues.append( list( opt_params.values() ) )
            rmse_in_ls.append(rmse_in)
            mae_in_ls.append(mae_in)
        elif use_model == 'CNNf':
            r_forecast, rmse_in, mae_in, opt_params = CNNf(R[s:(s+wsize+1)].reshape(-1, 1), X[s:(s+wsize+1), :].reshape(-1, dim), tau, batch_size, num_epochs)
            list_opt_params_keys = [k for k in opt_params.keys()]
            list_opt_params_vlues.append( list( opt_params.values() ) )
            rmse_in_ls.append(rmse_in)
            mae_in_ls.append(mae_in)
        elif use_model == 'XGBf':
            r_forecast, rmse_in, mae_in, opt_params = XGBf(R[s:(s+wsize+1)].reshape(-1, 1), X[s:(s+wsize+1), :].reshape(-1, dim), tau, n_jobs = n_jobs)
            list_opt_params_keys = [k for k in opt_params.keys()]
            list_opt_params_vlues.append( list( opt_params.values() ) )
            rmse_in_ls.append(rmse_in)
            mae_in_ls.append(mae_in)
        elif use_model == 'GBMf':
            r_forecast, rmse_in, mae_in, opt_params = GBMf(R[s:(s+wsize+1)].reshape(-1, 1), X[s:(s+wsize+1), :].reshape(-1, dim), tau, n_jobs = n_jobs)
            list_opt_params_keys = [k for k in opt_params.keys()]
            list_opt_params_vlues.append( list( opt_params.values() ) )
            rmse_in_ls.append(rmse_in)
            mae_in_ls.append(mae_in)
        elif use_model == 'RFf':
            r_forecast, rmse_in, mae_in, opt_params = RFf(R[s:(s+wsize+1)].reshape(-1, 1), X[s:(s+wsize+1), :].reshape(-1, dim), tau, n_jobs = n_jobs)
            list_opt_params_keys = [k for k in opt_params.keys()]
            list_opt_params_vlues.append( list( opt_params.values() ) )
            rmse_in_ls.append(rmse_in)
            mae_in_ls.append(mae_in)
        else:
            print(f'Model {use_model} does not exist!')
            sys.exit()

        r = R[s+wsize+tau] # actual returns
        rmse +=  pow(r - r_forecast, 2) / (T1-wsize-tau)
        var += pow(r - np.mean(R[s:(s+wsize+1)]), 2) / (T1-wsize-tau)
        mae += abs(r - r_forecast) / (T1-wsize-tau)
        list_err.append(r - r_forecast)
    err = np.array(list_err)
    mad = robust.mad(err, c = 1)

    # save optimal hyperparameters to a dataframe
    list_opt_params_vlues = np.array(list_opt_params_vlues)
    opt_params_df = pd.DataFrame({list_opt_params_keys[i]: list_opt_params_vlues[:,i] for i in np.arange( len(list_opt_params_keys) )})
    
    del df # delete this copy of the dataframe
    gc.collect()

    output_df = pd.DataFrame({'sstart': sstart, 'mad': float(mad), 'mae': float(mae), 'rmse': math.sqrt( float(rmse) ), 'r_sq': float( 1 - rmse/(var + 0.00001) ), \
                                                    'rmse_in': np.median(rmse_in_ls), 'mae_in': np.median(mae_in_ls)}, index = [0])
    opt_params_df = pd.DataFrame( opt_params_df.select_dtypes(include=np.number).median(axis=0) ).transpose()
    output_df = pd.concat([output_df, opt_params_df], axis = 1)
    return output_df

##### Define a wrapper to parallel compute values of the OoS R^2
def OoS_R_sq_wrapper(df: pd.DataFrame, df_name, freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, use_model='prophetf', processes=1, n_jobs = -10):
    """
        INPUT
            df: pandas dataframe starting with a date column, then a column of responses, and  other columns of predictors
            df_name: name of this dataframe
            freq: data frequency of string type (e.g., '3M', '1D', '1Y', or '1Q' etc.)
            tau: forecast horizon
            batch_size: batch size to train an algorithm
            num_epochs: number of epochs
            wsize: size of rolling windows
            T1: size of a subsample
            ylag: number of lagged dependent variables used as predictors
            num_PCs: number of principal components
            processes: number of processors to be used (set 'processes = 1'  for serial processing)
            n_jobs: number of jobs for an algorithm to run in parallel
        OUTPUT
            a dataframe of performance metrics and a dataframe of optimal hyperparameters
    """
    
    df = df.copy() # create a copy of the dataframe

	# decide how many proccesses will be created
    if processes <=0:
        num_cpus = psutil.cpu_count(logical=False) - 10
    else:
        num_cpus = processes

    start = time.time()

    # create a list of all feasible rolling-window starting points
    ssize = len(df) - ylag
    step = 1
    sstarts, ssample_start_dates, ssample_end_dates = [], [], []
    for i in np.arange(0, ssize+step, step):
        if i+T1 <= ssize:
            sstarts.append(i) 
            ssample_start_dates.append( df.iloc[i, 0] )
            ssample_end_dates.append( df.iloc[i+T1-1, 0] )

    if processes == 1: # Serial implementation
        allout_df = pd.DataFrame()
        for sstart in sstarts:
            output_df = OoS_R_sq(sstart, df=df, freq=freq, tau=tau, batch_size=batch_size, num_epochs=num_epochs, wsize=wsize, T1=T1, ylag=ylag, \
                                                                                                                                                     num_PCs=num_PCs, use_model=use_model, n_jobs = n_jobs)
            display( output_df.head() )
            allout_df = pd.concat([allout_df, output_df], axis = 0, join = 'outer')
        # display( allout_df.head() )
    else: # Parallel implementation
        # start processes in the pool
        print( 'Start multiprocessing . . .' )
        OoS_R_sq_partial = partial(OoS_R_sq, df=df, freq=freq, tau=tau, batch_size=batch_size, num_epochs=num_epochs, wsize=wsize, T1=T1, ylag=ylag, \
                                                                                                                                                                    num_PCs=num_PCs, use_model=use_model, n_jobs = 1)
        with multi.Pool(processes = num_cpus) as process_pool:
            out_dfs = process_pool.map(OoS_R_sq_partial, sstarts)
        process_pool.close()
        process_pool.join()
        print( 'Multiprocessing done!' )
        
        # put all performance metrics into a dataframe
        allout_df = pd.concat(out_dfs)
        # display( allout_df.head() )

    # delete data
    del df
    gc.collect()

    allout_df.insert(loc=1, column='ssample_start_date', value=ssample_start_dates)
    allout_df.insert(loc=2, column='ssample_end_date', value=ssample_end_dates)

    perf_df = allout_df.iloc[:, 0:9]
    perf_df.to_csv('../Data/FRED/perf_out_{df}_model_{use_model}_sampsize_{ssize}_subsize_{T1}_wsize_{wsize}_fhorizon_{tau}_'\
                                                    'ylag_{ylag}_num_pcs_{num_PCs}_sklearn.csv'.format(df=df_name, use_model=use_model, ssize=ssize, T1=T1, \
                                                    wsize=wsize, tau=tau, ylag=ylag, num_PCs=num_PCs), index = False, header = True)

    opt_params_df  = allout_df.drop(allout_df.columns[3:9], axis = 1)
    opt_params_df.to_csv('../Data/FRED/opt_params_out_{df}_model_{use_model}_sampsize_{ssize}_subsize_{T1}_wsize_{wsize}_fhorizon_{tau}_'\
                                                'ylag_{ylag}_num_pcs_{num_PCs}_sklearn.csv'.format(df=df_name, use_model=use_model, ssize=ssize, T1=T1, \
                                                wsize=wsize, tau=tau, ylag=ylag, num_PCs=num_PCs), index = False, header = True)

    end = time.time()
    print( 'Completed in: %s sec'%(end - start) )
    return perf_df, opt_params_df

if __name__ == '__main__':
    multi.freeze_support()
    # multi.set_start_method("spawn")
    start_time = time.time() # start the timer

    ##### Import data
    # import a sample with many predictors
    US_df_big = pd.read_csv('../Data/FRED/quarterly_transformed_pca.csv', engine = 'python', encoding='utf-8', skipinitialspace=True, sep = ',', parse_dates=True)
    US_df_big1 = US_df_big.copy()[ US_df_big.columns[US_df_big.isnull().mean() < 0.10] ] # keep only columns with less than 10% missing values

    # import FRED variables selected by variable screening
    US_df_small = pd.read_csv('../Data/FRED/quarterly_transformed_pdc_sis.csv', engine = 'python', encoding='utf-8', skipinitialspace=True, sep = ',', parse_dates=True)
  
    # import ADSI in addition to FRED variables selected by variable screening 
    US_df_ADS_small = pd.read_csv('../Data/combined_data_ADSI.csv', engine = 'python', encoding='utf-8', skipinitialspace=True, sep = ',', parse_dates=True)

    ##### Set parameters
    batch_size = 40
    num_epochs = 100
    freq = '3M'
    wsize = 60 # set a rolling-window size
    ylag = 1 # set a maximum autoregressive lag for the dependent variable
    T1 = 100 # set a sub-sample size
    num_PCs = 7 # set number of principal components

    ############################################################## Forecast with 'US_df_small' #######################################################################

    tau_step = 1
    taus = np.arange(2, 4+tau_step, tau_step)  # create a list of forecast horizons
    for tau in taus:
        use_model = 'ANNf'
        perf, parms = OoS_R_sq_wrapper(US_df_small, nameof(US_df_small), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                            use_model=use_model, processes=1)
        print(perf.head() )
        print(parms.head() )

        use_model = 'GBMf'
        perf, parms = OoS_R_sq_wrapper(US_df_small, nameof(US_df_small), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                            use_model=use_model, processes=1)
        print(perf.head() )
        print(parms.head() )

        use_model = 'RFf'
        perf, parms = OoS_R_sq_wrapper(US_df_small, nameof(US_df_small), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                            use_model=use_model, processes=-1)
        print(perf.head() )
        print(parms.head() )

        use_model = 'XGBf'
        perf, parms = OoS_R_sq_wrapper(US_df_small, nameof(US_df_small), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                            use_model=use_model, processes=1)
        print(perf.head() )
        print(parms.head() )

        use_model = 'multivar_lhOLSPC'
        perf, parms = OoS_R_sq_wrapper(US_df_small, nameof(US_df_small), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                            use_model=use_model, processes=-1)
        print(perf.head() )
        print(parms.head() )

        use_model = 'lasso'
        perf, parms = OoS_R_sq_wrapper(US_df_small, nameof(US_df_small), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                            use_model=use_model, processes=-1)
        print(perf.head() )
        print(parms.head() )

        use_model = 'ridge'
        perf, parms = OoS_R_sq_wrapper(US_df_small, nameof(US_df_small), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                            use_model=use_model, processes=-1)
        print(perf.head() )
        print(parms.head() )

        use_model = 'LSTMf'
        perf, parms = OoS_R_sq_wrapper(US_df_small, nameof(US_df_small), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                                use_model=use_model, processes=1)
        print(perf.head() )
        print(parms.head() )

        use_model = 'prophetf'
        perf, parms = OoS_R_sq_wrapper(US_df_small, nameof(US_df_small), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                            use_model=use_model, processes=-1)
        print(perf.head() )
        print(parms.head() )
	
    ############################################################## Forecast with 'US_df_big1' #######################################################################
    tau_step = 1
    taus = np.arange(1, 4+tau_step, tau_step)  # create a list of forecast horizons
    for tau in taus:
        if tau > 1:
            use_model = 'ANNf'
            perf, parms = OoS_R_sq_wrapper(US_df_big1, nameof(US_df_big1), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                                use_model=use_model, processes=1)
            print(perf.head() )
            print(parms.head() )

            use_model = 'GBMf'
            perf, parms = OoS_R_sq_wrapper(US_df_big1, nameof(US_df_big1), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                                use_model=use_model, processes=1)
            print(perf.head() )
            print(parms.head() )

            use_model = 'RFf'
            perf, parms = OoS_R_sq_wrapper(US_df_big1, nameof(US_df_big1), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                                use_model=use_model, processes=-1)
            print(perf.head() )
            print(parms.head() )

            # use_model = 'XGBf'
            # perf, parms = OoS_R_sq_wrapper(US_df_big1, nameof(US_df_big1), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
            #                                                                                                                                                                     use_model=use_model, processes=1)
            # print(perf.head() )
            # print(parms.head() )

            use_model = 'multivar_lhOLSPC'
            perf, parms = OoS_R_sq_wrapper(US_df_big1, nameof(US_df_big1), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                                use_model=use_model, processes=-1)
            print(perf.head() )
            print(parms.head() )

            use_model = 'lasso'
            perf, parms = OoS_R_sq_wrapper(US_df_big1, nameof(US_df_big1), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                                use_model=use_model, processes=-1)
            print(perf.head() )
            print(parms.head() )

            use_model = 'ridge'
            perf, parms = OoS_R_sq_wrapper(US_df_big1, nameof(US_df_big1), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                                use_model=use_model, processes=-1)
            print(perf.head() )
            print(parms.head() )

            use_model = 'LSTMf'
            perf, parms = OoS_R_sq_wrapper(US_df_big1, nameof(US_df_big1), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                                    use_model=use_model, processes=1)
            print(perf.head() )
            print(parms.head() )

            use_model = 'prophetf'
            perf, parms = OoS_R_sq_wrapper(US_df_big1, nameof(US_df_big1), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                                use_model=use_model, processes=-1)
            print(perf.head() )
            print(parms.head() )
	
    ############################################################## Forecast with 'US_df_ADS_small' #######################################################################
    tau_step = 1
    taus = np.arange(1, 4+tau_step, tau_step)  # create a list of forecast horizons
    for tau in taus:
        use_model = 'ANNf'
        perf, parms = OoS_R_sq_wrapper(US_df_ADS_small, nameof(US_df_ADS_small), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                            use_model=use_model, processes=1)
        print(perf.head() )
        print(parms.head() )

        use_model = 'GBMf'
        perf, parms = OoS_R_sq_wrapper(US_df_ADS_small, nameof(US_df_ADS_small), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                            use_model=use_model, processes=1)
        print(perf.head() )
        print(parms.head() )

        use_model = 'RFf'
        perf, parms = OoS_R_sq_wrapper(US_df_ADS_small, nameof(US_df_ADS_small), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                            use_model=use_model, processes=-1)
        print(perf.head() )
        print(parms.head() )

        use_model = 'XGBf'
        perf, parms = OoS_R_sq_wrapper(US_df_ADS_small, nameof(US_df_ADS_small), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                            use_model=use_model, processes=1)
        print(perf.head() )
        print(parms.head() )

        use_model = 'multivar_lhOLSPC'
        perf, parms = OoS_R_sq_wrapper(US_df_ADS_small, nameof(US_df_ADS_small), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                            use_model=use_model, processes=-1)
        print(perf.head() )
        print(parms.head() )

        use_model = 'lasso'
        perf, parms = OoS_R_sq_wrapper(US_df_ADS_small, nameof(US_df_ADS_small), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                            use_model=use_model, processes=-1)
        print(perf.head() )
        print(parms.head() )

        use_model = 'ridge'
        perf, parms = OoS_R_sq_wrapper(US_df_ADS_small, nameof(US_df_ADS_small), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                            use_model=use_model, processes=-1)
        print(perf.head() )
        print(parms.head() )

        use_model = 'LSTMf'
        perf, parms = OoS_R_sq_wrapper(US_df_ADS_small, nameof(US_df_ADS_small), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                                use_model=use_model, processes=1)
        print(perf.head() )
        print(parms.head() )

        use_model = 'prophetf'
        perf, parms = OoS_R_sq_wrapper(US_df_ADS_small, nameof(US_df_ADS_small), freq, tau, batch_size, num_epochs, wsize, T1, ylag, num_PCs, \
                                                                                                                                                                            use_model=use_model, processes=-1)
        print(perf.head() )
        print(parms.head() )

    print( 'Completed in: %s sec'%(time.time() - start_time) )