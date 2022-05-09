# ======================================= Compute OoS Performance Metrics for Forecasts produced by MIDAS ==================================================== #
# ============================================================================================================================================== #
import pandas as pd
from pandas.core.common import flatten
import numpy as np
import os
import sys
import math
from varname import nameof
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels import robust
from statsmodels.tsa.api import VAR
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
import datetime
import time
import psutil
import multiprocessing as multi
from itertools import product, repeat
from functools import partial

# from dask.distributed import Client, LocalCluster
# import joblib

path_midas = "e:\\Copy\\SCRIPTS\\Forecast_Covid_Recovery\\Code\\midas\\"
sys.path.append(path_midas)

from mix import mix_freq, mix_freq2
from adl import estimate, forecast, midas_adl, rmse, estimate2,forecast2, midas_adl2

os.environ['NUMEXPR_MAX_THREADS'] = '30'

##### Set the current working directory
path="e:\\Copy\\SCRIPTS\\Forecast_Covid_Recovery\\Code\\"
os.chdir(path)

##### parse dates and times
def date_parser(date): 
    dt = datetime.datetime.strptime(date, '%Y:%m:%d')
    return dt.strftime('%Y-%m-%d') 


##### Calculate rolling-window OoS R^2 for tau-step-ahead forecasts based on a univariate MIDAS model
def OoS_R_sq_midas(sstart:int, y_df: pd.DataFrame, x_df: pd.DataFrame, wsize: int, T1: int, xlag: str, ylag: int, tau: int, poly = 'expalmon', method='rolling') -> tuple:
	""" 
		INPUT:   
			y_df and x_df: pandas dataframes with dates in the index column for both y and x;
			'wsize': size of rolling windows; 'sstart': sub-sample starting point; 'T1': size of each sub-sample;
			'xlag': number of lags for the high-frequency variable; 'ylag': number of autoregressive lags for the dependant low-frequency variable;
			'tau': how much the high-frequency data is lagged before frequency mixing; 'poly': a MIDAS polynomial; 'method': a forecasting method ('rolling' or 'recursive') 
		OUTPUT:
			a tuple of performance metrics: 
	"""
	assert (T1 > wsize+tau), "size of a subsample must be much greater than the window size!"
	assert ( sstart+T1 <= len(y_df) ), "end point of the last subsample must be less than or equal to size of the dataframe!"

	y_df = y_df.iloc[sstart:(sstart+T1), :].copy()
	x_df = x_df.copy()
	
	start_date = y_df.index[3]
	end_date = y_df.index[3+wsize]

	# call 'midas_adl' function
	rmse_in, rmse, fc = midas_adl(y_df.iloc[:, 0], x_df.iloc[:, 0], start_date=start_date, end_date=end_date, xlag=xlag, ylag=ylag, horizon=tau, poly=poly, method=method)

	# compute rolling- window mean forecasts
	n_month = y_df.index.get_loc(end_date) - y_df.index.get_loc(start_date)
	y_df['benchmark'] = y_df.iloc[:, 0].rolling(window=n_month).mean()

	# merge 'y_df' to 'fc'
	fc = fc.merge(y_df, how = 'left', left_index=True, right_index=True)

	# calculate mad, mae, rmse, r^2
	fc.reset_index(inplace = True)
	OoS_size = len(fc)
	# print(fc)
	rmse = 0
	var = 0
	mae = 0
	list_err = []
	for s in np.arange(OoS_size):
		r_forecast = fc.loc[s, 'preds']
		r = fc.loc[s, 'targets'] # actual returns
		# print(fc.loc[s, 'index'], r, r_forecast)
		rmse +=  pow(r - r_forecast, 2) / OoS_size
		var += pow(r - fc.loc[s, 'benchmark'], 2) / OoS_size
		mae += abs(r - r_forecast) / OoS_size
		list_err.append(r - r_forecast)
	err = np.array(list_err)
	mad = robust.mad(err, c = 1)
	del y_df, x_df

	return sstart, rmse_in, float(mad), float(mae), math.sqrt(float(rmse) ), float(1 - rmse/(var + 0.00001) )

##### Define a wrapper to parallel compute values of the OoS R^2
def OoS_R_sq_midas_wrapper(y_df, x_df, x_df_name: str, wsize: int, T1: int, xlag: str, ylag: int, tau: int, poly = 'expalmon', method='rolling', processes=-1):
	assert(tau > 0), "the forecast horizon must be greater than zero!"

	# decide how many proccesses will be created
	ssize = len(y_df) - ylag
	if processes <=0:
		num_cpus = psutil.cpu_count(logical=False)
	else:
		num_cpus = processes

	start = time.time()

	# create a list of all feasible rolling-window starting points
	step = 1
	sstarts, ssample_start_dates, ssample_end_dates = [], [], []
	for i in np.arange(0, ssize+step, step):
		if i+T1 <= ssize:
			sstarts.append(i)
			ssample_start_dates.append( y_df.index[i] )
			ssample_end_dates.append( y_df.index[i+T1-1] )

	# start processes in the pool
	print( 'Start multiprocessing . . .' )
	with multi.Pool(processes = num_cpus) as process_pool:
		# perfs = process_pool.starmap( process_OoS_R_sq, zip(sstarts, repeat(use_model) ) )
		perfs = process_pool.starmap( OoS_R_sq_midas, zip(sstarts, repeat(y_df), repeat(x_df), repeat(wsize), repeat(T1), repeat(xlag), \
																																repeat(ylag), repeat(tau), repeat(poly), repeat(method) ) )
	process_pool.close()
	# process_pool.terminate()
	process_pool.join()
	print( 'Multiprocessing done!' )

	# put all numbers into a dataframe
	data = pd.DataFrame( perfs, columns = ['sstart', 'rmse_in', 'mad', 'mae', 'rmse', 'r_sq'] )
	data.insert(loc=1, column='ssample_start_date', value=ssample_start_dates)
	data.insert(loc=2, column='ssample_end_date', value=ssample_end_dates)
	data.to_csv('../Data/FRED/perf_out_{df}_model_midas_sampsize_{ssize}_subsize_{T1}_wsize_{wsize}_fhorizon_{tau}_y'\
													'lag_{ylag}.csv'.format(df=x_df_name, ssize=ssize, T1=T1, wsize=wsize, tau=tau, ylag=ylag), index = False, header = True)
	end = time.time()
	print( 'Completed in: %s sec'%(end - start) )
	return data

if __name__ == '__main__':
	# multi.freeze_support()
	# multi.set_start_method("spawn")
	start_time = time.time() # start the timer

	# Import GDP and ADS data (from 9/1/1959 to 12/1/2020) and plot
	fields_GDP = ['date', 'GDP']
	US_GDP_df = pd.read_csv('../Data\\FRED\\quarterly_transformed_pca.csv', engine = 'python', encoding='utf-8', skipinitialspace=True, usecols = fields_GDP, sep = ',', \
	                                                                                                                                                                                                                parse_dates = ['date'], index_col = 'date')
	US_GDP_df.dropna(inplace=True)
	US_GDP_df = US_GDP_df.copy().resample('Q').last() # convert beginning-of-the-month dates to end-of-the-month dates

	fields_ADS = ['date', 'ADS_Index']
	US_ADS_df = pd.read_csv('../Data\\ADS_Index_Most_Current_Vintage.csv', engine = 'python', encoding='utf-8', skipinitialspace=True, usecols = fields_ADS, sep = ',')
	US_ADS_df.date = pd.to_datetime( US_ADS_df.date.astype(str).apply(date_parser) )# parse dates
	US_ADS_df.dropna(inplace=True)
	US_ADS_df.set_index('date', inplace=True)
	US_ADS_df_monthly = US_ADS_df.copy().resample('M').mean() # convert a time series to the monthly frequency by sampling then taking mean
	US_ADS_df_monthly = US_ADS_df_monthly.copy().shift(3)
	US_ADS_df_monthly.dropna(inplace=True)

	US_GDP_df.plot(subplots=True, figsize=(10, 8), grid=True, rot=10, sharex=False, sharey=False)
	plt.savefig('../Data\\US_gdp_big.pdf', dpi=400)
	US_ADS_df_monthly.plot(subplots=True, figsize=(10, 8), grid=True, rot=10, sharex=False, sharey=False)
	plt.savefig('../Data\\US_ADS.pdf', dpi=400)



	##### Set parameters
	wsize = 60 # set a rolling-window size
	xlag = '3M' # set number of lags for the high-frequency variable to match with data of the low-frequency variable
	ylag = 1 # set a maximum autoregressive lag for the dependent variable
	T1 = 100 # set a sub-sample size

	processes = 60 # set a number of processes to be launched
	tau_step = 1
	taus = np.arange(1, 6+tau_step, tau_step)  # create a list of forecast horizons
	for tau in taus:
		OoS_R_sq_midas_wrapper(US_GDP_df, US_ADS_df_monthly, nameof(US_ADS_df), wsize,  T1, xlag, ylag, tau, \
																														poly = 'expalmon', method='rolling', processes=processes)

	print( 'Completed in: %s sec'%(time.time() - start_time) )



