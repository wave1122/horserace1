# =========================================== ML Algorithms to Train/Validate/Forecast ==================================================== #
# ============================================================================================================================== #

import pandas as pd
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels import robust
from statsmodels.tsa.api import VAR
from statistics import median
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# import Sklearn modules to train machine-learning models
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# import data pre-processing modules
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# import Random Forest module
from sklearn.ensemble import RandomForestRegressor

# import GBM
from sklearn.ensemble import GradientBoostingRegressor

# import XGBoost module
from xgboost import XGBRegressor
from skopt import gp_minimize
from skopt.space import Real, Integer
from functools import partial

# import Facebook Prophet module
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.utilities import regressor_coefficients 
import utils_fprophet

import logging
logging.getLogger('prophet').setLevel(logging.ERROR)

from suppress_stdout_stderr import suppress_stdout_stderr

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Dense, Activation, LSTM, Dropout, SimpleRNN
# from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Conv1D
from tensorflow.python.keras.layers import MaxPooling1D

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

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Split a multivariate sequence into samples that comform with the format required by LSTM/CNN
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Reshape samples to the format required by LSTM
def create_dataset(y, X, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps), :]
        Xs.append(v)        
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Create a dataset that has 'time_steps' periods' values of the predictors so that we can predict the GDP for time point 'time_steps'+1
def create_dataset2(y, X, time_steps = 1):
    """ 
    time_steps: a time step ( >= 1)
    y, X: numpy arrays of predictand and predictors
    """
    Xs, ys = [], []
    nobs = X.shape[0]
    if (nobs == time_steps):
        Xs.append(X[0:time_steps])
        ys.append(y[time_steps-1])
    else:
        for i in np.arange(time_steps, nobs):
            Xs.append(X[(i-time_steps+1):(i+1)])
            ys.append(y[i])
    return np.array(Xs), np.array(ys)

##### Calculate the OLS estimates for long-horizon univariate regression models
def univar_lhOLS(R, X, tau):
    assert (R.shape[0] == X.shape[0]), "numbers of rows not match!"
    T = X.shape[0]
    R = R.flatten()
    X = X.flatten() # flatten arrays

    lhR = np.empty( shape = (0, 1) )
    lhX = np.empty( shape = (0, 1) )
    for t in np.arange(0, T-tau):
        lhR = np.append(lhR, R[t+tau])
        lhX = np.append(lhX, X[t])
    # estimate the regression coefficient
    data_pd = pd.DataFrame({'y': lhR, 'x': lhX})
    linearModel = smf.ols(formula='y ~ x', data=data_pd)
    Reg_coeff = linearModel.fit()
    B = Reg_coeff.params
    epsilon = lhR - B[0] - B[1]*lhX # calculate residuals
    forecast_tau = B[0] + B[1]*X[T-1] # calculate an out-of-sample forecast
    # print(IEB)
    return forecast_tau, epsilon

#### Calculate the OLS estimates for long-horizon multivariate regression models
def multivar_lhOLS(R, X, tau):
    assert (R.shape[0] == X.shape[0]), "numbers of rows not match!"
    T = X.shape[0]
    dim = X.shape[1]
    # R = R.flatten()
    # X = X.flatten() # flatten arrays

    R1 = np.empty( shape = (0, 1) )
    X1 = np.empty( shape = (0, dim) )
    for t in np.arange(0, T-tau):
        R1 = np.append(R1, R[t+tau].reshape(1, 1), axis = 0)
        X1 = np.append(X1, X[t, :].reshape(1, dim), axis = 0)

        # Studentize data
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1)

    # # Rescale data
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # X1 = scaler.fit_transform(X1)

    R1 = np.mat(R1)
    X1 = np.mat(X1)
    
    # do time series regressions
    X11 = sm.add_constant(X1)
    ts_res = sm.OLS(R1, X11).fit()
    alpha = ts_res.params[0]
    beta = np.mat(ts_res.params[1:])
    params = {f'beta{i}': beta[0,i] for i in np.arange(beta.shape[1])}
    
    # compute RMSE and MAE
    epsilon = R1 - alpha - (X1 @ beta.T)
    rmse = np.sqrt( np.mean(np.power(epsilon.tolist(), 2.) ) )
    mae = np.mean( np.abs(epsilon.tolist() ) )

    # compute forecast
    forecast_tau = alpha + ( beta @ X[T-1, :].reshape(dim, 1) )
    return float(forecast_tau), rmse, mae, params

def multivar_lhOLSPC(R, X, num_PCs, tau):
    assert (R.shape[0] == X.shape[0]), "numbers of rows not match!"
    T = X.shape[0]
    dim = X.shape[1]
    # R = R.flatten()
    # X = X.flatten() # flatten arrays

    R1 = np.empty( shape = (0, 1) )
    X1 = np.empty( shape = (0, dim) )
    for t in np.arange(0, T-tau):
        R1 = np.append(R1, R[t+tau].reshape(1, 1), axis = 0)
        X1 = np.append(X1, X[t, :].reshape(1, dim), axis = 0)

       # Studentize data
    scaler = StandardScaler()
    X11 = scaler.fit_transform(X1)

    # # Rescale data
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # X11 = scaler.fit_transform(X1)

    # estimate principal components
    pca = PCA(n_components = num_PCs)
    PCs = pca.fit_transform(X11)
    # PCs_df = pd.DataFrame( data = PCs, columns = [f'PC{i+1}' for i in range(num_PCs)] )
    variance_percentages = pca.explained_variance_ratio_
    variance_proportions = {f'var{i}': variance_percentages[i] for i in np.arange( len(variance_percentages) )}

    # do time series regressions
    PCs = np.mat(PCs)
    PCs1 = sm.add_constant(PCs)
    R1 = np.mat(R1)
    ts_res = sm.OLS(R1, PCs1).fit()
    alpha = ts_res.params[0]
    beta = np.mat(ts_res.params[1:])
    
    # compute the RMSE and MAE
    epsilon = R1 - alpha - (PCs @ beta.T)
    rmse = np.sqrt( np.mean(np.power(epsilon.tolist(), 2.) ) )
    mae = np.mean( np.abs(epsilon.tolist() ) )

    # compute forecast
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    forecast_tau = alpha + ( beta @ (loadings.T @ X[T-1, :].reshape(dim, 1) ) )
    return float(forecast_tau), rmse, mae, variance_proportions
    
def CNNf(R, X, tau, batch_size: int, num_epochs: int):
        assert (R.shape[0] == X.shape[0]), "numbers of rows not match!"
        T = X.shape[0]
        dim = X.shape[1]
        # R = R.flatten()
        # X = X.flatten() # flatten arrays

        R1 = np.empty( shape = (0, 1) )
        X1 = np.empty( shape = (0, dim) )
        for t in np.arange(0, T):
            if t < T-tau:
                R1 = np.append(R1, R[t+tau].reshape(1, 1), axis = 0)
            else:
                R1 = np.append(R1, np.array([0]).reshape(1, 1), axis = 0)
            X1 = np.append(X1, X[t, :].reshape(1, dim), axis = 0)

        # studentize data
        # X1 = StandardScaler().fit_transform(X1)

        # choose a number of time steps
        n_steps = 3

        # split the numpy array into train and test data
        X1_train, X1_test = X1[0:(T-n_steps), :], X1[(T-n_steps):, :]
        R1_train, R1_test = R1[0:(T-n_steps)], R1[(T-n_steps):]

        # create an array that conforms with the format of CNN
        dataset = np.hstack( (X1_train, R1_train) )
        # convert into the input and output array format
        X_train, R_train = split_sequences(dataset, n_steps)
        n_features = X_train.shape[2]
        # print(f'n_steps = {n_steps}; n_features = {n_features}')
        
        # create the model
        model = Sequential()
        model.add( Conv1D( filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features) ) )
        model.add( MaxPooling1D(pool_size=2) )
        model.add( Flatten() )
        model.add( Dense(50, activation='relu') )
        model.add( Dense(1, activation = 'relu') )

        # compile the CNN model
        model.compile(optimizer='adam', loss='mse')
        # model.summary()

        # train the model
        model.fit(X_train, R_train, batch_size=batch_size, shuffle=False, epochs=num_epochs, verbose=0) # Verbosity mode 0: silent

        # calculate residuals for the train data
        R_pred = model(X_train)
        residuals = R_train - R_pred

        # forecast for the test data
        forecasts = model( X1_test.reshape(-1, n_steps, n_features) )
        # print('forecasts = \n', forecasts)
        forecast_tau = float(forecasts[len(forecasts) - 1])
        
        K.clear_session()
        del model # delete the model

        # output results
        if tau == 0:
            return forecast_tau, residuals.numpy()
        else:
            return forecast_tau, residuals.numpy()

def Regularized_Reg(R: np.array, X: np.array, tau: int, use_model = 'lasso', n_jobs = 1):
    assert (R.shape[0] == X.shape[0]), "numbers of rows not match!"
    assert(tau > 0), "the forecast horizon must be greater than zero!"
    
    T = X.shape[0]
    dim = X.shape[1]
    # R = R.flatten()
    # X = X.flatten() # flatten arrays

    R1 = np.empty( shape = (0, 1) )
    X1 = np.empty( shape = (0, dim) )
    for t in np.arange(0, T):
        if t < T-tau:
            R1 = np.append(R1, R[t+tau].reshape(1, 1), axis = 0)
        else:
            R1 = np.append(R1, np.array([0]).reshape(1, 1), axis = 0)
        X1 = np.append(X1, X[t, :].reshape(1, dim), axis = 0)

    # Studentize data
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1)

    # # Rescale data
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # X1 = scaler.fit_transform(X1)

    # split the numpy array into train and test data
    X1_train, X1_test = X1[0:(T-tau), :], X1[(T-tau):, :]
    R1_train, R1_test = R1[0:(T-tau)].ravel(), R1[(T-tau):].ravel()

    # use time-series cross-validation
    tscv = TimeSeriesSplit(n_splits = 2, test_size = 10) 

    if use_model == 'lasso':
        # define the model
        alphas = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
        model = LassoCV(alphas = alphas, cv=tscv, n_jobs=n_jobs, max_iter = 1000000)
    elif use_model == 'ridge':
        # define the model
        alphas = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
        model = RidgeCV(alphas = alphas, cv=tscv)
    else:
        print(f'Regularized_Reg: Model {use_model} does not exist!')
        sys.exit()

    # train the model
    model.fit(X1_train, R1_train)

    # calculate the absolute values of the estimated coefficients
    # coefs = np.abs(model.coef_)
    alpha = {'alpha': model.alpha_}

    # Calculate RMSE and MAE for the validation data
    scores = cross_validate(model, X1_train, R1_train, cv = tscv, scoring = ('neg_root_mean_squared_error', 'neg_mean_absolute_error'), n_jobs = n_jobs)
    rmse = -np.mean(scores['test_neg_root_mean_squared_error'])
    mae = -np.mean(scores['test_neg_mean_absolute_error'])

    # forecast for the test data
    forecasts = model.predict( X1_test.reshape(-1, dim) )
    # print('forecasts = \n', forecasts)
    forecast_tau = float(forecasts[len(forecasts) - 1])
    
    del model # delete the model

    # output results
    return forecast_tau, rmse, mae, alpha

##### Calculate the mean squared error of XGBoost forecasts for validating samples
def rmsfe_XGBoost(args, R_train, X_train, R_valid, X_valid, seed):
    """   seed               : model seed
    booster : booster to use (gbtree, gblinear or dart; gbtree and dart use tree based models while gblinear uses linear functions)
    n_estimators       : number of boosted trees to fit
    max_depth          : maximum tree depth for base learners
    learning_rate      : boosting learning rate (xgb’s “eta”)
    max_delta_step : maximum step size
    min_child_weight   : minimum sum of instance weight(hessian) needed in a child
    subsample          : subsample ratio of the training instance
    colsample_bytree   : subsample ratio of columns when constructing each tree
    colsample_bylevel  : subsample ratio of columns for each split, in each level
    gamma              :  regularization hyperparameter 
    reg_alpha, reg_lambda: regularization parameters """

    # global models, train_scores, test_scores, curr_model_hyper_params
    curr_model_hyper_params = ['colsample_bylevel', 'colsample_bytree', 'gamma', 'learning_rate', 'max_delta_step', 'max_depth', \
                                                        'min_child_weight', 'n_estimators', 'reg_alpha', 'reg_lambda', 'subsample']
    params = {curr_model_hyper_params[i]: args[i] for i, j in enumerate(curr_model_hyper_params)}
    model = XGBRegressor(booster='gbtree', objective ='reg:squarederror', random_state=42, seed=seed)
    model.set_params(**params)
    model.fit(X_train, R_train) # fit training samples to model
    R_pred = model.predict(X_valid)
    msfe = mean_squared_error(R_valid, R_pred)
    del model
    return msfe

##### Find optimal hyperparameters for XGBoost with cross validation
def minimize_XGBoost(R_train, X_train, R_valid, X_valid, seed = 100, n_calls = 50):
    # defining the space
    space = [
        Real(0.1, 1, name="colsample_bylevel"),
        Real(0.1, 1, name="colsample_bytree"),
        Real(0, 1, name="gamma"),
        Real(0, 1, name="learning_rate"),
        Real(0, 10, name="max_delta_step"),
        Integer(1, 15, name="max_depth"),
        Real(0.1, 500, name="min_child_weight"),
        Integer(10, 100, name="n_estimators"),
        Real(0, 0.5, name="reg_alpha"),
        Real(0, 0.5, name="reg_lambda"),
        Real(0.1, 1, name="subsample"),
    ]

    objective_function = partial(rmsfe_XGBoost, R_train=R_train, X_train=X_train, R_valid=R_valid, X_valid=X_valid, seed=seed)

    # minimize the RMSFE
    res = gp_minimize(objective_function, space, base_estimator=None, n_calls=n_calls, n_random_starts=n_calls-1, random_state=42, n_jobs=1)
    return res.x

##### Implement XGBoost using cross validation
def XGBoostf_CV(R, X, tau: int, seed=1234, n_calls = 150):
    assert (R.shape[0] == X.shape[0]), "numbers of rows not match!"
    assert(tau > 0), "the forecast horizon must be greater than zero!"
    
    T = X.shape[0]
    dim = X.shape[1]
    # R = R.flatten()
    # X = X.flatten() # flatten arrays

    R1 = np.empty( shape = (0, 1) )
    X1 = np.empty( shape = (0, dim) )
    for t in np.arange(0, T):
        if t < T-tau:
            R1 = np.append(R1, R[t+tau].reshape(1, 1), axis = 0)
        else:
            R1 = np.append(R1, np.array([0]).reshape(1, 1), axis = 0)
        X1 = np.append(X1, X[t, :].reshape(1, dim), axis = 0)

    # studentize data
    X1 = StandardScaler().fit_transform(X1)

    # split the numpy array into train, validation, and test data
    sratio = 0.8
    X1_train, X1_valid, X1_test = X1[0:(math.floor(sratio*T)-tau), :].reshape(-1, dim), X1[(math.floor(sratio*T)-tau):(T-tau), :].reshape(-1, dim), \
                                                                                                                                                                                X1[(T-tau):, :].reshape(-1, dim)
    R1_train, R1_valid, R1_test = R1[0:(math.floor(sratio*T)-tau), :].reshape(-1, 1), R1[(math.floor(sratio*T)-tau):(T-tau), :].reshape(-1, 1), \
                                                                                                                                                                                    R1[(T-tau):, :].reshape(-1, 1)

    # find optimal hyperparameters for XGBoost using cross validation
    x = minimize_XGBoost(R1_train, X1_train, R1_valid, X1_valid, seed = seed, n_calls = n_calls)

    # define model with optimal hyperparameters
    model = XGBRegressor(booster='gbtree', objective ='reg:squarederror', seed=seed, colsample_bylevel=x[0], colsample_bytree=x[1], gamma=x[2], \
                                            learning_rate=x[3], max_delta_step=x[4], max_depth=x[5], min_child_weight=x[6], n_estimators=x[7], \
                                            reg_alpha=x[8], reg_lambda=x[9], subsample=x[10])

    # train the model
    X1_train_valid = np.concatenate( (X1_train, X1_valid), axis=0)
    R1_train_valid = np.concatenate( (R1_train, R1_valid), axis=0)
    model.fit(X1_train_valid, R1_train_valid)

    # calculate residuals for the train data
    R1_pred = model.predict(X1_train_valid)
    residuals = R1_train_valid - R1_pred

    # forecast for the test data
    forecasts = model.predict(X1_test)
    # print('forecasts = \n', forecasts)
    forecast_tau = float(forecasts[len(forecasts) - 1])
    
    del model # delete the model

    # output results
    return forecast_tau, residuals

##### Implement XGBoost without cross validation
def XGBoostf(R, X, tau: int, seed=100, n_estimators=100, max_depth=3, learning_rate=0.1, min_child_weight=1, subsample=0.8, colsample_bytree=1, \
                        colsample_bylevel=1, reg_alpha=0, gamma=0):
    """   seed               : model seed
            n_estimators       : number of boosted trees to fit
            max_depth          : maximum tree depth for base learners
            learning_rate      : boosting learning rate (xgb’s “eta”)
            min_child_weight   : minimum sum of instance weight(hessian) needed in a child
            subsample          : subsample ratio of the training instance
            colsample_bytree   : subsample ratio of columns when constructing each tree
            colsample_bylevel  : subsample ratio of columns for each split, in each level
            reg_alpha: L1 regularization parameter
            gamma              :  value of the minimum loss reduction required to make a split """
    assert (R.shape[0] == X.shape[0]), "numbers of rows not match!"
    assert(tau > 0), "the forecast horizon must be greater than zero!"
    
    T = X.shape[0]
    dim = X.shape[1]
    # R = R.flatten()
    # X = X.flatten() # flatten arrays

    R1 = np.empty( shape = (0, 1) )
    X1 = np.empty( shape = (0, dim) )
    for t in np.arange(0, T):
        if t < T-tau:
            R1 = np.append(R1, R[t+tau].reshape(1, 1), axis = 0)
        else:
            R1 = np.append(R1, np.array([0]).reshape(1, 1), axis = 0)
        X1 = np.append(X1, X[t, :].reshape(1, dim), axis = 0)

    # studentize data
    X1 = StandardScaler().fit_transform(X1)

    # split the numpy array into train and test data
    X1_train, X1_test = X1[0:(T-tau), :], X1[(T-tau):, :]
    R1_train, R1_test = R1[0:(T-tau)], R1[(T-tau):]

    # define the model
    model = XGBRegressor(booster='gbtree', objective ='reg:squarederror', seed=seed, n_estimators=n_estimators, max_depth=max_depth, \
                                            learning_rate=learning_rate, reg_alpha=reg_alpha, min_child_weight=min_child_weight, subsample=subsample, \
                                            colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel, gamma=gamma)
    
    # Train the model
    model.fit(X1_train, R1_train)

    # calculate residuals for the train data
    R1_pred = model.predict(X1_train)
    residuals = R1_train - R1_pred

    # forecast for the test data
    forecasts = model.predict( X1_test.reshape(-1, dim) )
    # print('forecasts = \n', forecasts)
    forecast_tau = float(forecasts[len(forecasts) - 1])
    
    del model # delete the model

    # output results
    return forecast_tau, residuals


# Create a GBM model
def create_GBM_model(learning_rate = 0.1, n_estimators = 100, subsample = 1.0, max_depth = 3):
    model = GradientBoostingRegressor(loss='squared_error', learning_rate=learning_rate, n_estimators=n_estimators, \
                                                                        subsample=subsample, criterion='squared_error', max_depth=max_depth)
    return model

# Forecast with GBM
def GBMf(R, X, tau: int, n_jobs = 1):
    assert (R.shape[0] == X.shape[0]), "numbers of rows not match!"
    assert(tau > 0), "the forecast horizon must be greater than zero!"
    
    T = X.shape[0]
    dim = X.shape[1]
    # R = R.flatten()
    # X = X.flatten() # flatten arrays

    R1 = np.empty( shape = (0, 1) )
    X1 = np.empty( shape = (0, dim) )
    for t in np.arange(0, T):
        if t < T-tau:
            R1 = np.append(R1, R[t+tau].reshape(1, 1), axis = 0)
        else:
            R1 = np.append(R1, np.array([0]).reshape(1, 1), axis = 0)
        X1 = np.append(X1, X[t, :].reshape(1, dim), axis = 0)

    # Studentize data
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1)

    # # Rescale data
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # X1 = scaler.fit_transform(X1)

    # split the numpy array into train and test data
    X1_train, X1_test = X1[0:(T-tau), :], X1[(T-tau):, :]
    R1_train, R1_test = R1[0:(T-tau)].ravel(), R1[(T-tau):].ravel()

    # Build a GBM model
    model = KerasRegressor(build_fn = create_GBM_model)

    # Define the grid search parameters
    learning_rate = [0.0001, 0.001, 0.01, 0.3]
    n_estimators = [100, 500, 1000]
    subsample = [0.5, 1.0]
    max_depth = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

    param_grid = dict(learning_rate = learning_rate, n_estimators = n_estimators, subsample = subsample, max_depth = max_depth)

	# use time-series cross-validation
    tscv = TimeSeriesSplit(n_splits = 2, test_size = 10) 

	# Perform grid search. Note the convention that higher score values are better than lower score values
    model_cv = GridSearchCV(model, param_grid = param_grid, cv = tscv, scoring = "neg_mean_squared_error", refit = True, n_jobs = n_jobs)

    # Cross-validate a model by using the grid search
    model_cv.fit(X1_train, R1_train)

     # Forecast the test data
    forecasts = model_cv.predict( X1_test)
    if tau > 1:
        forecast_tau = forecasts[len(forecasts)-1]
    else:
        forecast_tau = forecasts

    # Get the optimal hyperparameters
    opt_params = model_cv.best_params_
    # print(f'Optimal hyperparameters:\n {opt_params}')

    # Create a GBM model using the optimal hyperparameters
    best_model = create_GBM_model(**opt_params)
    
    # Calculate RMSE and MAE for the validation data
    scores = cross_validate(best_model, X1_train, R1_train, cv = tscv, scoring = ('neg_root_mean_squared_error', 'neg_mean_absolute_error'), n_jobs = n_jobs)
    rmse = -np.mean(scores['test_neg_root_mean_squared_error'])
    mae = -np.mean(scores['test_neg_mean_absolute_error'])

    del model, model_cv, best_model # delete all models

    # output results
    return float(forecast_tau), rmse, mae, opt_params

# Create a Random Forest model
def create_RF_model(n_estimators = 100, max_depth = 3, bootstrap = True):
    model =  RandomForestRegressor(n_estimators=n_estimators, criterion='squared_error', max_depth=max_depth, bootstrap=bootstrap)
    return model

# Forecast with Random Forest
def RFf(R, X, tau: int, n_jobs = 1):
    assert (R.shape[0] == X.shape[0]), "numbers of rows not match!"
    assert(tau > 0), "the forecast horizon must be greater than zero!"
    
    T = X.shape[0]
    dim = X.shape[1]
    # R = R.flatten()
    # X = X.flatten() # flatten arrays

    R1 = np.empty( shape = (0, 1) )
    X1 = np.empty( shape = (0, dim) )
    for t in np.arange(0, T):
        if t < T-tau:
            R1 = np.append(R1, R[t+tau].reshape(1, 1), axis = 0)
        else:
            R1 = np.append(R1, np.array([0]).reshape(1, 1), axis = 0)
        X1 = np.append(X1, X[t, :].reshape(1, dim), axis = 0)

    # Studentize data
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1)

    # # Rescale data
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # X1 = scaler.fit_transform(X1)

    # split the numpy array into train and test data
    X1_train, X1_test = X1[0:(T-tau), :], X1[(T-tau):, :]
    R1_train, R1_test = R1[0:(T-tau)].ravel(), R1[(T-tau):].ravel()

    # Build a Random Forest model
    model = create_RF_model()

    # Define the grid search parameters
    n_estimators = [100, 500, 1000]
    max_depth = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    bootstrap = [True, False]

    param_grid = dict(n_estimators = n_estimators, max_depth = max_depth, bootstrap = bootstrap)

	# use time-series cross-validation
    tscv = TimeSeriesSplit(n_splits = 2, test_size = 10) 

	# Perform grid search. Note the convention that higher score values are better than lower score values
    model_cv = GridSearchCV(model, param_grid = param_grid, cv = tscv, refit = True, scoring = "neg_mean_squared_error", n_jobs = n_jobs)

    # Cross-validate a model by using the grid search
    model_cv.fit(X1_train, R1_train)

     # Forecast the test data
    forecasts = model_cv.predict(X1_test)
    if tau > 1:
        forecast_tau = forecasts[len(forecasts)-1]
    else:
        forecast_tau = forecasts

    # Get the optimal hyperparameters
    opt_params = model_cv.best_params_
    # print(f'Optimal hyperparameters:\n {opt_params}')

    # Create a Random Forest model using the optimal hyperparameters
    best_model = create_RF_model(**opt_params)
    
    # Calculate RMSE and MAE for the validation data
    scores = cross_validate(best_model, X1_train, R1_train, cv = tscv, scoring = ('neg_root_mean_squared_error', 'neg_mean_absolute_error'), n_jobs = n_jobs)
    rmse = -np.mean(scores['test_neg_root_mean_squared_error'])
    mae = -np.mean(scores['test_neg_mean_absolute_error'])

    del model, model_cv, best_model # delete all models
    gc.collect()

    # output results
    return float(forecast_tau), rmse, mae, opt_params

# Create a XGBoost model
def create_XGB_model(booster = 'gbtree', 
                                        colsample_bynode = 0.6, # subsample ratio of columns for each node (split).
                                        colsample_bytree = 0.7, # subsample ratio of columns when constructing each tree
                                        max_depth = 5, # maximum depth of a tree
                                        min_child_weight = 20, # minimum sum of instance weight (hessian) needed in a child
                                        n_estimators = 100, # number of gradient boosted trees
                                        reg_alpha = 0, # L1regularization parameter on weights
                                        reg_lambda = 1, # L2 regularization parameter on weights
                                        subsample = 0.5): # subsample ratio of the training instances
    model = XGBRegressor(booster=booster, objective ='reg:squarederror', seed=1234, n_estimators=n_estimators, max_depth=max_depth, \
                                            reg_alpha=reg_alpha, reg_lambda=reg_lambda, min_child_weight=min_child_weight, subsample=subsample, \
                                            colsample_bytree=colsample_bytree, colsample_bynode=colsample_bynode)
    return model

# Forecast with XGBoost
def XGBf(R, X, tau: int, n_jobs = 1):
    assert (R.shape[0] == X.shape[0]), "numbers of rows not match!"
    assert(tau > 0), "the forecast horizon must be greater than zero!"
    
    T = X.shape[0]
    dim = X.shape[1]
    # R = R.flatten()
    # X = X.flatten() # flatten arrays

    R1 = np.empty( shape = (0, 1) )
    X1 = np.empty( shape = (0, dim) )
    for t in np.arange(0, T):
        if t < T-tau:
            R1 = np.append(R1, R[t+tau].reshape(1, 1), axis = 0)
        else:
            R1 = np.append(R1, np.array([0]).reshape(1, 1), axis = 0)
        X1 = np.append(X1, X[t, :].reshape(1, dim), axis = 0)

    # Studentize data
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1)

    # # Rescale data
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # X1 = scaler.fit_transform(X1)

    # split the numpy array into train and test data
    X1_train, X1_test = X1[0:(T-tau), :], X1[(T-tau):, :]
    R1_train, R1_test = R1[0:(T-tau)].ravel(), R1[(T-tau):].ravel()

    # Build a XGB model
    model = create_XGB_model()

    # Define the grid search parameters
    booster = ['gbtree', 'dart'] 
    colsample_bynode = [0.6, 0.8, 1.0] # subsample ratio of columns for each node (split).
    colsample_bytree = [0.7, 0.8, 0.9, 1.0] # subsample ratio of columns when constructing each tree
    max_depth = [5, 10, 15, 20] # maximum depth of a tree
    min_child_weight = [0.01, 0.1, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0] # minimum sum of instance weight (hessian) needed in a child
    n_estimators = [10] # number of gradient boosted trees
    reg_alpha = [0.10] # [0.001, 0.01, 0.1] # L1regularization parameter on weights
    reg_lambda = [0.10] # [0.001, 0.01, 0.1] # L2 regularization parameter on weights
    subsample = [0.8] # [0.6, 0.8, 1.0]

    param_grid = dict(booster = booster, colsample_bynode = colsample_bynode, colsample_bytree = colsample_bytree, max_depth = max_depth, \
                                    min_child_weight = min_child_weight, n_estimators = n_estimators, reg_alpha = reg_alpha, reg_lambda = reg_lambda, \
                                    subsample = subsample)

	# use time-series cross-validation
    tscv = TimeSeriesSplit(n_splits = 2, test_size = 10) 

	# Perform grid search. Note the convention that higher score values are better than lower score values
    model_cv = GridSearchCV(model, param_grid = param_grid, cv = tscv, scoring = "neg_mean_squared_error", refit = True, n_jobs = n_jobs)

    # Cross-validate a model by using the grid search
    model_cv.fit(X1_train, R1_train)

     # Forecast the test data
    forecasts = model_cv.predict(X1_test)
    if tau > 1:
        forecast_tau = forecasts[len(forecasts)-1]
    else:
        forecast_tau = forecasts

    # Get the optimal hyperparameters
    opt_params = model_cv.best_params_
    # print(f'Optimal hyperparameters:\n {opt_params}')

    # Create a XGB model using the optimal hyperparameters
    best_model = create_XGB_model(**opt_params)

    # Calculate RMSE and MAE for the validation data
    scores = cross_validate(best_model, X1_train, R1_train, cv = tscv, scoring = ('neg_root_mean_squared_error', 'neg_mean_absolute_error'), n_jobs = n_jobs)
    rmse = -np.mean(scores['test_neg_root_mean_squared_error'])
    mae = -np.mean(scores['test_neg_mean_absolute_error'])

    del model, model_cv, best_model # delete all models
    gc.collect()

    # output results
    return float(forecast_tau), rmse, mae, opt_params

# Create a Facebook Prophet model
def create_Prophet_model( seasonality_mode = 'additive', # specify how seasonality components should be integrated with the predictions ('additive' or 'multiplicative')
                                            seasonality_prior_scale = 20, # specify how flexible the seasonality components are allowed to be
                                            n_changepoints = 5, # number of change points to be automatically included
                                            changepoint_prior_scale = 20, # specify how flexible the changepoints are allowed to be
                                            fourier_order_quarter = 5, # number of Fourier components that each quarterly seasonality is composed of
                                            fourier_order_year = 20, # number of Fourier components that each yearly seasonality is composed of
                                            n_regressors = 3 # number of regressors to be added
                                            ):
    model = Prophet(growth = 'linear', 
                                seasonality_mode = seasonality_mode, 
                                seasonality_prior_scale = seasonality_prior_scale,
                                n_changepoints = n_changepoints,
                                changepoint_prior_scale = changepoint_prior_scale,
                                yearly_seasonality = False, 
                                weekly_seasonality=False, 
                                daily_seasonality=False)
    # model.add_seasonality(name='quarterly', period=365.25/4, fourier_order=fourier_order_quarter)
    # model.add_seasonality(name='yearly', period=365.25, fourier_order=fourier_order_year)
    model.add_country_holidays(country_name='US') # adding US/CA holiday regressor
    for i in range(n_regressors):
        model.add_regressor(f'x{i+1}') # adding all regressors
    return model

# Calculate the OoS RMSE  and MAE of forecasts made by a Facebook Prophet model
def perf_Prophet( df: pd.DataFrame, # a dataframe starting with a date column, then a column of labels, and columns of predictors 
                                freq = '3M', # data frequency (default: quarterly)
                                seasonality_mode = 'additive', # specify how seasonality components should be integrated with the predictions ('additive' or 'multiplicative')
                                seasonality_prior_scale = 20, # specify how flexible the seasonality components are allowed to be
                                n_changepoints = 5, # number of change points to be automatically included
                                changepoint_prior_scale = 20, # specify how flexible the changepoints are allowed to be
                                fourier_order_quarter = 5, # number of Fourier components that each quarterly seasonality is composed of
                                fourier_order_year = 20, # number of Fourier components that each yearly seasonality is composed of
                                ):
    """
        output: OoS RMSE and MAE
    """

    # Split the dataframe into training and validation data
    T = df.shape[0] # number of time periods
    dim = df.shape[1] - 2 # number of predictors
    sratio = 0.8 # proportion of the sample used to train model

    df = df.copy()
    train_df = df.iloc[0:math.floor(sratio*T), :]
    valid_df = df.iloc[math.floor(sratio*T):T, :]

    # Define a Prophet model
    model = Prophet(growth = 'linear', 
                                seasonality_mode = seasonality_mode, 
                                seasonality_prior_scale = seasonality_prior_scale,
                                n_changepoints = n_changepoints,
                                changepoint_prior_scale = changepoint_prior_scale,
                                yearly_seasonality = False, 
                                weekly_seasonality=False, 
                                daily_seasonality=False)
    # model.add_seasonality(name='quarterly', period=365.25/4, fourier_order=fourier_order_quarter)
    # model.add_seasonality(name='yearly', period=365.25, fourier_order=fourier_order_year)
    model.add_country_holidays(country_name='US') # adding US/CA holiday regressor
    for i in range(dim):
        model.add_regressor(f'x{i+1}') # adding all regressors
    
    # Train the model
    with suppress_stdout_stderr():
        model.fit(train_df) 
	# coefficients = regressor_coefficients(m)
	# print(coefficients)

    # Compute the RMSE of the forecasts of validation data
    future = model.make_future_dataframe(periods = len(valid_df),  freq = freq, include_history = True)
    df.set_index('ds', inplace = True)
    futures = utils_fprophet.add_regressor_to_future(future, [df[f'x{i+1}'] for i in range(dim)])
    # print( futures.head() )
    forecasts = model.predict(futures)

    df = pd.merge(df.reset_index(), forecasts, on='ds')
    residuals = df['yhat'] - df['y']

    # return RMSE and MAE
    return np.sqrt( np.mean(residuals.iloc[math.floor(sratio*T):T]**2) ), np.mean( np.abs(residuals.iloc[math.floor(sratio*T):T]) )


# Forecast with Facebook Prophet
def prophetf(R: np.array, X: np.array, start_date: str, freq: str, tau):
    """   start_date: a string of form 'mm/dd/yyyy'
            freq: a string of form '3M' or '1Y' or '1D' etc. 
            tau: a forecast horizon
            Note that this model uses the US holidays [add_country_holidays(country_name='US') ], which must be modified accordingly. 
    """
    assert (R.shape[0] == X.shape[0]), "numbers of rows not match!"
    T = X.shape[0]
    dim = X.shape[1]
    # R = R.flatten()
    # X = X.flatten() # flatten arrays

    R1 = np.empty( shape = (0, 1) )
    X1 = np.empty( shape = (0, dim) )
    for t in np.arange(0, T):
        if t < T-tau:
            R1 = np.append(R1, R[t+tau].reshape(1, 1), axis = 0)
        else:
            R1 = np.append(R1, np.array([0]).reshape(1, 1), axis = 0)
        X1 = np.append(X1, X[t, :].reshape(1, dim), axis = 0)

    # Studentize data
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1)

    # # Rescale data
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # X1 = scaler.fit_transform(X1)

    # put all numpy arrays into a pandas dataframe
    data_pd = pd.DataFrame(data = np.concatenate( (R1, X1), axis=1 ), index=[i for i in range(R1.shape[0])], columns=['x'+str(i) for i in range(dim+1)])
    data_pd.rename(columns = {'x0': 'y'}, inplace = True)
    # create a dummy date column
    data_pd['ds'] = pd.date_range(start=start_date, periods=len(data_pd), freq=freq)
    # print( data_pd.head() )

    # split the dataframe into train and test data
    data_train = data_pd.iloc[0:(T-tau), :].copy()
    data_test = data_pd.iloc[(T-tau):T, :].copy()

    # Define the grid search parameters
    seasonality_mode = ['additive', 'multiplicative'] # specify how seasonality components should be integrated with the predictions ('additive' or 'multiplicative')
    seasonality_prior_scale = [20] # specify how flexible the seasonality components are allowed to be
    n_changepoints = [5, 10, 20] # number of change points to be automatically included
    changepoint_prior_scale = [5] # specify how flexible the changepoints are allowed to be
    fourier_order_quarter = [10] # number of Fourier components that each quarterly seasonality is composed of
    fourier_order_year = [10] # number of Fourier components that each yearly seasonality is composed of

    param_grid = dict(  seasonality_mode = seasonality_mode, seasonality_prior_scale = seasonality_prior_scale, \
                                    n_changepoints = n_changepoints, changepoint_prior_scale = changepoint_prior_scale, fourier_order_quarter = fourier_order_quarter, \
                                    fourier_order_year = fourier_order_year)

    grid = ParameterGrid(param_grid)
    # print(grid)

    # Perform grid search for hyperparameters with lowest RMSE on a validation subset
    rmse_ls, mae_ls, params_ls = [], [], []
    for params in grid:
        rmse, mae = perf_Prophet(data_train, freq, **params)
        rmse_ls.append(rmse)
        mae_ls.append(mae)
        params_ls.append(params)
    perf_df = pd.DataFrame({'rmse': rmse_ls, 'mae': mae_ls, 'params': params_ls})
    perf_df.sort_values(by = 'rmse', inplace = True)
    # display(perf_df)

    # Create a Prophet model using the optimal hyperparameters
    opt_params = dict(perf_df.iloc[0, 2])
    model = create_Prophet_model(**opt_params, n_regressors = dim)

    # Train this model
    with suppress_stdout_stderr():
        model.fit(data_train)

    # Forecast the test data
    future = model.make_future_dataframe(periods = len(data_test),  freq = freq, include_history = True)
    data_pd.set_index('ds', inplace = True)
    futures = utils_fprophet.add_regressor_to_future(future, [data_pd[f'x{i+1}'] for i in range(dim)])
    # print( futures.head() )
    forecast = model.predict(futures)
    # print( forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']] )
    forecast_tau = float( forecast.iloc[T-1] ['yhat'] )

    del data_pd # delete all dataframes
    gc.collect()

    # output results
    return forecast_tau, perf_df.iloc[0, 0].squeeze(), perf_df.iloc[0, 1].squeeze(), opt_params

# Create an ANN model
def create_ANN_model(n_neurons = 50, n_features = 10, dropout = 0.4, l1 = 0.01, l2 = 0.01):
    model = Sequential()

    # define the input layer
    model.add( InputLayer(input_shape=(n_features, ), name='Input_layer') )

    # define three hidden layers
    model.add( Dense(  units = n_neurons, 
                                    activation='relu', 
                                    name='First_hidden_layer',
                                    kernel_regularizer = regularizers.L1L2(l1=l1, l2=l2),
                                    activity_regularizer = regularizers.l1_l2(l1=l1, l2=l2) ) )
    model.add( Dropout(dropout) )
    model.add( Dense(  units = int(n_neurons/2), 
                                    activation='relu', 
                                    name='Second_hidden_layer',
                                    kernel_regularizer = regularizers.L1L2(l1=l1, l2=l2),
                                    activity_regularizer = regularizers.l1_l2(l1=l1, l2=l2) ) )
    model.add( Dropout(dropout/2.) )
    # model.add( Dense( units = int(n_neurons/4), 
    #                                 activation='relu', 
    #                                 name='Third_hidden_layer',
    #                                 kernel_regularizer = regularizers.L1L2(l1=l1, l2=l2),
    #                                 activity_regularizer = regularizers.l1_l2(l1=l1, l2=l2) ) )
    # model.add( Dropout(dropout/4.) )

    # define the output layer
    model.add( Dense(1, name='Output_layer') )

    # compile the ANN model
    model.compile(loss='mean_squared_error', optimizer='adam')
    # model.summary()
    return model

# Forecast with ANN
def ANNf(R: np.array, X: np.array, tau: int, batch_size: int, num_epochs: int, n_jobs = 1):
    assert (R.shape[0] == X.shape[0]), "numbers of rows not match!"
    assert (tau > 0), "the forecast horizon (tau) must be an integer"
    T = X.shape[0]
    dim = X.shape[1]
    # R = R.flatten()
    # X = X.flatten() # flatten arrays

    R1 = np.empty( shape = (0, 1) )
    X1 = np.empty( shape = (0, dim) )
    for t in np.arange(0, T):
        if t < T-tau:
            R1 = np.append(R1, R[t+tau].reshape(1, 1), axis = 0)
        else:
            R1 = np.append(R1, np.array([0]).reshape(1, 1), axis = 0)
        X1 = np.append(X1, X[t, :].reshape(1, dim), axis = 0)

    # Rescale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X1 = scaler.fit_transform(X1)
    # R1 = scaler.fit_transform(R1)

    # Studentize data
    # scaler = StandardScaler()
    # X1 = scaler.fit_transform(X1)
    # R1 = scaler.fit_transform(R1)

    # split the numpy array into train and test data
    if tau == 0:
        X1_train, X1_test = X1[0:(T-1), :], X1[(T-1):, :]
        R1_train, R1_test = R1[0:(T-1), :], R1[(T-1):, :]
    else:
        X1_train, X1_test = X1[0:(T-tau), :], X1[(T-tau):, :]
        R1_train, R1_test = R1[0:(T-tau)], R1[(T-tau):]

    # Build an ANN model
    model = KerasRegressor(build_fn = create_ANN_model, verbose=0)

    # Define the grid search parameters
    n_neurons = [500]
    n_features = [dim] 
    dropout = [0.1, 0.2] 
    l1 = [0.001, 0.01, 0.1]
    l2 = [0.001, 0.01, 0.1]

    param_grid = dict(n_neurons = n_neurons, n_features = n_features, dropout = dropout, l1 = l1, l2 = l2)

	# use time-series cross-validation
    tscv = TimeSeriesSplit(n_splits = 2, test_size = 10) 

	# Perform grid search. Note the convention that higher score values are better than lower score values
    model_cv = GridSearchCV(model, param_grid = param_grid, cv = tscv, scoring = "neg_mean_squared_error", refit = True, n_jobs = n_jobs, verbose=True)

    # Cross-validate a model by using the grid search
    model_cv.fit(X1_train, R1_train, epochs=num_epochs, batch_size=batch_size, verbose=0) # Verbosity mode 0: silent

     # Forecast the test data
    forecasts = model_cv.predict(X1_test)
    if tau > 1:
        forecast_tau = forecasts[len(forecasts)-1]
    else:
        forecast_tau = forecasts

    # Get the optimal hyperparameters
    opt_params = model_cv.best_params_
    # print(f'Optimal hyperparameters:\n {opt_params}')

    # Create an ANN model using the optimal hyperparameters
    best_model = KerasRegressor(build_fn = create_ANN_model, **opt_params, verbose = 0)
    
    # Calculate RMSE and MAE for the validation data
    scores = cross_validate(best_model, X1_train, R1_train, cv = tscv, scoring = ('neg_root_mean_squared_error', 'neg_mean_absolute_error'), n_jobs = n_jobs)
    rmse = -np.mean(scores['test_neg_root_mean_squared_error'])
    mae = -np.mean(scores['test_neg_mean_absolute_error'])

    K.clear_session()
    del model, model_cv, best_model # delete all models
    gc.collect()

    # output results
    return float(forecast_tau), rmse, mae, opt_params

# Create a LSTM model
def create_LSTM_model(n_neurons = 10, n_steps = 1, n_features = 10, dropout = 0.2, l1 = 0.01, l2 = 0.01):
    # # create the model
    model = Sequential()
    model.add(LSTM(  units = n_neurons, \
                                    activation = 'relu',
                                    input_shape = (n_steps, n_features), \
                                    return_sequences = True,  # set this option to 'True' except the last LSTM layer
                                    activity_regularizer = regularizers.l1_l2(l1=l1, l2=l2), \
                                    recurrent_regularizer = regularizers.l1_l2(l1=l1, l2=l2) ) )
    model.add( Dropout(dropout) )
    model.add( LSTM(units = 2*n_neurons, activation = 'relu', return_sequences = False) )
    model.add( Dropout(2*dropout) )
    # model.add( LSTM(units = 4*n_neurons, activation = 'relu') )
    # model.add( Dropout(3*dropout) )
    model.add( Dense(units = 1) ) # , activation = 'relu'

    # compile the LSTM model
    model.compile(loss='mean_squared_error', optimizer='adam')
    # model.summary()
    return model

# Calculate the OoS RMSE  and MAE of forecasts made by a LSTM model
def perf_LSTM(R: np.array, X: np.array, n_neurons = 500, dropout = 0.2, l1 = 0.01, l2 = 0.01, batch_size = 40, num_epochs = 1):
    """
        Input:
            R: A numpy array of dimension (nobs, 1)
            X: A numpy array of dimension (nobs, n_steps, n_features) 
        Output: OoS RMSE and MAE
    """
    T = R.shape[0]
    n_steps = X.shape[1]
    n_features = X.shape[2]
    assert(T == X.shape[0]), "the dimensions of matrices do not match!"

    # Split the dataframe into training and validation data
    sratio = 0.8 # proportion of the sample used to train model
    X_train1, X_valid1 = X[0:math.floor(sratio*T), :, :],  X[math.floor(sratio*T):T, :, :]
    R_train1, R_valid1 = R[0:math.floor(sratio*T), :], R[math.floor(sratio*T):T, :]

    # Define a LSTM model
    params_dict = dict(n_neurons = n_neurons, n_steps = n_steps, n_features = n_features, dropout = dropout, l1 = l1, l2 = l2)
    model = KerasRegressor(build_fn =create_LSTM_model, **params_dict, verbose = 0)
    
    # Train the model
    model.fit(X_train1, R_train1, epochs=num_epochs, batch_size=batch_size, verbose=0) # Verbosity mode 0: silent
	# coefficients = regressor_coefficients(m)
	# print(coefficients)

    # Compute the errors of forecasts of validation data
    R_pred1 = model.predict(X_valid1)
    residuals = R_valid1 - R_pred1.reshape(-1, 1)
    # print(residuals.shape)

    # return RMSE and MAE
    return np.sqrt( np.mean(residuals**2) ), np.mean( np.abs(residuals) )

# Compute forecasts using a LSTM model
def LSTMf(R: np.array, X: np.array, tau: int, batch_size: int, num_epochs: int, n_jobs = 1):
    assert (R.shape[0] == X.shape[0]), "numbers of rows not match!"
    assert (tau > 0), "the forecast horizon must be greater than zero!"

    T = X.shape[0]
    dim = X.shape[1]
    # R = R.flatten()
    # X = X.flatten() # flatten arrays

    R1 = np.empty( shape = (0, 1) )
    X1 = np.empty( shape = (0, dim) )
    for t in np.arange(0, T):
        if t < T-tau:
            R1 = np.append(R1, R[t+tau].reshape(1, 1), axis = 0)
        else:
            R1 = np.append(R1, np.array([0]).reshape(1, 1), axis = 0)
        X1 = np.append(X1, X[t, :].reshape(1, dim), axis = 0)

    # Rescale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X1 = scaler.fit_transform(X1)
    # R1 = scaler.fit_transform(R1)

    # Studentize data
    # scaler = StandardScaler()
    # X1 = scaler.fit_transform(X1)
    # R1 = scaler.fit_transform(R1)

    # Choose a number of time steps
    n_steps = 3

    # Split the numpy array into train and test data
    X1_train, X1_test = X1[0:(T-n_steps), :], X1[(T-n_steps):, :]
    R1_train, R1_test = R1[0:(T-n_steps), :], R1[(T-n_steps):, :]

  # Create arrays that conforms with the format of LSTM
    X_train, R_train = create_dataset2(R1_train, X1_train, n_steps)
    X_test, R_test = create_dataset2(R1_test, X1_test, n_steps)
    n_features = X_train.shape[2]
    # print(X_test.shape)

    # Build a LSTM model
    model = KerasRegressor(build_fn = create_LSTM_model, verbose=0)

    # Define the grid search parameters
    n_neurons = [500]
    n_steps_ls = [n_steps] 
    n_features = [n_features] 
    dropout = [0.01, 0.1, 0.2] 
    l1 = [0.001, 0.01, 0.1]
    l2 = [0.001, 0.01, 0.1]

    param_grid = dict(n_neurons = n_neurons, n_steps = n_steps_ls, n_features = n_features, dropout = dropout, l1 = l1, l2 = l2)

	# use time-series cross-validation
    tscv = TimeSeriesSplit(n_splits = 2, test_size = 10) 

	# Perform grid search. Note the convention that higher score values are better than lower score values
    model_cv = GridSearchCV(model, param_grid = param_grid, cv = tscv, scoring = "neg_mean_squared_error", refit = True, n_jobs = n_jobs, verbose=False)

    # Cross-validate a model by using the grid search
    model_cv.fit(X_train, R_train, epochs=num_epochs, batch_size=batch_size, verbose=0) # Verbosity mode 0: silent

     # Forecast the test data
    forecasts = model_cv.predict(X_test)
    if  (X1_test.shape[0] > n_steps):
        forecast_tau = forecasts[len(forecasts)-1]
    else:
        forecast_tau = forecasts

    # Get the optimal hyperparameters
    opt_params = model_cv.best_params_
    # print(f'Optimal hyperparameters:\n {opt_params}')

    # Create a LSTM model using the optimal hyperparameters
    best_model = KerasRegressor(build_fn =create_LSTM_model, **opt_params, verbose = 0)
    
    # Calculate RMSE and MAE for the validation data
    scores = cross_validate(best_model, X_train, R_train, cv = tscv, scoring = ('neg_root_mean_squared_error', 'neg_mean_absolute_error'), n_jobs = n_jobs)
    rmse = -np.mean(scores['test_neg_root_mean_squared_error'])
    mae = -np.mean(scores['test_neg_mean_absolute_error'])

    K.clear_session()
    del model, model_cv, best_model # delete all models
    gc.collect()

    # output results
    return float(forecast_tau), rmse, mae, opt_params
