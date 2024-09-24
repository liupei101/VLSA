import numpy as np
import pandas as pd
from typing import Optional
import warnings
from tqdm import trange

from .custom_types import NumericArrayLike
from .util import (check_and_convert, KaplanMeierArea, km_mean,
                   predict_mean_survival_time, predict_median_survival_time)


def mae_pycox(
        predicted_survival_curves: pd.DataFrame,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        train_event_times: Optional[NumericArrayLike] = None,
        train_event_indicators: Optional[NumericArrayLike] = None,
        method: str = "Hinge",
        log_scale: bool = False,
        predicted_time_method: str = "Median"
) -> float:
    """
    Calculate the MAE score for the predicted survival curves from PyCox models.
    param predicted_survival_curves: pd.DataFrame, shape = (n_samples, n_times)
        Predicted survival curves for the testing samples
        DataFrame index represents the time coordinates for the given curves.
        DataFrame value represents the survival probabilities.
    param event_times: structured array, shape = (n_samples, )
        Actual event/censor time for the testing samples.
    param event_indicators: structured array, shape = (n_samples, )
        Binary indicators of censoring for the testing samples
    param train_event_times:structured array, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    param train_event_indicators: structured array, shape = (n_train_samples, )
        Binary indicators of censoring for the training samples
    param method: string, default: "Hinge"
    param log_scale: boolean, default: False
    param predicted_time_method: string, default: "Median"
    :return:
        Value for the calculated MAE score.
    """
    warnings.warn("This function is deprecated and might be deleted in the future. "
                  "Please use the class 'PyCoxEvaluator' from Evaluator.py.", DeprecationWarning)
    event_times, event_indicators = check_and_convert(event_times, event_indicators)
    if (train_event_times is not None) and (train_event_indicators is not None):
        train_event_times, train_event_indicators = check_and_convert(train_event_times, train_event_indicators)
    # Extracting the time buckets
    time_coordinates = predicted_survival_curves.index.values
    # computing the Survival function, and set the small negative value to zero
    survival_curves = predicted_survival_curves.values.T
    survival_curves[survival_curves < 0] = 0

    if predicted_time_method == "Median":
        predict_method = predict_median_survival_time
    elif predicted_time_method == "Mean":
        predict_method = predict_mean_survival_time
    else:
        error = "Please enter one of 'Median' or 'Mean' for calculating predicted survival time."
        raise TypeError(error)

    # get median/mean survival time from the predicted curve
    predicted_times = []
    for i in range(survival_curves.shape[0]):
        predicted_time = predict_method(survival_curves[i, :], time_coordinates)
        predicted_times.append(predicted_time)
    predicted_times = np.array(predicted_times)
    return mean_error(predicted_times, event_times, event_indicators, train_event_times,
                      train_event_indicators, "absolute", method, True, log_scale)


def mae_sksurv(
        predicted_survival_curves: pd.DataFrame,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        train_event_times: Optional[NumericArrayLike] = None,
        train_event_indicators: Optional[NumericArrayLike] = None,
        method: str = "Hinge",
        log_scale: bool = False,
        predicted_time_method: str = "Median"
) -> float:
    """
    Calculate the MAE score for the predicted survival curves from scikit-survival models.
    param predicted_survival_curves: pd.DataFrame, shape = (n_samples, n_times)
        Predicted survival curves for the testing samples
        DataFrame index represents the time coordinates for the given curves.
        DataFrame value represents the survival probabilities.
    param event_times: structured array, shape = (n_samples, )
        Actual event/censor time for the testing samples.
    param event_indicators: structured array, shape = (n_samples, )
        Binary indicators of censoring for the testing samples
    param train_event_times:structured array, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    param train_event_indicators: structured array, shape = (n_train_samples, )
        Binary indicators of censoring for the training samples
    param method: string, default: "Hinge"
    param log_scale: boolean, default: False
    param predicted_time_method: string, default: "Median"
    :return:
        Value for the calculated MAE score.
    """
    warnings.warn("This function is deprecated and might be deleted in the future. "
                  "Please use the class 'ScikitSurvivalEvaluator' from Evaluator.py.", DeprecationWarning)
    event_times, event_indicators = check_and_convert(event_times, event_indicators)
    if (train_event_times is not None) and (train_event_indicators is not None):
        train_event_times, train_event_indicators = check_and_convert(train_event_times, train_event_indicators)

    if predicted_time_method == "Median":
        predict_method = predict_median_survival_time
    elif predicted_time_method == "Mean":
        predict_method = predict_mean_survival_time
    else:
        error = "Please enter one of 'Median' or 'Mean' for calculating predicted survival time."
        raise TypeError(error)

    # get median/mean survival time from the predicted curve
    predicted_times = []
    for i in range(predicted_survival_curves.shape[0]):
        predicted_time = predict_method(predicted_survival_curves[i].y, predicted_survival_curves[i].x)
        predicted_times.append(predicted_time)
    predicted_times = np.array(predicted_times)
    return mean_error(predicted_times, event_times, event_indicators, train_event_times,
                      train_event_indicators, "absolute", method, True, log_scale)


def mean_error(
        predicted_times: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        train_event_times: Optional[np.ndarray] = None,
        train_event_indicators: Optional[np.ndarray] = None,
        error_type: str = "absolute",
        method: str = "Hinge",
        weighted: bool = True,
        log_scale: bool = False,
        reduction: bool = True,
        verbose: bool = False
) -> float:
    """
    Calculate the mean absolute/squared error score for the predicted survival times.
    Parameters
    ----------
    predicted_times: np.ndarray, shape = (n_samples, )
        Predicted survival times for the testing samples
    event_times: np.ndarray, shape = (n_samples, )
        Actual event/censor time for the testing samples.
    event_indicators: np.ndarray, shape = (n_samples, )
        Binary indicators of censoring for the testing samples
    train_event_times: np.ndarray, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    train_event_indicators: np.ndarray, shape = (n_train_samples, )
        Binary indicators of censoring for the training samples
    error_type: string, default: "absolute"
        Type of mean error to use. Options are "absolute" and "squared".
    method: string, default: "Hinge"
        Method of handling censorship.
        Options are "Uncensored", "Hinge", "Margin", "IPCW-v1", "IPCW-v2", "Pseudo_obs", and "Pseudo_obs_pop"
    weighted: boolean, default: True
        Whether to use weighting scheme for MAE.
        If true, each best guess value / surrogate value will have a confidence weight = 1/ (1 - KM(censoring time)).
    log_scale: boolean, default: False
        Whether to use log scale for the loss function.
    verbose: boolean, default: False
        Whether to show the progress bar.

    Returns
    -------
    Value for the calculated MAE score.
    """
    event_indicators = event_indicators.astype(bool)
    n_test = event_times.size
    if train_event_indicators is not None:
        train_event_indicators = train_event_indicators.astype(bool)

    # calculate the weighting for each sample
    if method in ["Margin", "IPCW-v1", "IPCW-v2", "Pseudo_obs", "Pseudo_obs_pop"]:
        if train_event_times is None or train_event_indicators is None:
            raise ValueError("If method is '{}', training set values must be included.".format(method))

        km_model = KaplanMeierArea(train_event_times, train_event_indicators)
        km_linear_zero = km_model.km_linear_zero
        if np.isinf(km_linear_zero):
            km_linear_zero = max(km_model.survival_times)
        # predicted_times = np.clip(predicted_times, a_max=km_linear_zero, a_min=None)

        censor_times = event_times[~event_indicators]
        weights = np.ones(n_test)
        if weighted:
            weights[~event_indicators] = 1 - km_model.predict(censor_times)

    # set the error function
    if error_type == "absolute":
        error_func = np.abs
    elif error_type == "squared":
        error_func = np.square
    else:
        raise TypeError("Please enter one of 'absolute' or 'squared' for calculating error.")

    if method == "Uncensored":
        # only use uncensored data
        if log_scale:
            errors = np.log(event_times[event_indicators]) - np.log(predicted_times[event_indicators])
        else:
            errors = event_times[event_indicators] - predicted_times[event_indicators]
        if not reduction:
            return error_func(errors)
        return error_func(errors).mean()
    elif method == "Hinge":
        # consider only the early prediction error
        # if prediction is higher than the censored time, it is not penalized
        weights = np.ones(predicted_times.size)
        if weighted:
            if train_event_times is None or train_event_indicators is None:
                raise ValueError("If 'weighted' is True for calculating Hinge, training set values must be included.")
            km_model = KaplanMeierArea(train_event_times, train_event_indicators)
            censor_times = event_times[~event_indicators]
            weights[~event_indicators] = 1 - km_model.predict(censor_times)

        if log_scale:
            errors = np.log(event_times) - np.log(predicted_times)
        else:
            errors = event_times - predicted_times
        errors[~event_indicators] = np.maximum(errors[~event_indicators], 0)
        if not reduction:
            return error_func(errors)
        return np.average(error_func(errors), weights=weights)
    elif method == "Margin":
        # The L1-margin method proposed by https://www.jmlr.org/papers/v21/18-772.html
        # Calculate the best guess survival time given the KM curve and censoring time of that patient
        best_guesses = km_model.best_guess(censor_times)
        best_guesses[censor_times > km_linear_zero] = censor_times[censor_times > km_linear_zero]

        errors = np.empty(predicted_times.size)
        if log_scale:
            errors[event_indicators] = (np.log(event_times[event_indicators]) -
                                        np.log(predicted_times[event_indicators]))
            errors[~event_indicators] = np.log(best_guesses) - np.log(predicted_times[~event_indicators])
        else:
            errors[event_indicators] = event_times[event_indicators] - predicted_times[event_indicators]
            errors[~event_indicators] = best_guesses - predicted_times[~event_indicators]
        if not reduction:
            return error_func(errors)
        return np.average(error_func(errors), weights=weights)
    elif method == "IPCW-v1":
        # This is the IPCW-T method from https://arxiv.org/pdf/2306.01196.pdf
        # Calculate the best guess time (surrogate time) based on the subsequent uncensored subjects
        best_guesses = np.empty(shape=n_test)
        for i in range(n_test):
            if event_indicators[i] == 1:
                best_guesses[i] = event_times[i]
            else:
                # Numpy will throw a warning if afterward_event_times are all false. TODO: consider change the code.
                afterward_event_idx = train_event_times[train_event_indicators == 1] > event_times[i]
                best_guesses[i] = np.mean(train_event_times[train_event_indicators == 1][afterward_event_idx])
        # NaN values are generated because there are no events after the censor times
        nan_idx = np.argwhere(np.isnan(best_guesses))
        predicted_times = np.delete(predicted_times, nan_idx)
        best_guesses = np.delete(best_guesses, nan_idx)
        weights = np.delete(weights, nan_idx)
        if log_scale:
            errors = np.log(best_guesses) - np.log(predicted_times)
        else:
            errors = best_guesses - predicted_times
        if not reduction:
            return error_func(errors)
        return np.average(error_func(errors), weights=weights)
    elif method == "IPCW-v2":
        # This is the IPCW-D method from https://arxiv.org/pdf/2306.01196.pdf
        # Using IPCW weights to transfer the censored subjects to uncensored subjects
        inverse_train_event_indicators = 1 - train_event_indicators

        ipc_model = KaplanMeierArea(train_event_times, inverse_train_event_indicators)
        ipc_pred = ipc_model.predict(event_times)
        # Catch if denominator is 0. This happens when the time is later than the last event time in trainset.
        ipc_pred[ipc_pred == 0] = np.inf
        if log_scale:
            errors = np.log(event_times) - np.log(predicted_times)
        else:
            errors = event_times - predicted_times
        if not reduction:
            return error_func(errors)[event_indicators] / ipc_pred[event_indicators]
        return (error_func(errors)[event_indicators] / ipc_pred[event_indicators]).mean()
    elif method == "Pseudo_obs":
        # Calculate the best guess time (surrogate time) by the contribution of the censored subjects to KM curve
        n_train = train_event_times.size

        events, population_counts = km_model.events.copy(), km_model.population_count.copy()
        times = km_model.survival_times.copy()
        probs = km_model.survival_probabilities.copy()
        # get the discrete time points where the event happens, then calculate the area under those discrete time only
        # this doesn't make any difference for step function, but it does for trapezoid rule.
        unique_idx = np.where(events != 0)[0]
        if unique_idx[-1] != len(events) - 1:
            unique_idx = np.append(unique_idx, len(events) - 1)
        times = times[unique_idx]
        population_counts = population_counts[unique_idx]
        events = events[unique_idx]
        probs = probs[unique_idx]
        sub_expect_time = km_mean(times.copy(), probs.copy())

        # use the idea of dynamic programming to calculate the multiplier of the KM estimator in advances.
        # if we add a new time point to the KM curve, the multiplier before the new time point will be
        # 1 - event_counts / (population_counts + 1), and the multiplier after the new time point will be
        # the same as before.
        multiplier = 1 - events / population_counts
        multiplier_total = 1 - events / (population_counts + 1)
        best_guesses = event_times.copy().astype(float)

        for i in trange(n_test, desc="Calculating surrogate times for MAE-PO", disable=not verbose):
            if event_indicators[i] != 1:
                total_multiplier = multiplier.copy()
                insert_index = np.searchsorted(times, event_times[i], side='right')
                total_multiplier[:insert_index] = multiplier_total[:insert_index]
                survival_probabilities = np.cumprod(total_multiplier)
                if insert_index == len(times):
                    times_addition = np.append(times, event_times[i])
                    survival_probabilities_addition = np.append(survival_probabilities, survival_probabilities[-1])
                    total_expect_time = km_mean(times_addition, survival_probabilities_addition)
                else:
                    total_expect_time = km_mean(times, survival_probabilities)
                best_guesses[i] = (n_train + 1) * total_expect_time - n_train * sub_expect_time
        if log_scale:
            errors = np.log(best_guesses) - np.log(predicted_times)
        else:
            errors = best_guesses - predicted_times
        if not reduction:
            return error_func(errors)
        return np.average(error_func(errors), weights=weights)
    elif method == "Pseudo_obs_pop":
        # This is the MAE-Pop-PO method from https://arxiv.org/pdf/2306.01196.pdf
        # The best guess time is the same for all subjects, which is the mean of the KM curve.
        sub_expect_time = km_model.mean
        best_guesses = event_times.copy()
        best_guesses[~event_indicators] = sub_expect_time

        if log_scale:
            errors = np.log(best_guesses) - np.log(predicted_times)
        else:
            errors = best_guesses - predicted_times
        if not reduction:
            return error_func(errors)
        return np.average(error_func(errors), weights=weights)
    else:
        raise ValueError("Method must be one of 'Uncensored', 'Hinge', 'Margin', 'IPCW-v1', 'IPCW-v2' "
                         "'Pseudo_obs', or 'Pseudo_obs_pop'. Got '{}' instead.".format(method))


def get_best_guess_from_training_data(
        train_event_times: np.ndarray,
        train_event_indicators: np.ndarray,
        method: str = "KM",
        verbose: bool = False
) -> float:
    """
    Calculate the best guess of event times for all the patients from training set.

    Parameters
    ----------
    train_event_times: np.ndarray, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    train_event_indicators: np.ndarray, shape = (n_train_samples, )
        Binary indicators of censoring for the training samples
    method: string, default: "KM"
        Method of handling censorship.
        Options are "KM" and "WKM".
    verbose: boolean, default: False
        Whether to show the progress bar.

    Returns
    -------
    Best guess of event times for all the patients from training set.
    """
    train_event_indicators = train_event_indicators.astype(bool)

    km_model = KaplanMeierArea(train_event_times, train_event_indicators)
    km_linear_zero = km_model.km_linear_zero
    if np.isinf(km_linear_zero):
        km_linear_zero = max(km_model.survival_times)

    # initialize the best guess
    best_guesses = train_event_times.copy().astype(float)
    censor_times = train_event_times[~train_event_indicators]

    if method == 'KM':
        # The L1-margin method proposed by https://www.jmlr.org/papers/v21/18-772.html
        # Calculate the best guess survival time given the KM curve and censoring time of that patient
        censor_best_guesses = km_model.best_guess(censor_times)
        censor_best_guesses[censor_times > km_linear_zero] = censor_times[censor_times > km_linear_zero]

        best_guesses[~train_event_indicators] = censor_best_guesses
        return best_guesses

    elif method == "WKM":
        # Calculate the best guess time (surrogate time) by the contribution of the censored subjects to KM curve
        n_train = train_event_times.size

        events, population_counts = km_model.events.copy(), km_model.population_count.copy()
        times = km_model.survival_times.copy()
        probs = km_model.survival_probabilities.copy()
        # get the discrete time points where the event happens, then calculate the area under those discrete time only
        # this doesn't make any difference for step function, but it does for trapezoid rule.
        unique_idx = np.where(events != 0)[0]
        if unique_idx[-1] != len(events) - 1:
            unique_idx = np.append(unique_idx, len(events) - 1)
        times = times[unique_idx]
        population_counts = population_counts[unique_idx]
        events = events[unique_idx]
        probs = probs[unique_idx]
        sub_expect_time = km_mean(times.copy(), probs.copy())

        # use the idea of dynamic programming to calculate the multiplier of the KM estimator in advances.
        # if we add a new time point to the KM curve, the multiplier before the new time point will be
        # 1 - event_counts / (population_counts + 1), and the multiplier after the new time point will be
        # the same as before.
        multiplier = 1 - events / population_counts
        multiplier_total = 1 - events / (population_counts + 1)

        for i in trange(n_train, desc="Calculating surrogate times", disable=not verbose):
            if train_event_indicators[i] != 1:
                total_multiplier = multiplier.copy()
                insert_index = np.searchsorted(times, train_event_times[i], side='right')
                total_multiplier[:insert_index] = multiplier_total[:insert_index]
                survival_probabilities = np.cumprod(total_multiplier)
                if insert_index == len(times):
                    times_addition = np.append(times, train_event_times[i])
                    survival_probabilities_addition = np.append(survival_probabilities, survival_probabilities[-1])
                    total_expect_time = km_mean(times_addition, survival_probabilities_addition)
                else:
                    total_expect_time = km_mean(times, survival_probabilities)
                best_guesses[i] = (n_train + 1) * total_expect_time - n_train * sub_expect_time

        return best_guesses
    
    else:
        raise ValueError("Method must be one of 'KM' and 'WKM'. Got '{}' instead.".format(method))


def insert_km(
        survival_times: np.ndarray,
        event_count: np.ndarray,
        as_risk_count: np.ndarray,
        new_t: float,
        new_e: int
) -> (np.ndarray, np.ndarray):
    """
    Insert a new time point into the Kaplan-Meier curve.

    Parameters
    ----------
    survival_times: np.ndarray, shape = (n_samples, )
        Survival times for KM curve of the testing samples
    event_count: np.ndarray, shape = (n_samples, )
        Event count for KM curve of the testing samples at each time point
    as_risk_count: np.ndarray, shape = (n_samples, )
        At-risk count for KM curve of the testing samples at each time point
    new_t: float
        New time point to be inserted
    new_e: int
        New event count to be inserted

    Returns
    -------
    survival_times: np.ndarray, shape = (n_samples, )
        Survival times for KM curve of the testing samples
    survival_probabilities: np.ndarray, shape = (n_samples, )
        Survival probabilities for KM curve of the testing samples
    """
    # Find the index where new_t should be inserted
    insert_index = np.searchsorted(survival_times, new_t)

    # Check if new_t is already at the found index
    if insert_index < len(survival_times) and survival_times[insert_index] == new_t:
        # If new_t is already in the array, increment the event count
        event_count[insert_index] += new_e
        as_risk_count[:insert_index + 1] += 1
    else:
        # Insert new_t into
        survival_times = np.insert(survival_times, insert_index, new_t)
        event_count = np.insert(event_count, insert_index, new_e)
        # if beyond the last time point, manually insert the last at-risk count with 0, then add 1 to all before
        if insert_index == survival_times.size - 1:
            as_risk_count = np.insert(as_risk_count, insert_index, 0)
        else:
            as_risk_count = np.insert(as_risk_count, insert_index, as_risk_count[insert_index])
        as_risk_count[:insert_index + 1] += 1

    event_ratios = 1 - event_count / as_risk_count
    survival_probabilities = np.cumprod(event_ratios)

    return survival_times, survival_probabilities


if __name__ == "__main__":
    # Test the functions
    train_t = np.array([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                        26, 27, 28, 29, 30, 31, 32, 33, 34,  60, 61, 62, 63, 64, 65, 66, 67,
                        74, 75, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
                        98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                        117, 118, 119, 120, 120, 120, 121, 121, 124, 125, 126, 127, 128, 129,
                        136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
                        155, 156, 157, 158, 159, 161, 182, 183, 186, 190, 191, 192, 192, 192,
                        193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 202, 203,
                        204, 202, 203, 204, 212, 213, 214, 215, 216, 217, 222, 223, 224])
    train_e = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                        1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                        0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                        1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                        0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    t = np.array([5, 10, 19, 31, 43, 59, 63, 75, 97, 113, 134, 151, 163, 176, 182, 195, 200, 210, 220])
    e = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
    predict_time = np.array([18, 19, 5, 12, 75, 100, 120, 85, 36, 95, 170, 41, 200, 210, 260, 86, 100, 120, 140])
    mae_score = mean_error(predict_time, t, e, train_t, train_e, method='Pseudo_obs', verbose=True)
    print(mae_score)
