import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import warnings

from .custom_types import NumericArrayLike
from .util import check_and_convert, KaplanMeier, predict_prob_from_curve, predict_multi_probs_from_curve


def single_brier_score_pycox(
        predicted_survival_curves: pd.DataFrame,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        train_event_times: NumericArrayLike,
        train_event_indicators: NumericArrayLike,
        target_time: float = None
) -> float:
    """
    Calculate the Brier score at a specific time.

    The time-dependent Brier score is the mean squared error at time point :math:`t`:

    param predicted_survival_curves: pd.DataFrame, shape = (n_samples, n_times)
        Predicted survival curves for the testing samples
        DataFrame index represents the time coordinates for the given curves.
        DataFrame value represents the survival probabilities.
    param event_times: structured array, shape = (n_samples, )
        Actual event/censor time for the testing samples.
    param event_indicators: structured array, shape = (n_samples, )
        Binary indicators of censoring for the testing samples
    param train_event_times: structured array, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    param train_event_indicators: structured array, shape = (n_train_samples, )
        Binary indicators of censoring for the training samples
    param target_time: float, default: None
        The specific time point for which to estimate the Brier score.
    :return:
        Value of the brier score.
    """
    warnings.warn("This function is deprecated and might be deleted in the future. "
                  "Please use the class 'PyCoxEvaluator' from Evaluator.py.", DeprecationWarning)
    event_times, event_indicators = check_and_convert(event_times, event_indicators)
    train_event_times, train_event_indicators = check_and_convert(train_event_times, train_event_indicators)

    # Extracting the time buckets
    time_coordinates = predicted_survival_curves.index.values
    # computing the Survival function, and set the small negative value to zero
    survival_curves = predicted_survival_curves.values.T
    survival_curves[survival_curves < 0] = 0

    if target_time is None:
        target_time = np.quantile(np.concatenate((event_times, train_event_times)), 0.5)

    predict_probs = []
    for i in range(survival_curves.shape[0]):
        predict_prob = predict_prob_from_curve(survival_curves[i, :], time_coordinates, target_time)
        predict_probs.append(predict_prob)
    predict_probs = np.array(predict_probs)

    return single_brier_score(predict_probs, event_times, event_indicators, train_event_times, train_event_indicators,
                              target_time)


def single_brier_score(
        predict_probs: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        train_event_times: np.ndarray,
        train_event_indicators: np.ndarray,
        target_time: float = None,
        ipcw: bool = True
) -> float:
    """

    param predict_probs: numpy array, shape = (n_samples, )
        Estimated survival probabilities at the specific time for the testing samples.
    param event_times: numpy array, shape = (n_samples, )
        Actual event/censor time for the testing samples.
    param event_indicators: numpy array, shape = (n_samples, )
        Binary indicators of censoring for the testing samples
    param train_event_times:numpy array, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    param train_event_indicators: numpy array, shape = (n_train_samples, )
        Binary indicators of censoring for the training samples
    param target_time: float, default: None
        The specific time point for which to estimate the Brier score.
    param ipcw: boolean, default: True
        Whether to use Inverse Probability of Censoring Weighting (IPCW) in the calculation.
    :return:
        Values of the brier score.
    """
    if target_time is None:
        target_time = np.median(event_times)

    event_indicators = event_indicators.astype(bool)
    train_event_indicators = train_event_indicators.astype(bool)

    if ipcw:
        inverse_train_event_indicators = 1 - train_event_indicators
        ipc_model = KaplanMeier(train_event_times, inverse_train_event_indicators)

        ipc_pred = ipc_model.predict(event_times)
        # Catch if denominator is 0.
        ipc_pred[ipc_pred == 0] = np.inf
        # Category one calculates IPCW weight at observed time point.
        # Category one is individuals with event time lower than the time of interest and were NOT censored.
        weight_cat1 = ((event_times <= target_time) & event_indicators) / ipc_pred
        # Catch if event times goes over max training event time, i.e. predict gives NA
        weight_cat1[np.isnan(weight_cat1)] = 0
        # Category 2 is individuals whose time was greater than the time of interest (singleBrierTime)
        # contain both censored and uncensored individuals.
        weight_cat2 = (event_times > target_time) / ipc_model.predict(target_time)
        # predict returns NA if the passed-in time is greater than any of the times used to build the inverse probability
        # of censoring model.
        weight_cat2[np.isnan(weight_cat2)] = 0
    else:
        weight_cat1 = ((event_times <= target_time) & event_indicators)
        weight_cat2 = (event_times > target_time)

    b_score = (np.square(predict_probs) * weight_cat1 + np.square(1 - predict_probs) * weight_cat2).mean()
    ###########################
    # Here we are ordering event times and then using predict with level.chaos = 1 which returns
    # predictions ordered by time.
    # This is from Haider's code in R but I feel it doesn't need to be ordered by time.
    # Refer above few lines for the justified code
    ###########################
    # order_of_times = np.argsort(event_times)
    # # Catch if event times goes over max training event time, i.e. predict gives NA
    # weight_cat1 = ((event_times[order_of_times] <= target_time) & event_indicators[order_of_times]) /\
    #               ipc_model.predict(event_times[order_of_times])
    # weight_cat1[np.isnan(weight_cat1)] = 0
    # weight_cat2 = (event_times[order_of_times] > target_time) / ipc_model.predict(target_time)
    # weight_cat2[np.isnan(weight_cat2)] = 0
    #
    # survival_curves_ordered = survival_curves[order_of_times, :]
    # predict_probs = []
    # for i in range(survival_curves_ordered.shape[0]):
    #     predict_prob = predict_prob_from_curve(survival_curves_ordered[i, :], time_coordinates,
    #                                            event_times[order_of_times][i])
    #     predict_probs.append(predict_prob)
    # predict_probs = np.array(predict_probs)
    #
    # b_score = np.mean(np.square(predict_probs) * weight_cat1 + np.square(1 - predict_probs) * weight_cat2)
    return b_score


def brier_multiple_points(
        predict_probs_mat: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        train_event_times: np.ndarray,
        train_event_indicators: np.ndarray,
        target_times: np.ndarray,
        ipcw: bool = True
) -> np.ndarray:
    """
    Calculate multiple Brier scores at multiple specific times.

    :param predict_probs_mat: structured array, shape = (n_samples, n_time_points)
        Predicted probability array (2-D) for each instances at each time point.
    :param event_times: structured array, shape = (n_samples, )
        Actual event/censor time for the testing samples.
    :param event_indicators: structured array, shape = (n_samples, )
        Binary indicators of censoring for the testing samples
    :param train_event_times:structured array, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    :param train_event_indicators: structured array, shape = (n_train_samples, )
        Binary indicators of censoring for the training samples
    :param target_times: float, default: None
        The specific time points for which to estimate the Brier scores.
    :param ipcw: boolean, default: True
        Whether to use Inverse Probability of Censoring Weighting (IPCW) in the calculation.
    :return:
        Values of multiple Brier scores.
    """
    if target_times.ndim != 1:
        error = "'time_grids' is not a one-dimensional array."
        raise TypeError(error)

    # bs_points_matrix = np.tile(event_times, (len(target_times), 1))
    target_times_mat = np.repeat(target_times.reshape(1, -1), repeats=len(event_times), axis=0)
    event_times_mat = np.repeat(event_times.reshape(-1, 1), repeats=len(target_times), axis=1)
    event_indicators_mat = np.repeat(event_indicators.reshape(-1, 1), repeats=len(target_times), axis=1)
    event_indicators_mat = event_indicators_mat.astype(bool)

    if ipcw:
        inverse_train_event_indicators = 1 - train_event_indicators

        ipc_model = KaplanMeier(train_event_times, inverse_train_event_indicators)

        # Category one calculates IPCW weight at observed time point.
        # Category one is individuals with event time lower than the time of interest and were NOT censored.
        ipc_pred = ipc_model.predict(event_times_mat)
        # Catch if denominator is 0.
        ipc_pred[ipc_pred == 0] = np.inf
        weight_cat1 = ((event_times_mat <= target_times_mat) & event_indicators_mat) / ipc_pred
        # Catch if event times goes over max training event time, i.e. predict gives NA
        weight_cat1[np.isnan(weight_cat1)] = 0
        # Category 2 is individuals whose time was greater than the time of interest (singleBrierTime)
        # contain both censored and uncensored individuals.
        ipc_target_pred = ipc_model.predict(target_times_mat)
        # Catch if denominator is 0.
        ipc_target_pred[ipc_target_pred == 0] = np.inf
        weight_cat2 = (event_times_mat > target_times_mat) / ipc_target_pred
        # predict returns NA if the passed in time is greater than any of the times used to build
        # the inverse probability of censoring model.
        weight_cat2[np.isnan(weight_cat2)] = 0
    else:
        weight_cat1 = ((event_times_mat <= target_times_mat) & event_indicators_mat)
        weight_cat2 = (event_times_mat > target_times_mat)

    ipcw_square_error_mat = np.square(predict_probs_mat) * weight_cat1 + np.square(1 - predict_probs_mat) * weight_cat2
    brier_scores = np.mean(ipcw_square_error_mat, axis=0)
    return brier_scores


def integrated_brier_score_pycox(
        predicted_survival_curves: pd.DataFrame,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        train_event_times: NumericArrayLike,
        train_event_indicators: NumericArrayLike,
        num_points: int = None,
        draw_figure: bool = False
) -> float:
    """
    Calculate the integrated Brier score (IBS) for a set of predicted survival curves predicted from Pycox model.
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
    param num_points: integer, default: None
        The number of points to calculate the Integrated Brier Score.
        Default None, which will use the test set true event times as the time points.
    param draw_figure: boolean, default: False
        Whether to draw the figure of the Integrated Brier Score.
    :return:
    """
    warnings.warn("This function is deprecated and might be deleted in the future. "
                  "Please use the class 'PyCoxEvaluator' from Evaluator.py.", DeprecationWarning)
    event_times, event_indicators = check_and_convert(event_times, event_indicators)
    train_event_times, train_event_indicators = check_and_convert(train_event_times, train_event_indicators)
    max_target_time = np.amax(np.concatenate((event_times, train_event_times)))

    # If number of target time is not indicated, then we use the censored times obtained from test set
    if num_points is None:
        # test_censor_status = 1 - event_indicators
        censored_times = event_times[event_indicators == 0]
        sorted_censored_times = np.sort(censored_times)
        time_points = sorted_censored_times
        time_range = np.amax(time_points) - np.amin(time_points)
    else:
        time_points = np.linspace(0, max_target_time, num_points)
        time_range = max_target_time

    # Get single brier score from multiple target times, and use trapezoidal integral to calculate ISB.
    #########################
    # solution 1, implemented using metrics multiplication, this is geometrically faster than solution 2
    b_scores = brier_multiple_points_pycox(predicted_survival_curves, event_times, event_indicators,
                                           train_event_times, train_event_indicators, time_points)
    if np.isnan(b_scores).any():
        print("Time-dependent Brier Score contains nan")
        print(b_scores)
    integral_value = trapezoid(b_scores, time_points)
    ibs_score = integral_value / time_range

    if draw_figure:
        plt.plot(time_points, b_scores, 'bo-')
        plt.xlabel('Time')
        plt.ylabel('Brier Score')
        plt.show()
    ##########################
    # solution 2, implemented by iteratively call single_brier_score_pycox()
    # b_scores = []
    # for i in range(len(time_points)):
    #     b_score = single_brier_score_pycox(predicted_survival_curves, event_times, event_indicators,
    #                                        train_event_times, train_event_indicators, time_points[i])
    #     b_scores.append(b_score)
    # b_scores = np.array(b_scores)
    # integral_value = trapezoid(b_scores, time_points)
    # ibs_score = integral_value / time_range

    return ibs_score


def brier_multiple_points_pycox(
        predicted_survival_curves: pd.DataFrame,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        train_event_times: NumericArrayLike,
        train_event_indicators: NumericArrayLike,
        target_times: np.ndarray
) -> np.ndarray:
    """
    Calculate multiple Brier scores at multiple specific times.

    param predicted_survival_curves: pd.DataFrame, shape = (n_time_points, n_samples)
        Predicted survival curves for the testing samples
        DataFrame index represents the time coordinates for the given curves.
        DataFrame value represents transpose of the survival probabilities.
    param event_times: structured array, shape = (n_samples, )
        Actual event/censor time for the testing samples.
    param event_indicators: structured array, shape = (n_samples, )
        Binary indicators of censoring for the testing samples
    param train_event_times:structured array, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    param train_event_indicators: structured array, shape = (n_train_samples, )
        Binary indicators of censoring for the training samples
    param target_times: float, default: None
        The specific time points for which to estimate the Brier scores.
    :return:
        Values of multiple Brier scores.
    """
    warnings.warn("This function is deprecated and might be deleted in the future. "
                  "Please use the class 'PyCoxEvaluator' from Evaluator.py.", DeprecationWarning)
    event_times, event_indicators = check_and_convert(event_times, event_indicators)
    train_event_times, train_event_indicators = check_and_convert(train_event_times, train_event_indicators)
    # Extracting the time buckets
    time_coordinates = predicted_survival_curves.index.values
    # computing the Survival function, and set the small negative value to zero
    survival_curves = predicted_survival_curves.values.T
    survival_curves[survival_curves < 0] = 0

    predict_probs_mat = []
    for i in range(survival_curves.shape[0]):
        predict_probs = predict_multi_probs_from_curve(survival_curves[i, :], time_coordinates, target_times).tolist()
        predict_probs_mat.append(predict_probs)
    predict_probs_mat = np.array(predict_probs_mat)

    return brier_multiple_points(predict_probs_mat, event_times, event_indicators, train_event_times,
                                 train_event_indicators, target_times)


def integrated_brier_score_sksurv(
        predicted_survival_curves: pd.DataFrame,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        train_event_times: NumericArrayLike,
        train_event_indicators: NumericArrayLike,
        num_points: int = None,
        draw_figure: bool = False
) -> float:
    """
    Calculate the integrated Brier score (IBS) for the predicted survival curves from the scikit-survival package.
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
    param num_points: integer, default: None
        The number of points to calculate the Integrated Brier Score.
        Default None, which will use the test set true event times as the time points.
    :return:
    """
    warnings.warn("This function is deprecated and might be deleted in the future. "
                  "Please use the class 'ScikitSurvivalEvaluator' from Evaluator.py.", DeprecationWarning)
    event_times, event_indicators = check_and_convert(event_times, event_indicators)
    train_event_times, train_event_indicators = check_and_convert(train_event_times, train_event_indicators)
    max_target_time = np.amax(np.concatenate((event_times, train_event_times)))

    # If number of target time is not indicated, then we use the censored times obtained from test set
    if num_points is None:
        # test_censor_status = 1 - event_indicators
        censored_times = event_times[event_indicators == 0]
        sorted_censored_times = np.sort(censored_times)
        time_points = sorted_censored_times
        time_range = np.amax(time_points) - np.amin(time_points)
    else:
        time_points = np.linspace(0, max_target_time, num_points)
        time_range = max_target_time

    # Get single brier score from multiple target times, and use trapezoidal integral to calculate ISB.
    #########################
    # solution 1, implemented using metrics multiplication, this is geometrically faster than solution 2
    b_scores = brier_multiple_points_sksurv(predicted_survival_curves, event_times, event_indicators,
                                            train_event_times, train_event_indicators, time_points)
    if np.isnan(b_scores).any():
        print("Time-dependent Brier Score contains nan")
        print(b_scores)
    integral_value = trapezoid(b_scores, time_points)
    ibs_score = integral_value / time_range

    if draw_figure:
        plt.plot(time_points, b_scores, 'bo-')
        plt.xlabel('Time')
        plt.ylabel('Brier Score')
        plt.show()
    ##########################
    # solution 2, implemented by iteratively call single_brier_score_pycox()
    # b_scores = []
    # for i in range(len(time_points)):
    #     b_score = single_brier_score_pycox(predicted_survival_curves, event_times, event_indicators,
    #                                        train_event_times, train_event_indicators, time_points[i])
    #     b_scores.append(b_score)
    # b_scores = np.array(b_scores)
    # integral_value = trapezoid(b_scores, time_points)
    # ibs_score = integral_value / time_range

    return ibs_score


def brier_multiple_points_sksurv(
        predicted_survival_curves: pd.DataFrame,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        train_event_times: NumericArrayLike,
        train_event_indicators: NumericArrayLike,
        target_times: np.ndarray
) -> np.ndarray:
    """
    Calculate multiple Brier scores at multiple specific times.
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
    param target_times: float, default: None
        The specific time points for which to estimate the Brier scores.
    :return:
        Values of multiple Brier scores.
    """
    warnings.warn("This function is deprecated and might be deleted in the future. "
                  "Please use the class 'ScikitSurvivalEvaluator' from Evaluator.py.", DeprecationWarning)
    event_times, event_indicators = check_and_convert(event_times, event_indicators)
    train_event_times, train_event_indicators = check_and_convert(train_event_times, train_event_indicators)

    predict_probs_mat = []
    for i in range(predicted_survival_curves.shape[0]):
        predict_probs = predict_multi_probs_from_curve(predicted_survival_curves[i].x, predicted_survival_curves[i].y,
                                                       target_times).tolist()
        predict_probs_mat.append(predict_probs)
    predict_probs_mat = np.array(predict_probs_mat)

    inverse_train_event_indicators = 1 - train_event_indicators

    ipc_model = KaplanMeier(train_event_times, inverse_train_event_indicators)
    # sorted_test_event_times = np.argsort(event_times)

    if target_times.ndim != 1:
        error = "'time_grids' is not a one-dimensional array."
        raise TypeError(error)

    # bs_points_matrix = np.tile(event_times, (len(target_times), 1))
    target_times_mat = np.repeat(target_times.reshape(1, -1), repeats=len(event_times), axis=0)
    event_times_mat = np.repeat(event_times.reshape(-1, 1), repeats=len(target_times), axis=1)
    event_indicators_mat = np.repeat(event_indicators.reshape(-1, 1), repeats=len(target_times), axis=1)
    event_indicators_mat = event_indicators_mat.astype(bool)
    # Category one calculates IPCW weight at observed time point.
    # Category one is individuals with event time lower than the time of interest and were NOT censored.
    ipc_pred = ipc_model.predict(event_times_mat)
    # Catch if denominator is 0.
    ipc_pred[ipc_pred == 0] = np.inf
    weight_cat1 = ((event_times_mat <= target_times_mat) & event_indicators_mat) / ipc_pred
    # Catch if event times goes over max training event time, i.e. predict gives NA
    weight_cat1[np.isnan(weight_cat1)] = 0
    # Category 2 is individuals whose time was greater than the time of interest (singleBrierTime)
    # contain both censored and uncensored individuals.
    ipc_target_pred = ipc_model.predict(target_times_mat)
    # Catch if denominator is 0.
    ipc_target_pred[ipc_target_pred == 0] = np.inf
    weight_cat2 = (event_times_mat > target_times_mat) / ipc_target_pred
    # predict returns NA if the passed in time is greater than any of the times used to build
    # the inverse probability of censoring model.
    weight_cat2[np.isnan(weight_cat2)] = 0

    ipcw_square_error_mat = np.square(predict_probs_mat) * weight_cat1 + np.square(1 - predict_probs_mat) * weight_cat2
    brier_scores = np.mean(ipcw_square_error_mat, axis=0)
    return brier_scores
