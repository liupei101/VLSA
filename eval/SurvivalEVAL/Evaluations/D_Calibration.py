import numpy as np
import pandas as pd
from scipy.stats import chisquare
import warnings

from .custom_types import NumericArrayLike
from .util import check_and_convert, predict_prob_from_curve


def d_calibration_pycox(
        predicted_survival_curves: pd.DataFrame,
        event_time: NumericArrayLike,
        event_indicator: NumericArrayLike,
        num_bins: int = 10
) -> (float, np.ndarray):
    """
    Calculate the D-Calibration score for PyCox models.
    Parameters
    ----------
    predicted_survival_curves: pd.DataFrame
        The predicted survival curves from the PyCox model.
    event_time: NumericArrayLike
        The true event times.
    event_indicator: NumericArrayLike
        The true event indicators.
    num_bins: int
        The number of bins to use for the D-Calibration score.

    Returns
    -------
    float
        The D-Calibration score.
    np.ndarray
        The binning of the D-Calibration score.
    """
    warnings.warn("This function is deprecated and might be deleted in the future. "
                  "Please use the class 'PyCoxEvaluator' from Evaluator.py.", DeprecationWarning)
    # Checking the format of the data
    true_event_times, uncensor_status = check_and_convert(event_time, event_indicator)
    # Extracting the time buckets
    time_coordinates = predicted_survival_curves.index.values
    # computing the Survival function, and set the small negative value to zero
    survival_curves = predicted_survival_curves.values.T
    survival_curves[survival_curves < 0] = 0

    predict_probs = []
    for i in range(survival_curves.shape[0]):
        predict_prob = predict_prob_from_curve(survival_curves[i, :], time_coordinates, true_event_times[i])
        predict_probs.append(predict_prob)
    predict_probs = np.array(predict_probs)
    return d_calibration(predict_probs, uncensor_status, num_bins)


def d_calibration(
        predict_probs,
        event_indicators,
        num_bins: int = 10
) -> (float, np.ndarray):
    """
    Calculate the D-Calibration score.
    Parameters
    ----------
    predict_probs: np.ndarray
        The predicted survival probabilities at individual's event/censor time.
    event_indicators: np.ndarray
        The event indicators.
    num_bins: int
        The number of bins to use for the D-Calibration score.

    Returns
    -------
    pvalue: float
        The p-value of the D-Calibration test.
    combine_binning: np.ndarray
        The binning histogram of the D-Calibration test.
    """
    quantile = np.linspace(1, 0, num_bins + 1)
    censor_indicators = 1 - event_indicators

    event_probabilities = predict_probs[event_indicators.astype(bool)]
    event_position = np.digitize(event_probabilities, quantile)
    event_position[event_position == 0] = 1     # class probability==1 to the first bin

    event_binning = np.zeros([num_bins])
    for i in range(len(event_position)):
        event_binning[event_position[i] - 1] += 1

    censored_probabilities = predict_probs[censor_indicators.astype(bool)]

    censor_binning = np.zeros([num_bins])
    if len(censored_probabilities) > 0:
        for i in range(len(censored_probabilities)):
            partial_binning = create_censor_binning(censored_probabilities[i], num_bins)
            censor_binning += partial_binning

    combine_binning = event_binning + censor_binning
    _, pvalue = chisquare(combine_binning)
    return pvalue, combine_binning


def d_calibration_sksurv(
        predicted_survival_curves: pd.DataFrame,
        event_time: NumericArrayLike,
        event_indicator: NumericArrayLike,
        num_bins: int = 10
) -> (float, np.ndarray):
    """
    Calculate the D-Calibration score for scikit-survival models.
    :param predicted_survival_curves: pd.DataFrame
        The predicted survival curves from the scikit-survival model. The index is the time coordinates.
    :param event_time: NumericArrayLike
        The true event times.
    :param event_indicator: NumericArrayLike
        The true event indicators.
    :param num_bins: int
        The number of bins to use for the D-Calibration score.
    :return:
    pvalue: float
        The p-value of the D-Calibration test.
    combine_binning: np.ndarray
        The binning histogram of the D-Calibration test.
    """
    warnings.warn("This function is deprecated and might be deleted in the future. "
                  "Please use the class 'ScikitSurvivalEvaluator' from Evaluator.py.", DeprecationWarning)
    # Checking the format of the data
    true_event_times, uncensor_status = check_and_convert(event_time, event_indicator)

    predict_probs = []
    for i in range(predicted_survival_curves.shape[0]):
        predict_prob = predict_prob_from_curve(predicted_survival_curves[i].y, predicted_survival_curves[i].x,
                                               true_event_times[i])
        predict_probs.append(predict_prob)
    predict_probs = np.array(predict_probs)

    return d_calibration(predict_probs, uncensor_status, num_bins)


def d_calibration_pysurvival(
        model,
        X,
        T,
        E,
        num_bins=10
) -> (np.ndarray, float):
    warnings.warn("This function is deprecated and might be deleted in the future. "
                  "Please use the class 'PyCoxEvaluator' from Evaluator.py.", DeprecationWarning)
    # Checking the format of the data
    true_event_times, uncensor_status = check_and_convert(T, E)
    # computing the Survival function
    survival_curves = model.predict_survival(X, None)

    # Extracting the time buckets
    time_coordinates = model.times

    # if isinstance(model, RandomSurvivalForestModel):
    #     time_coordinates.pop()
    #     survival_curves = np.delete(survival_curves, -1, 1)
    #     print('use rsf, delete last time point')

    predict_probs = []
    for i in range(survival_curves.shape[0]):
        predict_prob = predict_prob_from_curve(survival_curves[i, :], time_coordinates, true_event_times[i])
        predict_probs.append(predict_prob)
    predict_probs = np.array(predict_probs)
    return d_calibration(predict_probs, uncensor_status, num_bins)


def create_censor_binning(
        probability: float,
        num_bins: int
) -> np.ndarray:
    """
    For censoring instance,
    b1 will be the infimum probability of the bin that contains S(c),
    for the bin of [b1, b2) which contains S(c), probability = (S(c) - b1) / S(c)
    for the rest of the bins, [b2, b3), [b3, b4), etc., probability = 1 / (B * S(c)), where B is the number of bins.
    :param probability: float
        The predicted probability at the censored time of a censoring instance.
    :param num_bins: int
        The number of bins to use for the D-Calibration score.
    :return:
    final_binning: np.ndarray
        The "split" histogram of this censored subject.
    """
    quantile = np.linspace(1, 0, num_bins + 1)
    censor_binning = np.zeros(num_bins)
    for i in range(num_bins):
        if probability == 1:
            censor_binning += 0.1
            break
        elif quantile[i] > probability >= quantile[i + 1]:
            first_bin = (probability - quantile[i + 1]) / probability if probability != 0 else 1
            rest_bins = 1 / (num_bins * probability) if probability != 0 else 0
            censor_binning[i] += first_bin
            censor_binning[i + 1:] += rest_bins
            break
    # assert len(censor_binning) == num_bins, f"censor binning should have size of {num_bins}"
    return censor_binning
