import numpy as np
import pandas as pd
from scipy.stats import chi2
import warnings

from .custom_types import Numeric, NumericArrayLike
from .util import check_and_convert, KaplanMeier, predict_prob_from_curve


def one_calibration_pycox(
        predicted_survival_curves: pd.DataFrame,
        event_time: NumericArrayLike,
        event_indicator: NumericArrayLike,
        target_time: Numeric,
        num_bins: int = 10,
        method: str = "DN"
) -> (float, list, list):
    warnings.warn("This function is deprecated and might be deleted in the future. "
                  "Please use the class 'PyCoxEvaluator' from Evaluator.py.", DeprecationWarning)
    # Checking the format of the data
    true_event_times, uncensor_status = check_and_convert(event_time, event_indicator)
    # Extracting the time buckets
    time_coordinates = predicted_survival_curves.index.values
    # computing the Survival function, and set the small negative value to zero
    survival_curves = predicted_survival_curves.values.T
    survival_curves[survival_curves < 0] = 0
    predictions = []
    for i in range(survival_curves.shape[0]):
        predict_prob = predict_prob_from_curve(survival_curves[i, :], time_coordinates, target_time)
        predictions.append(predict_prob)
    predictions = np.array(predictions)

    return one_calibration(predictions, true_event_times, uncensor_status, target_time, num_bins, method)


def one_calibration_sksurv(
        predicted_survival_curves: pd.DataFrame,
        event_time: NumericArrayLike,
        event_indicator: NumericArrayLike,
        target_time: Numeric,
        num_bins: int = 10,
        method: str = "DN"
) -> (float, list, list):
    warnings.warn("This function is deprecated and might be deleted in the future. "
                  "Please use the class 'ScikitSurvivalEvaluator' from Evaluator.py.", DeprecationWarning)
    # Checking the format of the data
    true_event_times, uncensor_status = check_and_convert(event_time, event_indicator)
    predictions = []
    for i in range(predicted_survival_curves.shape[0]):
        predict_prob = predict_prob_from_curve(predicted_survival_curves[i].y, predicted_survival_curves[i].x,
                                               target_time)
        predictions.append(predict_prob)
    predictions = np.array(predictions)

    return one_calibration(predictions, true_event_times, uncensor_status, target_time, num_bins, method)


def one_calibration(
        predictions: np.ndarray,
        event_time: np.ndarray,
        event_indicator: np.ndarray,
        target_time: Numeric,
        num_bins: int = 10,
        method: str = "DN"
) -> (float, list, list):
    """
    Compute the one calibration score for a given set of predictions and true event times.
    Parameters
    ----------
    predictions: np.ndarray
        The predicted probabilities at the time of interest.
    event_time: np.ndarray
        The true event times.
    event_indicator: np.ndarray
        The indicator of whether the event is observed or not.
    target_time: Numeric
        The time of interest.
    num_bins: int
        The number of bins to divide the predictions into.
    method: str
        The method to handle censored patients. The options are: "DN" (default), and "Uncensored".

    Returns
    -------
    score: float
        The one calibration score.
    observed_probabilities: list
        The observed probabilities in each bin.
    expected_probabilities: list
        The expected probabilities in each bin.
    """
    predictions = 1 - predictions
    sorted_idx = np.argsort(-predictions)
    sorted_predictions = predictions[sorted_idx]
    sorted_event_time = event_time[sorted_idx]
    sorted_event_indicator = event_indicator[sorted_idx]

    binned_event_time = np.array_split(sorted_event_time, num_bins)
    binned_event_indicator = np.array_split(sorted_event_indicator, num_bins)
    binned_predictions = np.array_split(sorted_predictions, num_bins)

    hl_statistics = 0
    observed_probabilities = []
    expected_probabilities = []
    for b in range(num_bins):
        # mean_prob = np.mean(binned_predictions[b])
        bin_size = len(binned_event_time[b])

        # For Uncensored method, we simply remove the censored patients,
        # for D'Agostina-Nam method, we will use 1-KM(t) as the observed probability.
        if method == "Uncensored":
            filter_idx = ~((binned_event_time[b] < target_time) & (binned_event_indicator[b] == 0))
            mean_prob = np.mean(binned_predictions[b][filter_idx])
            event_count = sum(binned_event_time[b][filter_idx] < target_time)
            event_probability = event_count / bin_size
            hl_statistics += (event_count - bin_size * mean_prob) ** 2 / (
                    bin_size * mean_prob * (1 - mean_prob))
        elif method == "DN":
            mean_prob = np.mean(binned_predictions[b])
            km_model = KaplanMeier(binned_event_time[b], binned_event_indicator[b])
            event_probability = 1 - km_model.predict(target_time)
            hl_statistics += (bin_size * event_probability - bin_size * mean_prob) ** 2 / (bin_size * mean_prob * (1 - mean_prob))
        else:
            error = "Please enter one of 'Uncensored','DN' for method."
            raise TypeError(error)
        observed_probabilities.append(event_probability)
        expected_probabilities.append(mean_prob)

    degree_of_freedom = num_bins - 1 if (num_bins <= 15 and method == "DN") else num_bins - 2
    p_value = 1 - chi2.cdf(hl_statistics, degree_of_freedom)

    return p_value, observed_probabilities, expected_probabilities
