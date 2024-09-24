import numpy as np
import pandas as pd
from typing import Optional
import warnings

from .custom_types import NumericArrayLike
from .util import (check_and_convert, KaplanMeierArea,
                   predict_mean_survival_time, predict_median_survival_time)


def concordance_pycox(
        predicted_survival_curves: pd.DataFrame,
        event_time: NumericArrayLike,
        event_indicator: NumericArrayLike,
        ties: str = "None",
        predicted_time_method: str = "Median"
) -> (float, float, int):
    warnings.warn("This function is deprecated and might be deleted in the future. "
                  "Please use the class 'PyCoxEvaluator' from Evaluator.py.", DeprecationWarning)
    event_time, event_indicator = check_and_convert(event_time, event_indicator)
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

    return concordance(predicted_times, event_time, event_indicator, ties=ties)


def concordance_sksurv(
        predicted_survival_curves: np.ndarray,
        event_time: NumericArrayLike,
        event_indicator: NumericArrayLike,
        ties: str = "None",
        predicted_time_method: str = "Median"
) -> (float, float, int):
    warnings.warn("This function is deprecated and might be deleted in the future. "
                  "Please use the class 'ScikitSurvivalEvaluator' from Evaluator.py.", DeprecationWarning)
    event_time, event_indicator = check_and_convert(event_time, event_indicator)

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

    return concordance(predicted_times, event_time, event_indicator, ties=ties)


def concordance(
        predicted_times: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        train_event_times: Optional[np.ndarray] = None,
        train_event_indicators: Optional[np.ndarray] = None,
        pair_method: str = "Comparable",
        ties: str = "Risk"
) -> (float, float, int):
    """
    Calculate the concordance index between the predicted survival times and the true survival times.
    param predicted_times: array-like, shape = (n_samples,)
        The predicted survival times.
    param event_times: array-like, shape = (n_samples,)
        The true survival times.
    param event_indicators: array-like, shape = (n_samples,)
        The event indicators of the true survival times.
    param train_event_times: array-like, shape = (n_train_samples,)
        The true survival times of the training set.
    param train_event_indicators: array-like, shape = (n_train_samples,)
        The event indicators of the true survival times of the training set.
    param pair_method: str, optional (default="Comparable")
        A string indicating the method for constructing the pairs of samples.
        "Comparable": the pairs are constructed by comparing the predicted survival time of each sample with the
        event time of all other samples. The pairs are only constructed between samples with comparable
        event times. For example, if sample i has a censor time of 10, then the pairs are constructed by
        comparing the predicted survival time of sample i with the event time of all samples with event
        time of 10 or less.
        "Margin": the pairs are constructed between all samples. A best-guess time for the censored samples
        will be calculated and used to construct the pairs.
    param ties: str, optional (default="Risk")
        A string indicating the way ties should be handled.
        Options: "None" (default), "Time", "Risk", or "All"
        "None" will throw out all ties in true survival time and all ties in predict survival times (risk scores).
        "Time" includes ties in true survival time but removes ties in predict survival times (risk scores).
        "Risk" includes ties in predict survival times (risk scores) but not in true survival time.
        "All" includes all ties.
        Note the concordance calculation is given by
        (Concordant Pairs + (Number of Ties/2))/(Concordant Pairs + Discordant Pairs + Number of Ties).
    :return: (float, float, int)
        The concordance index, the number of concordant pairs, and the number of total pairs.
    """
    # the scikit-survival concordance function only takes risk scores to calculate.
    # So at first we should transfer the predicted time -> risk score.
    # The risk score should be higher for subjects that live shorter (i.e. lower average survival time).

    event_indicators = event_indicators.astype(bool)

    if pair_method == "Comparable":
        risks = -1 * predicted_times
        partial_weights = None
        bg_event_times = None
    elif pair_method == "Margin":
        if train_event_times is None or train_event_indicators is None:
            error = "If 'Margin' is chosen, training set information must be provided."
            raise ValueError(error)

        train_event_indicators = train_event_indicators.astype(bool)

        km_model = KaplanMeierArea(train_event_times, train_event_indicators)
        km_linear_zero = -1 / ((1 - min(km_model.survival_probabilities))/(0 - max(km_model.survival_times)))
        if np.isinf(km_linear_zero):
            km_linear_zero = max(km_model.survival_times)
        predicted_times = np.clip(predicted_times, a_max=km_linear_zero, a_min=None)
        risks = -1 * predicted_times

        censor_times = event_times[~event_indicators]
        partial_weights = np.ones_like(event_indicators, dtype=float)
        partial_weights[~event_indicators] = 1 - km_model.predict(censor_times)

        best_guesses = km_model.best_guess(censor_times)
        best_guesses[censor_times > km_linear_zero] = censor_times[censor_times > km_linear_zero]

        bg_event_times = np.copy(event_times)
        bg_event_times[~event_indicators] = best_guesses
    else:
        raise TypeError("Method for calculating concordance is unrecognized.")
    # risk_ties means predicted times are the same while true times are different.
    # time_ties means true times are the same while predicted times are different.
    # cindex, concordant_pairs, discordant_pairs, risk_ties, time_ties = metrics.concordance_index_censored(
    #     event_indicators, event_times, estimate=risk)
    cindex, concordant_pairs, discordant_pairs, risk_ties, time_ties = _estimate_concordance_index(
        event_indicators, event_times, estimate=risks, bg_event_time=bg_event_times, partial_weights=partial_weights)
    if ties == "None":
        total_pairs = concordant_pairs + discordant_pairs
        cindex = concordant_pairs / total_pairs
    elif ties == "Time":
        total_pairs = concordant_pairs + discordant_pairs + time_ties
        concordant_pairs = concordant_pairs + 0.5 * time_ties
        cindex = concordant_pairs / total_pairs
    elif ties == "Risk":
        # This should be the same as original outputted cindex from above
        total_pairs = concordant_pairs + discordant_pairs + risk_ties
        concordant_pairs = concordant_pairs + 0.5 * risk_ties
        cindex = concordant_pairs / total_pairs
    elif ties == "All":
        total_pairs = concordant_pairs + discordant_pairs + risk_ties + time_ties
        concordant_pairs = concordant_pairs + 0.5 * (risk_ties + time_ties)
        cindex = concordant_pairs / total_pairs
    else:
        error = "Please enter one of 'None', 'Time', 'Risk', or 'All' for handling ties for concordance."
        raise TypeError(error)

    return cindex, concordant_pairs, total_pairs


def _estimate_concordance_index(
        event_indicator: np.ndarray,
        event_time: np.ndarray,
        estimate: np.ndarray,
        bg_event_time: np.ndarray = None,
        partial_weights: np.ndarray = None,
        tied_tol: float = 1e-8
):
    order = np.argsort(event_time, kind="stable")

    comparable, tied_time, weight = _get_comparable(event_indicator, event_time, order)

    if partial_weights is not None:
        event_indicator = np.ones_like(event_indicator)
        comparable_2, tied_time, weight = _get_comparable(event_indicator, bg_event_time, order, partial_weights)
        for ind, mask in comparable.items():
            weight[ind][mask] = 1
        comparable = comparable_2

    if len(comparable) == 0:
        raise ValueError("Data has no comparable pairs, cannot estimate concordance index.")

    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0.0
    denominator = 0.0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]
        # w_i = partial_weights[order[ind]] # change this
        w_i = weight[ind]
        weight_i = w_i[order[mask]]

        est = estimate[order[mask]]

        assert event_i, 'got censored sample at index %d, but expected uncensored' % order[ind]

        ties = np.absolute(est - est_i) <= tied_tol
        # n_ties = ties.sum()
        n_ties = np.dot(weight_i, ties.T)
        # an event should have a higher score
        con = est < est_i
        # n_con = con[~ties].sum()
        con[ties] = False
        n_con = np.dot(weight_i, con.T)

        # numerator += w_i * n_con + 0.5 * w_i * n_ties
        # denominator += w_i * mask.sum()
        numerator += n_con + 0.5 * n_ties
        denominator += np.dot(w_i, mask.T)

        tied_risk += n_ties
        concordant += n_con
        # discordant += est.size - n_con - n_ties
        discordant += np.dot(w_i, mask.T) - n_con - n_ties

    cindex = numerator / denominator
    return cindex, concordant, discordant, tied_risk, tied_time


def _get_comparable(event_indicator: np.ndarray, event_time: np.ndarray, order: np.ndarray,
                    partial_weights: np.ndarray = None):
    if partial_weights is None:
        partial_weights = np.ones_like(event_indicator, dtype=float)
    n_samples = len(event_time)
    tied_time = 0
    comparable = {}
    weight = {}

    i = 0
    while i < n_samples - 1:
        time_i = event_time[order[i]]
        end = i + 1
        while end < n_samples and event_time[order[end]] == time_i:
            end += 1

        # check for tied event times
        event_at_same_time = event_indicator[order[i:end]]
        censored_at_same_time = ~event_at_same_time

        for j in range(i, end):
            if event_indicator[order[j]]:
                mask = np.zeros(n_samples, dtype=bool)
                mask[end:] = True
                # an event is comparable to censored samples at same time point
                mask[i:end] = censored_at_same_time
                comparable[j] = mask
                tied_time += censored_at_same_time.sum()
                weight[j] = partial_weights[order] * partial_weights[order[j]]
        i = end

    return comparable, tied_time, weight
