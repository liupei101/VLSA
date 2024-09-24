import numpy as np
import pandas as pd
import warnings
from typing import Union, Optional, Callable
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from abc import ABC
from functools import cached_property

from .Evaluations.custom_types import NumericArrayLike
from .Evaluations.util import check_and_convert
from .Evaluations.util import predict_mean_survival_time, predict_median_survival_time
from .Evaluations.util import predict_prob_from_curve, predict_multi_probs_from_curve, quantile_to_survival

from .Evaluations.Concordance import concordance
from .Evaluations.AreaUnderCurve import auc
from .Evaluations.BrierScore import single_brier_score, brier_multiple_points
from .Evaluations.MeanError import mean_error
from .Evaluations.OneCalibration import one_calibration
from .Evaluations.D_Calibration import d_calibration
from .Evaluations.KM_Calibration import km_calibration


class SurvivalEvaluator:
    def __init__(
            self,
            predicted_survival_curves: NumericArrayLike,
            time_coordinates: NumericArrayLike,
            test_event_times: NumericArrayLike,
            test_event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None,
            predict_time_method: str = "Median",
            interpolation: str = "Linear"
    ):
        """
        Initialize the Evaluator
        param predicted_survival_curves: structured array, shape = (n_samples, n_time_points)
            Predicted survival curves for the testing samples.
        param time_coordinates: structured array, shape = (n_time_points, )
            Time coordinates for the given curves.
        param test_event_times: structured array, shape = (n_samples, )
            Actual event/censor time for the testing samples.
        param test_event_indicators: structured array, shape = (n_samples, )
            Binary indicators of censoring for the testing samples
        param train_event_times: structured array, shape = (n_train_samples, )
            Actual event/censor time for the training samples.
        param train_event_indicators: structured array, shape = (n_train_samples, )
            Binary indicators of censoring for the training samples
        param predict_time_method: str, default = "Median"
            Method for calculating predicted survival time. Available options are "Median" and "Mean".
        param interpolation: str, default = "Linear"
            Method for interpolation. Available options are ['Linear', 'Pchip'].
        """
        self._predicted_curves = check_and_convert(predicted_survival_curves)
        self._time_coordinates = check_and_convert(time_coordinates)

        test_event_times, test_event_indicators = check_and_convert(test_event_times, test_event_indicators)
        self.event_times = test_event_times
        self.event_indicators = test_event_indicators

        if (train_event_times is not None) and (train_event_indicators is not None):
            train_event_times, train_event_indicators = check_and_convert(train_event_times, train_event_indicators)
        self.train_event_times = train_event_times
        self.train_event_indicators = train_event_indicators

        if predict_time_method == "Median":
            self.predict_time_method = predict_median_survival_time
        elif predict_time_method == "Mean":
            self.predict_time_method = predict_mean_survival_time
        else:
            error = "Please enter one of 'Median' or 'Mean' for calculating predicted survival time."
            raise TypeError(error)

        self.interpolation = interpolation

    def _error_trainset(self, method_name: str):
        if (self.train_event_times is None) or (self.train_event_indicators is None):
            raise TypeError("Train set information is missing. "
                            "Evaluator cannot perform {} evaluation.".format(method_name))

    @property
    def predicted_curves(self):
        return self._predicted_curves

    @predicted_curves.setter
    def predicted_curves(self, val: NumericArrayLike):
        print("Setter called. Resetting predicted curves for this evaluator.")
        self._predicted_curves = check_and_convert(val)
        self._clear_cache()

    @property
    def actual_survival_time(self):
        return self.event_times

    @actual_survival_time.setter
    def actual_survival_time(self, val: NumericArrayLike):
        print("Setter called. Resetting actual survival time for this evaluator.")
        self.event_times = check_and_convert(val)

    @property
    def actual_survival_event(self):
        return self.event_indicators

    @actual_survival_event.setter
    def actual_survival_event(self, val: NumericArrayLike):
        print("Setter called. Resetting actual survival event indicator for this evaluator.")
        self.event_indicators = check_and_convert(val)

    @property
    def time_coordinates(self):
        return self._time_coordinates

    @time_coordinates.setter
    def time_coordinates(self, val: NumericArrayLike):
        print("Setter called. Resetting time coordinates for this evaluator.")
        self._time_coordinates = check_and_convert(val)
        self._clear_cache()

    @cached_property
    def predicted_event_times(self):
        return self.predict_time_from_curve(self.predict_time_method)

    def _clear_cache(self):
        # See how to clear cache in functools:
        # https://docs.python.org/3/library/functools.html#functools.cached_property
        # https://stackoverflow.com/questions/62662564/how-do-i-clear-the-cache-from-cached-property-decorator
        self.__dict__.pop('predicted_event_times', None)

    def predict_time_from_curve(
            self,
            predict_method: Callable,
    ) -> np.ndarray:
        """
        Predict survival time from survival curves.
        param predict_method: Callable
            A function that takes in a survival curve and returns a predicted survival time.
            There are two build-in methods: 'predict_median_survival_time' and 'predict_mean_survival_time'.
            'predict_median_survival_time' uses the median of the survival curve as the predicted survival time.
            'predict_mean_survival_time' uses the expected time of the survival curve as the predicted survival time.
        :return: np.ndarray
            Predicted survival time for each sample.
        """
        if (predict_method is not predict_mean_survival_time) and (predict_method is not predict_median_survival_time):
            error = "Prediction method must be 'predict_mean_survival_time' or 'predict_median_survival_time', " \
                    "got '{}' instead".format(predict_method.__name__)
            raise TypeError(error)

        predicted_times = []
        for i in range(self.predicted_curves.shape[0]):
            predicted_time = predict_method(self.predicted_curves[i, :], self.time_coordinates, self.interpolation)
            predicted_times.append(predicted_time)
        predicted_times = np.array(predicted_times)
        return predicted_times

    def predict_probability_from_curve(
            self,
            target_time: Union[float, int, np.ndarray],
    ) -> np.ndarray:
        """
        Predict a probability of event at a given time point from a predicted curve. Each predicted curve will only
        have one corresponding probability. Note that this method is different from the
        'predict_multi_probabilities_from_curve' method, which predicts the multiple probabilities at multiple time
        points from a predicted curve.
        param target_time: float, int, or array-like, shape = (n_samples, )
            Time point(s) at which the probability of event is to be predicted. If float or int, the same time point is
            used for all samples. If array-like, each sample will have it own target time. The length of the array must
            be the same as the number of samples.
        :return: array-like, shape = (n_samples, )
            Predicted probabilities of event at the target time point(s).
        """
        if isinstance(target_time, (float, int)):
            target_time = target_time * np.ones_like(self.event_times)
        elif isinstance(target_time, np.ndarray):
            assert target_time.ndim == 1, "Target time must be a 1D array"
            assert target_time.shape[0] == self.predicted_curves.shape[0], "Target time must have the same length as " \
                                                                           "the number of samples"
        else:
            error = "Target time must be a float, int, or 1D array, got '{}' instead".format(type(target_time))
            raise TypeError(error)

        predict_probs = []
        for i in range(self.predicted_curves.shape[0]):
            predict_prob = predict_prob_from_curve(self.predicted_curves[i, :], self.time_coordinates,
                                                   target_time[i], self.interpolation)
            predict_probs.append(predict_prob)
        predict_probs = np.array(predict_probs)
        return predict_probs

    def predict_multi_probabilities_from_curve(
            self,
            target_times: np.ndarray
    ) -> np.ndarray:
        """
        Predict the probability of event at multiple time points from the predicted curve.
        param target_times: array-like, shape = (n_target_times)
            Time points at which the probability of event is to be predicted.
        :return: array-like, shape = (n_samples, n_target_times)
            Predicted probabilities of event at the target time points.
        """
        predict_probs_mat = []
        for i in range(self.predicted_curves.shape[0]):
            predict_probs = predict_multi_probs_from_curve(self.predicted_curves[i, :], self.time_coordinates,
                                                           target_times, self.interpolation).tolist()
            predict_probs_mat.append(predict_probs)
        predict_probs_mat = np.array(predict_probs_mat)
        return predict_probs_mat

    def plot_survival_curves(
            self,
            curve_indices,
            color=None,
            x_lim: tuple = None,
            y_lim: tuple = None,
            x_label: str = 'Time',
            y_label: str = 'Survival probability'
    ):
        """Plot survival curves."""
        fig, ax = plt.subplots()
        ax.plot(self.time_coordinates, self.predicted_curves[curve_indices, :].T, color=color, label=curve_indices)
        if y_lim is None:
            ax.set_ylim(0, 1.02)
        else:
            ax.set_ylim(y_lim)

        if x_lim is not None:
            ax.set_xlim(x_lim)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
        return fig, ax

    def concordance(
            self,
            ties: str = "None",
            pair_method: str = "Comparable"
    ) -> (float, float, int):
        """
        Calculate the concordance index between the predicted survival times and the true survival times.
        param ties: str, default = "None"
            A string indicating the way ties should be handled.
            Options: "None" (default), "Time", "Risk", or "All"
            "None" will throw out all ties in true survival time and all ties in predict survival times (risk scores).
            "Time" includes ties in true survival time but removes ties in predict survival times (risk scores).
            "Risk" includes ties in predict survival times (risk scores) but not in true survival time.
            "All" includes all ties.
            Note the concordance calculation is given by
            (Concordant Pairs + (Number of Ties/2))/(Concordant Pairs + Discordant Pairs + Number of Ties).
        param pair_method: str, default = "Comparable"
            A string indicating the method for constructing the pairs of samples.
            Options: "Comparable" (default) or "Margin"
            "Comparable": the pairs are constructed by comparing the predicted survival time of each sample with the
            event time of all other samples. The pairs are only constructed between samples with comparable
            event times. For example, if sample i has a censor time of 10, then the pairs are constructed by
            comparing the predicted survival time of sample i with the event time of all samples with event
            time of 10 or less.
            "Margin": the pairs are constructed between all samples. A best-guess time for the censored samples
            will be calculated and used to construct the pairs.
        :return: (float, float, int)
            The concordance index, the number of concordant pairs, and the number of total pairs.
        """
        # Choose prediction method based on the input argument
        if pair_method == "Margin" and (self.train_event_times is None or self.train_event_indicators is None):
            self._error_trainset("margin concordance")

        return concordance(self.predicted_event_times, self.event_times, self.event_indicators, self.train_event_times,
                           self.train_event_indicators, pair_method, ties)

    def auc(
            self,
            target_time: Optional[Union[int, float]] = None
    ) -> float:
        """
        Calculate the area under the ROC curve (AUC) score at a given time point from the predicted survival curve.
        param target_time: float, int, or None, default = None
            Time point at which the AUC score is to be calculated. If None, the AUC score is calculated at the
            median time of all the event/censor times from the training and test sets.
        :return: float
            The Brier score at the target time point.
        """
        event_times = np.concatenate((self.event_times, self.train_event_times)) \
            if self.train_event_times is not None else self.event_times

        if target_time is None:
            target_time = np.quantile(event_times, 0.5)

        predict_probs = self.predict_probability_from_curve(target_time)

        return auc(predict_probs, self.event_times, self.event_indicators, target_time)

    def brier_score(
            self,
            target_time: Optional[Union[int, float]] = None,
            IPCW_weighted: bool = True
    ) -> float:
        """
        Calculate the Brier score at a given time point from the predicted survival curve.
        param target_time: float, int, or None, default = None
            Time point at which the Brier score is to be calculated. If None, the Brier score is calculated at the
            median time of all the event/censor times from the training and test sets.
        param IPCW_weighted: bool, default = True
            Whether to use IPCW weighting for the Brier score.
        :return: float
            The Brier score at the target time point.
        """
        self._error_trainset("Brier score (BS)")

        if target_time is None:
            target_time = np.quantile(np.concatenate((self.event_times, self.train_event_times)), 0.5)

        predict_probs = self.predict_probability_from_curve(target_time)

        return single_brier_score(predict_probs, self.event_times, self.event_indicators, self.train_event_times,
                                  self.train_event_indicators, target_time, IPCW_weighted)

    def brier_score_multiple_points(
            self,
            target_times: np.ndarray,
            IPCW_weighted: bool = True
    ) -> np.ndarray:
        """
        Calculate multiple Brier scores at multiple specific times.
        param target_times: float, default: None
            The specific time points for which to estimate the Brier scores.
        param IPCW_weighted: bool, default = True
            Whether to use IPCW weighting for the Brier score.
        :return:
            Values of multiple Brier scores.
        """
        self._error_trainset("Brier score (BS)")

        predict_probs_mat = self.predict_multi_probabilities_from_curve(target_times)

        return brier_multiple_points(predict_probs_mat, self.event_times, self.event_indicators, self.train_event_times,
                                     self.train_event_indicators, target_times, IPCW_weighted)

    def integrated_brier_score(
            self,
            num_points: int = None,
            IPCW_weighted: bool = True,
            draw_figure: bool = False
    ) -> float:
        """
        Calculate the integrated Brier score (IBS) from the predicted survival curve.
        param num_points: int, default = None
            Number of points at which the Brier score is to be calculated. If None, the number of points is set to
            the number of event/censor times from the training and test sets.
        param IPCW_weighted: bool, default = True
            Whether to use IPCW weighting for the Brier score.
        param draw_figure: bool, default = False
            Whether to draw the figure of the IBS.
        :return: float
            The integrated Brier score.
        """
        self._error_trainset("Integrated Brier Score (IBS)")

        max_target_time = np.max(np.concatenate((self.event_times, self.train_event_times)))

        # If number of target time is not indicated, then we use the censored times obtained from test set
        if num_points is None:
            censored_times = self.event_times[self.event_indicators == 0]
            time_points = np.unique(censored_times)
            if time_points.size == 0:
                raise ValueError("You don't have censor data in the testset, "
                                 "please provide \"num_points\" for calculating IBS")
            else:
                time_range = np.max(time_points) - np.min(time_points)
        else:
            time_points = np.linspace(0, max_target_time, num_points)
            time_range = max_target_time

        # Get single brier score from multiple target times, and use trapezoidal integral to calculate ISB.
        #########################
        # Solution 1, implemented using metrics multiplication, this is geometrically faster than solution 2
        b_scores = self.brier_score_multiple_points(time_points, IPCW_weighted)
        if np.isnan(b_scores).any():
            warnings.warn("Time-dependent Brier Score contains nan")
            bs_dict = {}
            for time_point, b_score in zip(time_points, b_scores):
                bs_dict[time_point] = b_score
            print("Brier scores for multiple time points are".format(bs_dict))
        integral_value = trapezoid(b_scores, time_points)
        ibs_score = integral_value / time_range
        ##########################
        # (Deprecated)
        # Solution 2, implemented by iteratively calling self.brier_score(),
        # this solution is much slower than solution 1
        # b_scores = []
        # for i in range(len(time_points)):
        #     b_score = self.brier_score(time_points[i])
        #     b_scores.append(b_score)
        # b_scores = np.array(b_scores)
        # integral_value = trapezoid(b_scores, time_points)
        # ibs_score = integral_value / time_range

        # Draw the Brier score graph
        if draw_figure:
            plt.plot(time_points, b_scores, 'bo-')
            score_text = r'IBS$= {:.3f}$'.format(ibs_score)
            plt.plot([], [], ' ', label=score_text)
            plt.legend()
            # plt.text(500, 0.05, r'IBS$= {:.3f}$'.format(ibs_score), verticalalignment='top',
            #          horizontalalignment='left', fontsize=12, color='Black')
            plt.xlabel('Time')
            plt.ylabel('Brier Score')
            plt.show()
        return ibs_score

    def mae(
            self,
            method: str = "Hinge",
            weighted: bool = True,
            log_scale: bool = False,
            reduction: bool = True,
            verbose: bool = False
    ) -> float:
        """
        Calculate the MAE score for the test set.
        param method: string, default: "Hinge"
            The method used to calculate the MAE score.
            Options: "Uncensored", "Hinge" (default), "Margin", "IPCW-v1", "IPCW-v2", or "Pseudo_obs"\
        param weighted: bool, default: True
            Whether to use weighting scheme for MAE.
        param log_scale: boolean, default: False
            Whether to use log scale for the time axis.
        param verbose: boolean, default: False
            Whether to show the progress bar.
        :return: float
            The MAE score for the test set.
        """
        return mean_error(
            predicted_times=self.predicted_event_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            error_type="absolute",
            method=method,
            weighted=weighted,
            log_scale=log_scale,
            reduction=reduction,
            verbose=verbose
        )

    def mse(
            self,
            method: str = "Hinge",
            weighted: bool = True,
            log_scale: bool = False,
            reduction: bool = True,
            verbose: bool = False
    ) -> float:
        """
        Calculate the MAE score for the test set.
        param method: string, default: "Hinge"
            The method used to calculate the MAE score.
            Options: "Uncensored", "Hinge" (default), "Margin", "IPCW-v1", "IPCW-v2", or "Pseudo_obs"\
        param weighted: bool, default: True
            Whether to use weighting scheme for MAE.
        param log_scale: boolean, default: False
            Whether to use log scale for the time axis.
        param verbose: boolean, default: False
            Whether to show the progress bar.
        :return: float
            The MAE score for the test set.
        """
        return mean_error(
            predicted_times=self.predicted_event_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            error_type="squared",
            method=method,
            weighted=weighted,
            log_scale=log_scale,
            reduction=reduction,
            verbose=verbose
        )

    def rmse(
            self,
            method: str = "Hinge",
            weighted: bool = True,
            log_scale: bool = False,
            verbose: bool = False
    ) -> float:
        """
        Calculate the root mean squared error (RMSE) score for the test set.
        param method: string, default: "Hinge"
            The method used to calculate the MAE score.
            Options: "Uncensored", "Hinge" (default), "Margin", "IPCW-v1", "IPCW-v2", or "Pseudo_obs"\
        param weighted: bool, default: True
            Whether to use weighting scheme for MAE.
        param log_scale: boolean, default: False
            Whether to use log scale for the time axis.
        param verbose: boolean, default: False
            Whether to show the progress bar.
        :return: float
            The MAE score for the test set.
        """
        return self.mse(method, weighted, log_scale, verbose) ** 0.5

    def one_calibration(
            self,
            target_time: Union[float, int],
            num_bins: int = 10,
            method: str = "DN"
    ) -> (float, list, list):
        """
        Calculate the one calibration score at a given time point from the predicted survival curve.
        param target_time: float, int
            Time point at which the one calibration score is to be calculated.
        param num_bins: int, default: 10
            Number of bins used to calculate the one calibration score.
        param method: string, default: "DN"
            The method used to calculate the one calibration score.
            Options: "Uncensored", or "DN" (default)
        :return: float, list, list
            (p-value, observed probabilities, expected probabilities)
        """
        predict_probs = self.predict_probability_from_curve(target_time)
        return one_calibration(predict_probs, self.event_times, self.event_indicators, target_time, num_bins, method)

    def d_calibration(
            self,
            num_bins: int = 10
    ) -> (float, np.ndarray):
        """
        Calculate the D calibration score from the predicted survival curve.
        param num_bins: int, default: 10
            Number of bins used to calculate the D calibration score.
        :return: float, np.ndarray
            (p-value, counts in bins)
        """
        predict_probs = self.predict_probability_from_curve(self.event_times)
        return d_calibration(predict_probs, self.event_indicators, num_bins)

    def x_calibration(
            self,
            num_bins: int = 10
    ) -> float:
        """
        Calculate the X calibration score from the predicted survival curve.
        Parameters
        ----------
        num_bins

        Returns
        -------

        """
        _, bin_hist = self.d_calibration(num_bins)
        n_bins = bin_hist.shape[0]
        # normalize the histogram
        d_cal_pdf = bin_hist / bin_hist.sum()
        # compute the x-calibration score
        optimal = np.ones_like(d_cal_pdf) / n_bins
        x_cal = np.sum(np.square(d_cal_pdf - optimal))
        return x_cal

    def km_calibration(self):
        """
        Calculate the KM calibration score from the predicted survival curve.
        :return: float
            KL divergence between the average predicted survival distribution and the Kaplan-Meier distribution.
        """
        average_survival_curve = np.mean(self._predicted_curves, axis=0)
        return km_calibration(average_survival_curve, self.time_coordinates, self.event_times, self.event_indicators)


class PycoxEvaluator(SurvivalEvaluator, ABC):
    def __init__(
            self,
            surv: pd.DataFrame,
            test_event_times: NumericArrayLike,
            test_event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None,
            predict_time_method: str = "Median",
            interpolation: str = "Linear"
    ):
        """
        Evaluator for survival models in PyCox packages.
        param surv: pd.DataFrame, shape = (n_time_points, n_samples)
            Predicted survival curves for the testing samples
            DataFrame index represents the time coordinates for the given curves.
            DataFrame value represents transpose of the survival probabilities.
        param test_event_times: NumericArrayLike, shape = (n_samples,)
            Event times for the testing samples.
        param test_event_indicators: NumericArrayLike, shape = (n_samples,)
            Event indicators for the testing samples.
        param train_event_times: NumericArrayLike, shape = (n_samples,), optional
            Event times for the training samples.
        param train_event_indicators: NumericArrayLike, shape = (n_samples,), optional
            Event indicators for the training samples.
        param predict_time_method: string, default: "Median"
            The method used to calculate the predicted event time. Options: "Median" (default), "Mean".
        param interpolation: string, default: "Linear"
            The interpolation method used to calculate the predicted event time.
            Options: "Linear" (default), "Pchip".
        """
        time_coordinates = surv.index.values
        predicted_survival_curves = surv.values.T
        # Pycox models can sometimes obtain -0 as survival probabilities. Need to convert that to 0.
        predicted_survival_curves[predicted_survival_curves < 0] = 0
        super(PycoxEvaluator, self).__init__(predicted_survival_curves, time_coordinates, test_event_times,
                                             test_event_indicators, train_event_times, train_event_indicators,
                                             predict_time_method, interpolation)


class LifelinesEvaluator(PycoxEvaluator, ABC):
    def __init__(
            self,
            surv: pd.DataFrame,
            test_event_times: NumericArrayLike,
            test_event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None,
            predict_time_method: str = "Median",
            interpolation: str = "Linear"
    ):
        """
        Evaluator for survival models in Lifelines packages.
        param surv: pd.DataFrame, shape = (n_time_points, n_samples)
            Predicted survival curves for the testing samples
        param test_event_times: NumericArrayLike, shape = (n_samples,)
            Event times for the testing samples.
        param test_event_indicators: NumericArrayLike, shape = (n_samples,)
            Event indicators for the testing samples.
        param train_event_times: NumericArrayLike, shape = (n_samples,), optional
            Event times for the training samples.
        param train_event_indicators: NumericArrayLike, shape = (n_samples,), optional
            Event indicators for the training samples.
        param predict_time_method: string, default: "Median"
            The method used to calculate the predicted event time. Options: "Median" (default), "Mean".
        param interpolation: string, default: "Linear"
            The interpolation method used to calculate the predicted event time.
            Options: "Linear" (default), "Pchip".
        """
        super(LifelinesEvaluator, self).__init__(surv, test_event_times, test_event_indicators, train_event_times,
                                                 train_event_indicators, predict_time_method, interpolation)


class ScikitSurvivalEvaluator(SurvivalEvaluator, ABC):
    def __init__(
            self,
            surv: np.ndarray,
            test_event_times: NumericArrayLike,
            test_event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None,
            predict_time_method: str = "Median",
            interpolation: str = "Linear"
    ):
        """
        Evaluator for survival models in scikit-survival packages.
        param surv: np.ndarray, shape = (n_samples,)
            Predicted survival curves for the testing samples. Each element is a scikit-survival customized object.
            '.x' attribute is the time coordinates for the given curve. '.y' attribute is the survival probabilities.
        param test_event_times: NumericArrayLike, shape = (n_samples,)
            Event times for the testing samples.
        param test_event_indicators: NumericArrayLike, shape = (n_samples,)
            Event indicators for the testing samples.
        param train_event_times: NumericArrayLike, shape = (n_samples,), optional
            Event times for the training samples.
        param train_event_indicators: NumericArrayLike, shape = (n_samples,), optional
            Event indicators for the training samples.
        param predict_time_method: string, default: "Median"
            The method used to calculate the predicted event time. Options: "Median" (default), "Mean".
        param interpolation: string, default: "Linear"
            The interpolation method used to calculate the predicted event time.
            Options: "Linear" (default), "Pchip".
        """
        time_coordinates = surv[0].x
        predict_curves = []
        for i in range(len(surv)):
            predict_curve = surv[i].y
            if False in (time_coordinates == surv[i].x):
                raise KeyError("{}-th survival curve does not have same time coordinates".format(i))
            predict_curves.append(predict_curve)
        predicted_curves = np.array(predict_curves)
        if time_coordinates[0] != 0:
            time_coordinates = np.concatenate([np.array([0]), time_coordinates], 0)
            predicted_curves = np.concatenate([np.ones([len(predicted_curves), 1]), predicted_curves], 1)
        # If some survival curves are all ones, we should do something.
        if np.any(predicted_curves[:, len(time_coordinates) - 1] == 1):
            idx_need_fix = predicted_curves[:, len(time_coordinates) - 1] == 1
            max_prob_at_end = np.max(predicted_curves[~idx_need_fix,
                                                      len(time_coordinates) - 1])
            # max_prob_at_end + (1 - max_prob_at_end) * 0.9
            predicted_curves[idx_need_fix, len(time_coordinates) - 1] = max(0.1 * max_prob_at_end + 0.9, 0.99)
        super(ScikitSurvivalEvaluator, self).__init__(predicted_curves, time_coordinates, test_event_times,
                                                      test_event_indicators, train_event_times, train_event_indicators,
                                                      predict_time_method, interpolation)


class PointEvaluator:
    def __init__(
            self,
            predicted_times: NumericArrayLike,
            test_event_times: NumericArrayLike,
            test_event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None,
    ):
        """
        Initialize the Evaluator
        param predicted_times: structured array, shape = (n_samples, )
            Predicted survival times for the testing samples.
        param test_event_times: structured array, shape = (n_samples, )
            Actual event/censor time for the testing samples.
        param test_event_indicators: structured array, shape = (n_samples, )
            Binary indicators of censoring for the testing samples
        param train_event_times: structured array, shape = (n_train_samples, )
            Actual event/censor time for the training samples.
        param train_event_indicators: structured array, shape = (n_train_samples, )
            Binary indicators of censoring for the training samples
        """
        self._predicted_times = check_and_convert(predicted_times)

        self.event_times, self.event_indicators = check_and_convert(test_event_times, test_event_indicators)

        if (train_event_times is not None) and (train_event_indicators is not None):
            train_event_times, train_event_indicators = check_and_convert(train_event_times, train_event_indicators)
        self.train_event_times = train_event_times
        self.train_event_indicators = train_event_indicators

    def _error_trainset(self, method_name: str):
        if (self.train_event_times is None) or (self.train_event_indicators is None):
            raise TypeError("Train set information is missing. "
                            "Evaluator cannot perform {} evaluation.".format(method_name))

    @property
    def predicted_times(self):
        return self._predicted_times

    @predicted_times.setter
    def predicted_times(self, predicted_times):
        print("Setter called. Resetting predicted_times.")
        self._predicted_times = predicted_times

    def concordance(
            self,
            ties: str = "None",
            pair_method: str = "Comparable"
    ) -> (float, float, int):
        """
        Calculate the concordance index between the predicted survival times and the true survival times.
        param ties: str, default = "None"
            A string indicating the way ties should be handled.
            Options: "None" (default), "Time", "Risk", or "All"
            "None" will throw out all ties in true survival time and all ties in predict survival times (risk scores).
            "Time" includes ties in true survival time but removes ties in predict survival times (risk scores).
            "Risk" includes ties in predict survival times (risk scores) but not in true survival time.
            "All" includes all ties.
            Note the concordance calculation is given by
            (Concordant Pairs + (Number of Ties/2))/(Concordant Pairs + Discordant Pairs + Number of Ties).
        param pair_method: str, default = "Comparable"
            A string indicating the method for constructing the pairs of samples.
            Options: "Comparable" (default) or "Margin"
            "Comparable": the pairs are constructed by comparing the predicted survival time of each sample with the
            event time of all other samples. The pairs are only constructed between samples with comparable
            event times. For example, if sample i has a censor time of 10, then the pairs are constructed by
            comparing the predicted survival time of sample i with the event time of all samples with event
            time of 10 or less.
            "Margin": the pairs are constructed between all samples. A best-guess time for the censored samples
            will be calculated and used to construct the pairs.
        :return: (float, float, int)
            The concordance index, the number of concordant pairs, and the number of total pairs.
        """
        # Choose prediction method based on the input argument
        if pair_method == "Margin" and (self.train_event_times is None or self.train_event_indicators is None):
            self._error_trainset("margin concordance")

        return concordance(self._predicted_times, self.event_times, self.event_indicators, self.train_event_times,
                           self.train_event_indicators, pair_method, ties)

    def mae(
            self,
            method: str = "Hinge",
            weighted: bool = True,
            reduction: bool = True,
            log_scale: bool = False
    ) -> float:
        """
        Calculate the MAE score for the test set.
        param method: string, default: "Hinge"
            The method used to calculate the MAE score.
            Options: "Uncensored", "Hinge" (default), "Margin", "IPCW-v1", "IPCW-v2", or "Pseudo_obs"\
        param weighted: bool, default: True
            Whether to use weighting scheme for MAE.
        param log_scale: boolean, default: False
            Whether to use log scale for the time axis.
        :return: float
            The MAE score for the test set.
        """
        return mean_error(
            predicted_times=self._predicted_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            error_type="absolute",
            method=method,
            weighted=weighted,
            reduction=reduction,
            log_scale=log_scale
        )

    def mse(
            self,
            method: str = "Hinge",
            weighted: bool = True,
            reduction: bool = True,
            log_scale: bool = False
    ) -> float:
        """
        Calculate the MAE score for the test set.
        param method: string, default: "Hinge"
            The method used to calculate the MAE score.
            Options: "Uncensored", "Hinge" (default), "Margin", "IPCW-v1", "IPCW-v2", or "Pseudo_obs"\
        param weighted: bool, default: True
            Whether to use weighting scheme for MAE.
        param log_scale: boolean, default: False
            Whether to use log scale for the time axis.
        :return: float
            The MAE score for the test set.
        """
        return mean_error(
            predicted_times=self._predicted_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            error_type="squared",
            method=method,
            weighted=weighted,
            reduction=reduction,
            log_scale=log_scale
        )


class QuantileRegEvaluator(SurvivalEvaluator):
    def __init__(
            self,
            quantile_regression: NumericArrayLike,
            quantile_levels: NumericArrayLike,
            test_event_times: NumericArrayLike,
            test_event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None,
            predict_time_method: str = "Median",
            interpolation: str = "Linear"
    ):
        """
        Initialize the quantile regression evaluator.
        Parameters
        ----------
        param quantile_regression: array-like, shape = (n_quantiles, n_samples)
            Predicted quantile curves for the testing samples.
        param quantile_levels: array-like, shape = (n_quantiles, )
            Quantile levels for the quantile curves.
        param test_event_times: structured array, shape = (n_samples, )
            Actual event/censor time for the testing samples.
        param test_event_indicators: structured array, shape = (n_samples, )
            Binary indicators of censoring for the testing samples
        param train_event_times: structured array, shape = (n_train_samples, )
            Actual event/censor time for the training samples.
        param train_event_indicators: structured array, shape = (n_train_samples, )
            Binary indicators of censoring for the training samples
        param predict_time_method: str, default = "Median"
            Method for calculating predicted survival time. Available options are "Median" and "Mean".
        param interpolation: str, default = "Linear"
            Method for interpolation. Available options are ['Linear', 'Pchip'].
        """
        survival_level = 1 - quantile_levels
        super(QuantileRegEvaluator, self).__init__(survival_level, quantile_regression, test_event_times,
                                                   test_event_indicators, train_event_times, train_event_indicators,
                                                   predict_time_method, interpolation)

    def predict_time_from_curve(
            self,
            predict_method: Callable,
    ) -> np.ndarray:
        """
        Predict survival time from survival curves.
        param predict_method: Callable
            A function that takes in a survival curve and returns a predicted survival time.
            There are two build-in methods: 'predict_median_survival_time' and 'predict_mean_survival_time'.
            'predict_median_survival_time' uses the median of the survival curve as the predicted survival time.
            'predict_mean_survival_time' uses the expected time of the survival curve as the predicted survival time.
        :return: np.ndarray
            Predicted survival time for each sample.
        """
        if (predict_method is not predict_mean_survival_time) and (predict_method is not predict_median_survival_time):
            error = "Prediction method must be 'predict_mean_survival_time' or 'predict_median_survival_time', " \
                    "got '{}' instead".format(predict_method.__name__)
            raise TypeError(error)

        predicted_times = []
        for i in range(self.time_coordinates.shape[0]):
            predicted_time = predict_method(self.predicted_curves, self.time_coordinates[i, :], self.interpolation)
            predicted_times.append(predicted_time)
        predicted_times = np.array(predicted_times)
        return predicted_times

    def predict_probability_from_curve(
            self,
            target_time: Union[float, int, np.ndarray],
    ) -> np.ndarray:
        """
        Predict a probability of event at a given time point from a predicted curve. Each predicted curve will only
        have one corresponding probability. Note that this method is different from the
        'predict_multi_probabilities_from_curve' method, which predicts the multiple probabilities at multiple time
        points from a predicted curve.
        param target_time: float, int, or array-like, shape = (n_samples, )
            Time point(s) at which the probability of event is to be predicted. If float or int, the same time point is
            used for all samples. If array-like, each sample will have it own target time. The length of the array must
            be the same as the number of samples.
        :return: array-like, shape = (n_samples, )
            Predicted probabilities of event at the target time point(s).
        """
        if isinstance(target_time, (float, int)):
            target_time = target_time * np.ones_like(self.event_times)
        elif isinstance(target_time, np.ndarray):
            assert target_time.ndim == 1, "Target time must be a 1D array"
            assert target_time.shape[0] == self.time_coordinates.shape[0], "Target time must have the same length as " \
                                                                           "the number of samples"
        else:
            error = "Target time must be a float, int, or 1D array, got '{}' instead".format(type(target_time))
            raise TypeError(error)

        predict_probs = []
        for i in range(self.time_coordinates.shape[0]):
            predict_prob = predict_prob_from_curve(self.predicted_curves, self.time_coordinates[i, :],
                                                   target_time[i], self.interpolation)
            predict_probs.append(predict_prob)
        predict_probs = np.array(predict_probs)
        return predict_probs

    def predict_multi_probabilities_from_curve(
            self,
            target_times: np.ndarray
    ) -> np.ndarray:
        """
        Predict the probability of event at multiple time points from the predicted curve.
        param target_times: array-like, shape = (n_target_times)
            Time points at which the probability of event is to be predicted.
        :return: array-like, shape = (n_samples, n_target_times)
            Predicted probabilities of event at the target time points.
        """
        predict_probs_mat = []
        for i in range(self.time_coordinates.shape[0]):
            predict_probs = predict_multi_probs_from_curve(self.predicted_curves, self.time_coordinates[i, :],
                                                           target_times, self.interpolation).tolist()
            predict_probs_mat.append(predict_probs)
        predict_probs_mat = np.array(predict_probs_mat)
        return predict_probs_mat

    def plot_survival_curves(
            self,
            curve_indices,
            color=None,
            x_lim: tuple = None,
            y_lim: tuple = None,
            x_label: str = 'Time',
            y_label: str = 'Survival probability'
    ):
        """Plot survival curves."""
        fig, ax = plt.subplots()
        ax.plot(self.time_coordinates[curve_indices, :].T, self.predicted_curves, color=color, label=curve_indices)
        if y_lim is None:
            ax.set_ylim(0, 1.02)
        else:
            ax.set_ylim(y_lim)

        if x_lim is not None:
            ax.set_xlim(x_lim)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
        return fig, ax

    def km_calibration(self, draw_figure: bool = False):
        """
        Calculate the KM calibration score from the predicted survival curve.
        :return: float
            KL divergence between the average predicted survival distribution and the Kaplan-Meier distribution.
        """
        unique_times = np.unique(self.event_times[self.event_indicators == 1])
        survival_curves = quantile_to_survival(1 - self.predicted_curves, self.time_coordinates,
                                               unique_times, interpolate=self.interpolation)
        avg_surv = np.mean(survival_curves, axis=0)

        return km_calibration(avg_surv, np.unique(self.event_times[self.event_indicators == 1]),
                              self.event_times, self.event_indicators,
                              interpolation_method=self.interpolation, draw_figure=draw_figure)
