import numpy as np
import emcee
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import njit
import scipy.stats
import string
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms

def plot_benefit_ratio(
    patient_data,
    benefit_results,
    show=True,
):
    """
    Plot prediction-error and uncertainty benefit ratios.

    Parameters
    ----------
    patient_data : dict
        Patient information containing treatment times.
    benefit_results : dict
        Output from ``calculate_benefit_ratios``.
    show : bool, optional
        Display the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Main axes object.
    """
    treatments = patient_data["treatments"]

    candidate_days = benefit_results["candidate_days"]
    benefit_pu = benefit_results["benefit_pu"]
    benefit_pe = benefit_results["benefit_pe"]

    mri2_day = benefit_results["reference_day"]
    earliest_day = benefit_results["earliest_day"]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        candidate_days,
        benefit_pu,
        "o",
        color="blue",
        label=r"$BR_{PU}$",
    )

    ax.plot(
        candidate_days,
        benefit_pe,
        "o",
        color="red",
        label=r"$BR_{PE}$",
    )

    # No change relative to reference MRI2
    ax.axhline(
        0,
        color="black",
        linestyle="--",
    )

    # Treatment times
    for treatment_time in treatments:
        ax.axvline(
            treatment_time,
            color="green",
            linestyle="-.",
        )

    # Actual MRI2
    ax.axvline(
        mri2_day,
        color="black",
        linestyle="--",
        label=r"$T_{MRI2}$",
    )

    # Earliest beneficial MRI2
    if earliest_day is not None:
        ax.axvline(
            earliest_day,
            color="orange",
            linestyle="--",
            label=r"$t^{**}_{MRI2}$",
        )

    # --------------------------------------------------
    # Axes formatting
    # --------------------------------------------------
    ax.set_xlabel(r"$t_{MRI2}$")
    ax.set_ylabel("Benefit ratio (BR)")
    ax.set_ylim(bottom=-5)

    ax.tick_params(labelsize=14)

    ax.legend(
        bbox_to_anchor=(0.05, 1),
        loc="upper left",
        frameon=False,
    )

    # --------------------------------------------------
    # Secondary x-axis for treatment times
    # --------------------------------------------------
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())

    treatment_labels = [
        rf"$A/C_{i}$"
        for i in range(1, len(treatments) + 1)
    ]

    ax_top.set_xticks(treatments)
    ax_top.set_xticklabels(treatment_labels)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax

def calculate_benefit_ratios(patient_data, patient_solutions):
    """
    Calculate prediction-error and uncertainty benefit ratios
    across candidate MRI2 timings.

    Parameters
    ----------
    patient_data : dict
        Patient information containing MRI visit times.
    patient_solutions : ndarray
        Array with columns:
        0 : MRI2 day
        1 : measured MRI3 tumor volume
        2 : median MRI3 prediction using true MRI2
        3 : median MRI3 prediction using fitted MRI2
        4 : MRI3 uncertainty width using true MRI2
        5 : MRI3 uncertainty width using fitted MRI2

    Returns
    -------
    dict
        Benefit ratios and relevant MRI2 timings.
    """
    mri2_day = patient_data["visits"][1]

    candidate_days = patient_solutions[:, 0]
    measured_v3 = patient_solutions[:, 1]
    predicted_v3 = patient_solutions[:, 3]
    uncertainty = patient_solutions[:, 5]

    # --------------------------------------------------
    # Find reference MRI2
    # --------------------------------------------------
    reference_indices = np.where(
        np.isclose(candidate_days, mri2_day)
    )[0]

    if reference_indices.size == 0:
        raise ValueError(
            f"MRI2 day {mri2_day} is not present in patient_solutions."
        )

    reference_index = reference_indices[0]

    # --------------------------------------------------
    # Reference uncertainty and prediction error
    # --------------------------------------------------
    reference_uncertainty = uncertainty[reference_index]

    prediction_error = np.abs(
        measured_v3 - predicted_v3
    )

    reference_error = prediction_error[reference_index]

    if np.isclose(reference_uncertainty, 0):
        raise ValueError("Reference uncertainty is zero.")

    if np.isclose(reference_error, 0):
        raise ValueError("Reference prediction error is zero.")

    # --------------------------------------------------
    # Benefit ratios
    # --------------------------------------------------
    br_pu = uncertainty / reference_uncertainty
    br_pe = prediction_error / reference_error

    # Improvement relative to actual MRI2
    benefit_pu = 1.0 - br_pu
    benefit_pe = 1.0 - br_pe

    # --------------------------------------------------
    # MRI2 times improving both quantities
    # --------------------------------------------------
    valid_indices = np.where(
        (br_pu <= 1.0) & (br_pe <= 1.0)
    )[0]

    earliest_day = None
    optimal_day = None

    if valid_indices.size > 0:
        earliest_index = valid_indices[0]
        earliest_day = candidate_days[earliest_index]

        br_sum = br_pu + br_pe

        optimal_index = valid_indices[
            np.argmin(br_sum[valid_indices])
        ]

        optimal_day = candidate_days[optimal_index]

    return {
        "candidate_days": candidate_days,
        "br_pu": br_pu,
        "br_pe": br_pe,
        "benefit_pu": benefit_pu,
        "benefit_pe": benefit_pe,
        "earliest_day": earliest_day,
        "optimal_day": optimal_day,
        "reference_day": mri2_day,
        "reference_error": reference_error,
        "reference_uncertainty": reference_uncertainty,
    }

def all_possible_fittings(
    l_bound,
    u_bound,
    patient_data,
    n_steps=500,
    burn_in=200,
    progress=False,
    rng=None,
):
    """
    Fit the model for every possible MRI2 timing using true and model-derived
    MRI2 tumor volumes, then predict tumor volume at MRI3.

    Parameters
    ----------
    l_bound : ndarray
        Lower parameter bounds.
    u_bound : ndarray
        Upper parameter bounds.
    patient_data : dict
        Patient data generated by ``generate_patient_data``.
    n_steps : int, optional
        Number of MCMC steps.
    burn_in : int, optional
        Number of MCMC samples discarded as burn-in.
    progress : bool, optional
        Display emcee progress bars.
    rng : numpy.random.Generator, optional
        Random-number generator.

    Returns
    -------
    ndarray
        Array with columns:

        0. MRI2 day
        1. Measured tumor volume at MRI3
        2. Median MRI3 prediction using true MRI2
        3. Median MRI3 prediction using fitted MRI2
        4. MRI3 posterior IQR using true MRI2
        5. MRI3 posterior IQR using fitted MRI2
    """
    l_bound = np.asarray(l_bound, dtype=float)
    u_bound = np.asarray(u_bound, dtype=float)

    if rng is None:
        rng = np.random.default_rng()

    ndim = l_bound.size
    nwalkers = 2 * ndim

    treatments = patient_data["treatments"]

    baseline_day = patient_data["visits"][0]
    v3_day = patient_data["visits"][2]

    baseline_volume = patient_data["measurements"][0]
    measured_v3 = patient_data["measurements"][-1]

    # Candidate MRI2 days and corresponding tumor volumes
    candidate_days = patient_data["days"][1:-1]
    true_v2_values = patient_data["full_data"][1:-1]
    fitted_v2_values = patient_data["full_model"][1:-1]

    n_candidates = candidate_days.size

    # Preallocate output array
    patient_solutions = np.empty((n_candidates, 6))

    # --------------------------------------------------
    # Loop over candidate MRI2 times
    # --------------------------------------------------
    for j, (v2_day, true_v2, fitted_v2) in enumerate(
        zip(
            candidate_days,
            true_v2_values,
            fitted_v2_values,
        )
    ):
        fit_times = np.array(
            [baseline_day, v2_day],
            dtype=float,
        )

        # ==================================================
        # Fit using TRUE MRI2 value
        # ==================================================
        measurements = np.array(
            [baseline_volume, true_v2],
            dtype=float,
        )

        initial_positions = rng.uniform(
            l_bound,
            u_bound,
            size=(nwalkers, ndim),
        )

        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            args=(
                fit_times,
                measurements,
                treatments,
                l_bound,
                u_bound,
            ),
        )

        sampler.run_mcmc(
            initial_positions,
            n_steps,
            progress=progress,
        )

        flat_samples = sampler.get_chain(
            discard=burn_in,
            flat=True,
        )

        median_true, q1_true, q3_true = summarize_v3_posterior(
            flat_samples,
            baseline_day,
            v3_day,
            treatments,
        )

        # ==================================================
        # Fit using MODEL-DERIVED MRI2 value
        # ==================================================
        measurements[1] = fitted_v2

        initial_positions = rng.uniform(
            l_bound,
            u_bound,
            size=(nwalkers, ndim),
        )

        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            args=(
                fit_times,
                measurements,
                treatments,
                l_bound,
                u_bound,
            ),
        )

        sampler.run_mcmc(
            initial_positions,
            n_steps,
            progress=progress,
        )

        flat_samples = sampler.get_chain(
            discard=burn_in,
            flat=True,
        )

        median_fit, q1_fit, q3_fit = summarize_v3_posterior(
            flat_samples,
            baseline_day,
            v3_day,
            treatments,
        )

        # ==================================================
        # Store results
        # ==================================================
        patient_solutions[j] = (
            v2_day,
            measured_v3,
            median_true,
            median_fit,
            q3_true - q1_true,
            q3_fit - q1_fit,
        )

    return patient_solutions

def summarize_v3_posterior(
    flat_samples,
    baseline_day,
    v3_day,
    treatments,
):
    """
    Compute the posterior median and IQR of tumor volume at V3.

    Parameters
    ----------
    flat_samples : ndarray
        Posterior parameter samples.
    baseline_day : float
        Baseline imaging time.
    v3_day : float
        V3 imaging time.
    treatments : ndarray
        Treatment administration times.

    Returns
    -------
    median : float
        Posterior median tumor volume at V3.
    q1 : float
        25th percentile at V3.
    q3 : float
        75th percentile at V3.
    """
    prediction_times = np.array(
        [baseline_day, v3_day],
        dtype=float,
    )

    trajectories = evaluate_posterior_trajectories(
        flat_samples,
        prediction_times,
        treatments,
    )

    v3_predictions = trajectories[:, -1]

    q1, median, q3 = np.percentile(
        v3_predictions,
        [2.5, 50, 97.5],
    )

    return median, q1, q3

def summarize_posterior_trajectories(
    flat_samples,
    data_times,
    treatments,
    time_step=1.0,
    n_samples=None,
    rng=None,
):
    """
    Compute the posterior median and interquartile range of tumor trajectories.

    Parameters
    ----------
    flat_samples : ndarray
        Posterior parameter samples with shape (n_samples, n_parameters).
    data_times : array_like
        Observed imaging times.
    treatments : array_like
        Treatment administration times.
    time_step : float, optional
        Time interval between model evaluations. Default is 1 day.
    n_samples : int or None, optional
        Number of posterior samples used to generate trajectories.
        If None, all posterior samples are used.
    rng : numpy.random.Generator, optional
        Random-number generator used when subsampling posterior samples.

    Returns
    -------
    times : ndarray
        Model evaluation times.
    median : ndarray
        Posterior median tumor trajectory.
    q1 : ndarray
        25th percentile tumor trajectory.
    q3 : ndarray
        75th percentile tumor trajectory.
    """
    flat_samples = np.asarray(flat_samples)
    data_times = np.asarray(data_times)
    treatments = np.asarray(treatments)

    times = np.arange(
        data_times[0],
        data_times[-1] + time_step,
        time_step,
    )

    # Optionally use only a subset of posterior samples
    if n_samples is not None and n_samples < len(flat_samples):
        if rng is None:
            rng = np.random.default_rng()

        indices = rng.choice(
            len(flat_samples),
            size=n_samples,
            replace=False,
        )

        samples = flat_samples[indices]

    else:
        samples = flat_samples

    # Evaluate posterior trajectories
    trajectories = evaluate_posterior_trajectories(
        samples,
        times,
        treatments,
    )

    # Compute IQR and median across posterior samples
    q1, median, q3 = np.percentile(
        trajectories,
        [2.5, 50, 97.5],
        axis=0,
    )

    return times, median, q1, q3

@njit(cache=True)
def evaluate_posterior_trajectories(samples, times, treatments):
    """
    Evaluate tumor trajectories for multiple posterior parameter samples.

    Parameters
    ----------
    samples : ndarray
        Posterior parameter samples with shape (n_samples, n_parameters).
    times : ndarray
        Times at which to evaluate the model.
    treatments : ndarray
        Treatment administration times.

    Returns
    -------
    ndarray
        Tumor trajectories with shape (n_samples, n_times).
    """
    n_samples = samples.shape[0]
    n_times = times.size

    trajectories = np.empty((n_samples, n_times))

    for i in range(n_samples):
        sample = samples[i]

        trajectories[i] = treated_tumor_exact(
            times,
            sample[-2],  # initial tumor volume
            sample[0],   # r
            sample[1],   # a
            sample[2],   # b
            treatments,
        )

    return trajectories

@njit(cache=True)
def log_prior(theta, l_bound, u_bound):
    """
    Compute the log-prior probability for uniform parameter bounds.

    Parameters
    ----------
    theta : ndarray
        Model parameters.
    l_bound : ndarray
        Lower parameter bounds.
    u_bound : ndarray
        Upper parameter bounds.

    Returns
    -------
    float
        0.0 if all parameters are within bounds; otherwise -inf.
    """
    for i in range(theta.size):
        if theta[i] <= l_bound[i] or theta[i] >= u_bound[i]:
            return -np.inf

    return 0.0


@njit(cache=True)
def log_probability(
    theta,
    times,
    tumor_volume,
    treatments,
    l_bound,
    u_bound,
):
    """
    Compute the log-posterior probability.

    Parameters
    ----------
    theta : ndarray
        Model parameters.
    times : ndarray
        Measurement times.
    tumor_volume : ndarray
        Observed tumor volumes.
    treatments : ndarray
        Treatment administration times.
    l_bound : ndarray
        Lower parameter bounds.
    u_bound : ndarray
        Upper parameter bounds.

    Returns
    -------
    float
        Log-posterior probability.
    """
    log_p = log_prior(theta, l_bound, u_bound)

    if log_p == -np.inf:
        return -np.inf

    return log_likelihood(
        theta,
        times,
        tumor_volume,
        treatments,
    )

@njit(cache=True)
def log_likelihood(theta, times, y, treatments):
    """
    Compute the Gaussian log-likelihood of the tumor model.

    Parameters
    ----------
    theta : ndarray
        Model parameters [r, a, b, ..., y0, sigma].
    times : ndarray
        Measurement times.
    y : ndarray
        Observed tumor volumes.
    treatments : ndarray
        Treatment administration times.

    Returns
    -------
    float
        Log-likelihood.
    """
    r = theta[0]
    a = theta[1]
    b = theta[2]
    y0 = theta[-2]
    sigma = theta[-1]

    model = treated_tumor_exact(
        times,
        y0,
        r,
        a,
        b,
        treatments,
    )

    residuals = y - model
    variance = sigma * sigma

    return -0.5 * (
        np.sum(residuals * residuals) / variance
        + len(y) * np.log(2.0 * np.pi * variance)
    )

@njit(cache=True)
def treated_tumor_exact(times, y0, r, a, b, treatments):
    """
    Evaluate the exact solution of the treated tumor model.

    Parameters
    ----------
    times : ndarray
        Times at which the tumor volume is evaluated.
    y0 : float
        Tumor volume at times[0].
    r : float
        Tumor proliferation rate.
    a : float
        Treatment efficacy.
    b : float
        Treatment-effect decay rate.
    treatments : ndarray
        Treatment administration times.

    Returns
    -------
    ndarray
        Tumor volume evaluated at each time.
    """
    n_times = len(times)
    model = np.empty(n_times)

    t0 = times[0]

    for j in range(n_times):
        t = times[j]

        exponent = r * (t - t0)

        for treatment_time in treatments:

            if t > treatment_time:

                # Treatment may have started before the initial time.
                start = max(t0, treatment_time)
                duration = t - start

                if duration > 0.0:

                    if abs(b) < 1e-12:
                        treatment_integral = duration
                    else:
                        treatment_integral = (
                            np.exp(-b * (start - treatment_time))
                            * (-np.expm1(-b * duration))
                            / b
                        )

                    exponent -= a * treatment_integral

        model[j] = y0 * np.exp(exponent)

    return model

def generate_patient_data(
    l_bound,
    u_bound,
    scans_treatment_dates,
    true_parameters=None,
    plot_figs=False,
    n_steps=5000,
    burn_in=2000,
    progress=False,
    rng=None,
):
    """
    Generate a synthetic patient and calibrate the tumor model using MCMC.

    Parameters
    ----------
    l_bound : array_like
        Lower bounds for the model parameters.
    u_bound : array_like
        Upper bounds for the model parameters.
    scans_treatment_dates : array_like
        Rows containing MRI visit times followed by treatment times.
    plot_figs : bool, optional
        Plot the generated and fitted tumor trajectories.
    n_steps : int, optional
        Number of MCMC steps.
    burn_in : int, optional
        Number of MCMC samples discarded as burn-in.
    progress : bool, optional
        Display the emcee progress bar.
    rng : numpy.random.Generator, optional
        Random-number generator. If None, a new generator is created.

    Returns
    -------
    dict
        Dictionary containing patient data, fitted model, confidence intervals,
        and concordance correlation coefficient.
    """

    # --------------------------------------------------
    # Input preparation
    # --------------------------------------------------
    l_bound = np.asarray(l_bound, dtype=float)
    u_bound = np.asarray(u_bound, dtype=float)
    scans_treatment_dates = np.asarray(
        scans_treatment_dates,
        dtype=float,
    )

    if l_bound.shape != u_bound.shape:
        raise ValueError("l_bound and u_bound must have the same shape.")

    if np.any(l_bound >= u_bound):
        raise ValueError("Every lower bound must be smaller than its upper bound.")

    if burn_in >= n_steps:
        raise ValueError("burn_in must be smaller than n_steps.")

    if rng is None:
        rng = np.random.default_rng()

    ndim = l_bound.size

    # --------------------------------------------------
    # Generate true patient parameters
    # --------------------------------------------------
    if true_parameters is None:
        true_pars = rng.uniform(l_bound, u_bound)
    else:
        true_pars = np.asarray(true_parameters, dtype=float)

        if true_pars.shape != l_bound.shape:
            raise ValueError(
                "true_parameters must contain one value per model parameter."
            )

        if np.any(true_pars <= l_bound) or np.any(true_pars >= u_bound):
            raise ValueError(
                "All parameter values must lie within their prescribed bounds."
            )    

    # Randomly select an imaging/treatment schedule
    row_index = rng.integers(scans_treatment_dates.shape[0])
    schedule = scans_treatment_dates[row_index]

    visits = schedule[:3].copy()
    treatments = schedule[-4:].copy()

    # Daily time grid between MRI1 and MRI3
    times = np.arange(visits[0], visits[2] + 1)

    # --------------------------------------------------
    # Generate true tumor trajectory
    # --------------------------------------------------
    model_args = (
        true_pars[0],
        true_pars[1],
        true_pars[2],
        *treatments,
    )
    
    true_data = treated_tumor_exact(
        times,
        true_pars[-2],
        true_pars[0],
        true_pars[1],
        true_pars[2],
        treatments,
    )

    # Find indices corresponding to the MRI measurements
    visit_indices = np.searchsorted(times, visits)

    if (
        np.any(visit_indices >= times.size)
        or not np.allclose(times[visit_indices], visits)
    ):
        raise ValueError(
            "MRI visit times must correspond to points on the daily time grid."
        )

    measured = true_data[visit_indices]

    # --------------------------------------------------
    # Plot synthetic patient
    # --------------------------------------------------
    if plot_figs:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=150)

        ax.plot(
            times,
            true_data/1000,
            color="black",
            linewidth=2,
            label="True data",
        )

        ax.scatter(
            visits,
            measured/1000,
            color="red",
            label="Measured data",
        )

        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Tumor volume (cm³)")
        ax.legend(frameon=False)

        fig.savefig(
            "just_data.pdf",
            bbox_inches="tight",
            pad_inches=0.02,
        )

        plt.show()

    # --------------------------------------------------
    # MCMC calibration
    # --------------------------------------------------
    nwalkers = 2 * ndim

    initial_positions = rng.uniform(
        l_bound,
        u_bound,
        size=(nwalkers, ndim),
    )

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(
            visits,
            measured,
            treatments,
            l_bound,
            u_bound,
        ),
    )

    sampler.run_mcmc(
        initial_positions,
        n_steps,
        progress=progress,
    )

    flat_samples = sampler.get_chain(
        discard=burn_in,
        flat=True,
    )

    # --------------------------------------------------
    # Posterior trajectory
    # --------------------------------------------------
    solution_times, solution_median, solution_q1, solution_q3 = (
        summarize_posterior_trajectories(
            flat_samples,
            visits,
            treatments,
        )
    )

    # --------------------------------------------------
    # Match fitted solution to true-data time grid
    # --------------------------------------------------
    solution_indices = np.searchsorted(
        solution_times,
        times,
    )

    if (
        np.any(solution_indices >= solution_times.size)
        or not np.allclose(
            solution_times[solution_indices],
            times,
        )
    ):
        raise ValueError(
            "Fitted solution times do not match the patient time grid."
        )

    full_model = solution_median[solution_indices]
    lower_ci = solution_q1[solution_indices]
    upper_ci = solution_q3[solution_indices]

    # --------------------------------------------------
    # Plot fitted trajectory
    # --------------------------------------------------
    if plot_figs:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=150)

        ax.plot(
            times,
            true_data/1000,
            color="black",
            linewidth=2,
            label="True data",
        )

        ax.scatter(
            visits,
            measured/1000,
            color="red",
            label="Measured data",
        )

        ax.fill_between(
            solution_times,
            solution_q1/1000,
            solution_q3/1000,
            alpha=0.25,
            color="blue",
            label="Fitted model",
        )

        ax.plot(
            solution_times,
            solution_q1/1000,
            linestyle="--",
            linewidth=1,
            color="blue",
        )

        ax.plot(
            solution_times,
            solution_q3/1000,
            linestyle="--",
            linewidth=1,
            color="blue",
        )

        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Tumor volume (cm³)")
        ax.legend(frameon=False)

        fig.savefig(
            "fitted_model.pdf",
            bbox_inches="tight",
            pad_inches=0.02,
        )

        plt.show()

    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    patient_data = {
        "visits": visits,
        "measurements": measured,
        "treatments": treatments,
        "days": times,
        "full_data": true_data,
        "full_model": solution_median,
        "q1": solution_q1,
        "q3": solution_q3,
    }

    return patient_data

l_bound = np.array([6.887058479539200605e-03, 4.766431028623440563e-02, 8.000000000000000167e-03, 1.999255789600000242e+02, 1.000000000000000021e-02])
u_bound = np.array([1.382448600534400163e-02, 6.345284814755555169e-01, 2.428954025700455599e+00, 1.474758722399999970e+05, 1.000000000000000000e+04])

# Load scan and treatment schedule data.
# Each row corresponds to a single subject / simulated patient.
# Columns:
# V1, V2, V3 - days of different imaging time points
# t1–t4      - treatment administration times
scans_treatment_dates = np.array([
    [0.0, 32.0, 62.0, 7.0, 21.0, 35.0, 49.0]
])

true_parameters = np.array([1.10689951e-02, 3.92920409e-01, 8.19531536e-01, 2.94391580e+04, 1.94040777e+02])