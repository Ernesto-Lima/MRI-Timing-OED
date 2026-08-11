# Optimal Experimental Design for MRI Timing

This repository contains an **in silico demonstration** of the optimal experimental design framework developed in the manuscript:

> **Optimizing MRI Acquisition Timing via Bayesian Data Assimilation to Improve Response Prediction Accuracy During Breast Cancer Neoadjuvant Therapy**  
> Chengyue Wu, Casey E. Stowers, Zhan Xu, Clinton Yam, Jingfei Ma, Gaiane M. Rauch, Thomas E. Yankeelov, and Ernesto A.B.F. Lima

The example illustrates how a mechanism-based model of tumor response, Bayesian calibration, and synthetic imaging measurements can be combined to investigate how the timing of a mid-treatment MRI affects the accuracy and uncertainty of treatment-response prediction.

This repository is intended as a **demonstration of the methodology** described in the paper. It does not contain patient data and is not intended for clinical use.

## Overview

The model describes changes in tumor volume during neoadjuvant therapy using tumor proliferation and treatment-induced cell death. The demonstration uses a virtual patient with user-defined model parameters and a fixed imaging/treatment schedule.

For each simulation, the code:

1. Generates an in silico tumor-response trajectory.
2. Samples tumor volume at three MRI time points.
3. Performs Bayesian calibration using the three synthetic MRI measurements.
4. Uses the calibrated trajectory to generate synthetic MRI2 measurements at candidate acquisition times.
5. Re-calibrates the model using MRI1 and each candidate MRI2 measurement.
6. Predicts tumor volume at MRI3.
7. Quantifies prediction error and prediction uncertainty relative to the reference MRI2 timing.
8. Identifies:
   - the **earliest MRI2 time** that does not increase prediction error or uncertainty; and
   - the **optimal MRI2 time** that minimizes the combined prediction-error and prediction-uncertainty criteria.

For computational efficiency, the implementation evaluates the exact solution of the scalar tumor-response ODE rather than repeatedly performing numerical ODE integration.

## Interactive Dash application

The repository includes a Dash interface titled **Optimal Experimental Design**.

The interface allows the user to specify the virtual-patient parameters:

| Parameter | Description | Default |
|---|---|---:|
| `r` | Tumor proliferation rate | `1.10689951e-02` |
| `a` | Treatment efficacy coefficient | `3.92920409e-01` |
| `b` | Treatment-effect decay rate | `8.19531536e-01` |
| `ic` | Initial tumor volume | `2.94391580e+04` |
| `std` | Standard deviation of measurement/model error used in calibration | `3.94040777e+02` |

After entering the desired values, click **Run Simulation**.

The application displays:

- the simulated patient tumor trajectory, synthetic MRI measurements, fitted trajectory, and posterior uncertainty interval; and
- the benefit-ratio analysis across candidate MRI2 acquisition times, including the reference MRI2 time and the earliest beneficial MRI2 time.

## Repository structure

```text
.
├── app.py
├── oed_model.py
├── environment.yml
├── requirements.txt
└── README.md
```

- `app.py` — Dash graphical interface.
- `oed_model.py` — mathematical model, Bayesian calibration, virtual-patient generation, MRI-timing analysis, and benefit-ratio calculations.
- `environment.yml` — reproducible Conda environment.
- `requirements.txt` — Python package requirements.
- `README.md` — project documentation.

## Installation

### Recommended: Conda environment

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <REPOSITORY-NAME>
```

Create the environment from the provided file:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate oed
```

If you update `environment.yml` later, the environment can be updated with:

```bash
conda env update -f environment.yml --prune
```

### Alternative: create the environment manually

```bash
conda create -n oed python=3.10
conda activate oed
```

Install Numba through Conda:

```bash
conda install numba
```

Then install the remaining Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Installing Numba through Conda is recommended because `llvmlite`, one of its dependencies, may otherwise require a local LLVM toolchain on some systems.

## Running the application

Activate the environment:

```bash
conda activate oed
```

Start the Dash server:

```bash
python app.py
```

The terminal should display an address similar to:

```text
http://127.0.0.1:8050/
```

Open that address in a web browser.

To stop the application, return to the terminal and press:

```text
Ctrl + C
```

## Methodological interpretation

The purpose of this example is to demonstrate the paper's **synthesis, re-calibration, evaluation** workflow.

A fully calibrated virtual patient provides a continuous tumor-response trajectory. Candidate synthetic MRI2 measurements are sampled from this trajectory at different times. For each candidate time, the model is re-calibrated using MRI1 and the candidate MRI2 measurement and then propagated to MRI3.

The resulting MRI3 prediction is evaluated using:

- **Prediction error (PE):** difference between the predicted and reference MRI3 tumor volume.
- **Prediction uncertainty (PU):** width of the posterior prediction interval at MRI3.
- **Benefit ratios:** PE and PU relative to those obtained using the reference MRI2 acquisition time.

These quantities are used to investigate how MRI acquisition timing changes the predictive value of the mid-treatment scan.

## Important notes

- This is an **in silico demonstration**, not the complete clinical cohort analysis from the manuscript.
- No patient-identifiable information or clinical dataset is included.
- The software is provided for research and educational purposes.
- The output should not be interpreted as clinical guidance or used for patient management.
- Model parameters, schedules, and numerical settings in this example can be modified for experimentation.

## Citation

If you use this code, please cite the associated manuscript:

> Wu C, Stowers CE, Xu Z, Yam C, Ma J, Rauch GM, Yankeelov TE, Lima EABF.  
> **Optimizing MRI Acquisition Timing via Bayesian Data Assimilation to Improve Response Prediction Accuracy During Breast Cancer Neoadjuvant Therapy.**

## License

This project is released under the **MIT License**.

You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, provided that the original copyright and license notice are retained.

See the [`LICENSE`](LICENSE) file for the full license text.

If you use this software in academic or scientific work, please also cite the associated manuscript listed in the **Citation** section above.

## Contact

For questions about the computational implementation:

**Ernesto A.B.F. Lima, Ph.D.**  
Oden Institute for Computational Engineering and Sciences  
The University of Texas at Austin  
Email: ernesto.lima@utexas.edu
