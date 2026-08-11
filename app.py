import numpy as np
import plotly.graph_objects as go

from dash import (
    Dash,
    Input,
    Output,
    State,
    callback,
    dcc,
    html,
)

from oed_model import (
    generate_patient_data,
    all_possible_fittings,
    calculate_benefit_ratios,
    l_bound,
    u_bound,
    scans_treatment_dates,
)


# ============================================================
# Default model parameters
# ============================================================

DEFAULT_PARAMETERS = {
    "r": 1.10689951e-02,
    "a": 3.92920409e-01,
    "b": 8.19531536e-01,
    "ic": 2.94391580e04,
    "std": 3.94040777e02,
}


# ============================================================
# Plot functions
# ============================================================

def create_patient_figure(patient_data):
    """Create Plotly figure showing patient data and fitted model."""

    days = patient_data["days"]

    true_data = patient_data["full_data"] / 1000
    median = patient_data["full_model"] / 1000

    # These currently correspond to 2.5 and 97.5 percentiles
    lower = patient_data["q1"] / 1000
    upper = patient_data["q3"] / 1000

    visits = patient_data["visits"]
    measurements = patient_data["measurements"] / 1000

    fig = go.Figure()

    # --------------------------------------------------------
    # 95% posterior interval
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=days,
            y=upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # fig.add_trace(
    #     go.Scatter(
    #         x=days,
    #         y=lower,
    #         mode="lines",
    #         line=dict(width=0),
    #         fill="tonexty",
    #         name="95% credible interval",
    #         hoverinfo="skip",
    #     )
    # )

    # --------------------------------------------------------
    # Fitted trajectory
    # --------------------------------------------------------

    # fig.add_trace(
    #     go.Scatter(
    #         x=days,
    #         y=median,
    #         mode="lines",
    #         name="Fitted model",
    #         line=dict(width=2),
    #     )
    # )

    # --------------------------------------------------------
    # True trajectory
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=days,
            y=true_data,
            mode="lines",
            name="True data",
            line=dict(
                color="black",
                width=2,
            ),
        )
    )

    # --------------------------------------------------------
    # MRI measurements
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=visits,
            y=measurements,
            mode="markers",
            name="Measured data",
            marker=dict(
                color="red",
                size=9,
            ),
        )
    )

    fig.update_layout(
        title="Patient-specific model",
        xaxis_title="Time (days)",
        yaxis_title="Tumor volume (cm³)",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=60, r=20, t=90, b=60),
    )

    return fig


def create_benefit_figure(patient_data, benefit_results):
    """Create Plotly figure showing benefit ratios."""

    candidate_days = benefit_results["candidate_days"]
    benefit_pu = benefit_results["benefit_pu"]
    benefit_pe = benefit_results["benefit_pe"]

    reference_day = benefit_results["reference_day"]
    earliest_day = benefit_results["earliest_day"]

    treatments = patient_data["treatments"]

    fig = go.Figure()

    # --------------------------------------------------------
    # Prediction uncertainty
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=candidate_days,
            y=benefit_pu,
            mode="markers",
            name="BR_PU",
            marker=dict(
                color="blue",
                size=7,
            ),
        )
    )

    # --------------------------------------------------------
    # Prediction error
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=candidate_days,
            y=benefit_pe,
            mode="markers",
            name="BR_PE",
            marker=dict(
                color="red",
                size=7,
            ),
        )
    )

    # Reference y = 0
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="black",
    )

    # Treatment times
    for i, treatment_time in enumerate(treatments, start=1):

        fig.add_vline(
            x=treatment_time,
            line_dash="dashdot",
            line_color="green",
        )

        fig.add_annotation(
            x=treatment_time,
            y=1.02,
            xref="x",
            yref="paper",
            text=f"A/C<sub>{i}</sub>",
            showarrow=False,
        )

    # Actual MRI2
    fig.add_vline(
        x=reference_day,
        line_dash="dash",
        line_color="black",
    )

    # Earliest beneficial MRI2
    if earliest_day is not None:
        fig.add_vline(
            x=earliest_day,
            line_dash="dash",
            line_color="orange",
        )

        fig.add_annotation(
            x=earliest_day,
            y=0.90,
            xref="x",
            yref="paper",
            text="t**<sub>MRI2</sub>",
            showarrow=False,
            font=dict(color="orange"),
        )

    fig.update_layout(
        title="Optimal MRI2 timing",
        xaxis_title="t_MRI2 (days)",
        yaxis_title="Benefit ratio (BR)",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=60, r=20, t=90, b=60),
    )

    fig.update_yaxes(
        range=[
            -5,
            None,
        ]
    )

    return fig


def empty_figure(message):
    """Create an empty figure displayed before simulation."""

    fig = go.Figure()

    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16),
    )

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    return fig


# ============================================================
# Dash application
# ============================================================

app = Dash(__name__)

app.title = "Optimizing MRI Acquisition Timing via Bayesian Data Assimilation to Improve Response Prediction Accuracy During Breast Cancer Neoadjuvant Therapy"


# ============================================================
# Parameter input helper
# ============================================================

def parameter_input(label, parameter_id, value):
    return html.Div(
        [
            html.Label(
                label,
                style={
                    "fontWeight": "600",
                    "marginBottom": "4px",
                },
            ),

            dcc.Input(
                id=parameter_id,
                type="number",
                value=value,
                debounce=True,
                style={
                    "width": "100%",
                    "padding": "8px",
                    "boxSizing": "border-box",
                },
            ),
        ],
        style={
            "marginBottom": "15px",
        },
    )


# ============================================================
# Layout
# ============================================================

app.layout = html.Div(
    [
        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------
        html.H1(
            "Optimizing MRI Acquisition Timing via Bayesian Data Assimilation to Improve Response Prediction Accuracy During Breast Cancer Neoadjuvant Therapy",
            style={
                "textAlign": "center",
                "marginBottom": "30px",
            },
        ),

        # ----------------------------------------------------
        # Main area
        # ----------------------------------------------------
        html.Div(
            [
                # ==================================================
                # LEFT: controls
                # ==================================================
                html.Div(
                    [
                        html.H3("Model Parameters"),

                        parameter_input(
                            "Tumor proliferation rate (r)",
                            "input-r",
                            DEFAULT_PARAMETERS["r"],
                        ),

                        parameter_input(
                            "Treatment efficacy (a)",
                            "input-a",
                            DEFAULT_PARAMETERS["a"],
                        ),

                        parameter_input(
                            "Treatment effect decay rate (b)",
                            "input-b",
                            DEFAULT_PARAMETERS["b"],
                        ),

                        parameter_input(
                            "Initial tumor volume (ic)",
                            "input-ic",
                            DEFAULT_PARAMETERS["ic"],
                        ),

                        parameter_input(
                            "Measurement/model error (std)",
                            "input-std",
                            DEFAULT_PARAMETERS["std"],
                        ),

                        html.Button(
                            "Run Simulation",
                            id="run-button",
                            n_clicks=0,
                            style={
                                "width": "100%",
                                "padding": "12px",
                                "fontSize": "16px",
                                "fontWeight": "600",
                                "cursor": "pointer",
                                "marginTop": "10px",
                            },
                        ),

                        html.Div(
                            id="simulation-status",
                            style={
                                "marginTop": "20px",
                                "fontSize": "14px",
                            },
                        ),
                    ],
                    style={
                        "width": "280px",
                        "minWidth": "280px",
                        "padding": "20px",
                        "border": "1px solid #ddd",
                        "borderRadius": "8px",
                        "height": "fit-content",
                    },
                ),

                # ==================================================
                # RIGHT: figures
                # ==================================================
                html.Div(
                    [
                        dcc.Loading(
                            dcc.Graph(
                                id="patient-figure",
                                figure=empty_figure(
                                    "Click Run Simulation"
                                ),
                            ),
                            type="circle",
                        ),

                        dcc.Loading(
                            dcc.Graph(
                                id="benefit-figure",
                                figure=empty_figure(
                                    "Click Run Simulation"
                                ),
                            ),
                            type="circle",
                        ),
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "20px",
                        "width": "100%",
                    },
                ),
            ],
            style={
                "display": "flex",
                "gap": "30px",
                "alignItems": "flex-start",
            },
        ),
    ],
    style={
        "maxWidth": "1600px",
        "margin": "0 auto",
        "padding": "25px",
        "fontFamily": "Arial, sans-serif",
    },
)


# ============================================================
# Callback
# ============================================================

@callback(
    Output("patient-figure", "figure"),
    Output("benefit-figure", "figure"),
    Output("simulation-status", "children"),

    Input("run-button", "n_clicks"),

    State("input-r", "value"),
    State("input-a", "value"),
    State("input-b", "value"),
    State("input-ic", "value"),
    State("input-std", "value"),

    prevent_initial_call=True,
)
def run_simulation(
    n_clicks,
    r,
    a,
    b,
    ic,
    std,
):

    true_parameters = np.array(
        [
            r,
            a,
            b,
            ic,
            std,
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # Generate simulated patient
    # --------------------------------------------------------

    patient_data = generate_patient_data(
        l_bound,
        u_bound,
        scans_treatment_dates,
        true_parameters=true_parameters,
        plot_figs=False,
        n_steps=5000,
        burn_in=100,
        progress=False,
    )

    # --------------------------------------------------------
    # Evaluate possible MRI2 timings
    # --------------------------------------------------------

    patient_solutions = all_possible_fittings(
        l_bound,
        u_bound,
        patient_data,
        n_steps=500,
        burn_in=200,
        progress=False,
    )

    # --------------------------------------------------------
    # Calculate benefit ratios
    # --------------------------------------------------------

    benefit_results = calculate_benefit_ratios(
        patient_data,
        patient_solutions,
    )

    # --------------------------------------------------------
    # Create figures
    # --------------------------------------------------------

    patient_figure = create_patient_figure(
        patient_data
    )

    benefit_figure = create_benefit_figure(
        patient_data,
        benefit_results,
    )

    earliest_day = benefit_results["earliest_day"]
    optimal_day = benefit_results["optimal_day"]

    status = (
        f"Earliest beneficial MRI2: day {earliest_day:.0f} | "
        f"Optimal MRI2: day {optimal_day:.0f}"
        if earliest_day is not None
        else "No MRI2 timing improved both criteria."
    )

    return (
        patient_figure,
        benefit_figure,
        status,
    )


# ============================================================
# Run application
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)