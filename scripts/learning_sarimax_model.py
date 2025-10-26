import warnings

import pmdarima as pm

from scripts.timing import timing

warnings.filterwarnings("ignore", category=FutureWarning)


@timing
def learning_sarimax_model(train_df, train_param="y", exogenous_param="ds"):
    print("Start learning the SARIMAX model...")
    SARIMAX_model = pm.auto_arima(
        train_df[[train_param]],
        exogenous=train_df[[exogenous_param]],
        start_p=1,
        start_q=1,
        test="adf",
        max_p=3,
        max_q=3,
        m=12,
        start_P=0,
        seasonal=True,
        d=None,
        D=1,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    print(
        "The Model is learned. Params:",
        SARIMAX_model.order,
        SARIMAX_model.seasonal_order,
    )
    return SARIMAX_model
