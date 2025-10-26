from prophet import Prophet
from scripts.is_valid_country_name import is_valid_country_name
from scripts.is_valid_seasonality_prior_scale import is_valid_seasonality_prior_scale


def learning_prophet_model(train_df, future_df, seasonality_prior_scale=25.0, country_name='US'):
    if is_valid_country_name(country_name) and is_valid_seasonality_prior_scale(seasonality_prior_scale):
        model = Prophet(
            seasonality_prior_scale=seasonality_prior_scale
        )
        model.add_country_holidays(country_name=country_name)
        model.fit(train_df)

        prediction_dates = future_df[['ds']]
        forecast = model.predict(prediction_dates)
        return forecast