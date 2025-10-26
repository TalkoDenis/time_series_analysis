def is_valid_seasonality_prior_scale(seasonality_prior_scale: float) -> bool:
    if seasonality_prior_scale >= 0 and seasonality_prior_scale <= 100:
        raise Exception(f'{seasonality_prior_scale} is not valid!')
    return True