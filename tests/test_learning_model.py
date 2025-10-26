from scripts.learning_model import learning_model


def test_learning_model(sample_prophet_df, mocker):
    fake_model = mocker.MagicMock()
    fake_model.order = (1, 0, 1)
    fake_model.seasonal_order = (0, 1, 0, 12)

    mock_arima = mocker.patch("pmdarima.auto_arima", return_value=fake_model)

    result = learning_model(sample_prophet_df)

    assert result == fake_model

    mock_arima.assert_called_once()
