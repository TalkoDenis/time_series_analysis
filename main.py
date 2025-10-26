from scripts.get_forecast import get_forecast
from scripts.learning_sarimax_model import learning_sarimax_model
from scripts.read_data import read_data
from scripts.rename_columns import rename_columns
from scripts.split_df import split_df
from scripts.validate_content import validate_content
from scripts.validate_structure import validate_structure
from scripts.visualize_data import visualize_data
from scripts.learning_prophet_model import learning_prophet_model

df = read_data("./data/data.csv")
df = validate_structure(df)
df_prophet = rename_columns(df)
del df
df_prophet = validate_content(df_prophet)
train_df, test_df = split_df(df_prophet)

model = learning_sarimax_model(train_df, train_param="y", exogenous_param="ds")
forecast_df = get_forecast(model, train_df, test_df)

visualize_data(forecast_df, train_df)
