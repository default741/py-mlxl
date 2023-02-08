import pandas as pd

data = {
    'col_1': [1, 2, 3, 4],
    'col_2': [10, 20, 30],
    'col_3': [100, 200, 300, 400, 500]
}

final_df = pd.DataFrame()

final_df = pd.concat(
    [final_df] + [pd.DataFrame({key: value}) for key, value in data.items()], axis=1)

print(final_df)
