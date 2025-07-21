import pandas as pd
import numpy as np

data = pd.read_csv("data\\temperature_by_hour.csv")
temperature = data["Temperature (C)"].to_numpy()
temperature_new = np.repeat(temperature, 12)

np.savetxt("data\\temperature_5minute.csv", temperature_new, header = "temp (C)",comments='')