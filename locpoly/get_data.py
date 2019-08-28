import pandas as pd
import numpy as np

df = pd.read_csv("4_clear_data.csv") # SiO2_MgO_CaO_Al2O3_Tliq.csv
dc = pd.read_csv("clustering_data.csv")

mv = ["Al2O3", "CaO", "MgO", "SiO2"]

dx = pd.concat((df, dc.loc[:, ["cluster_id"]]), axis=1)
dx = dx.rename(columns={"Tliq/C": "Tliq_C"})

vn = ["Al2O3", "CaO", "MgO", "SiO2"]
dx = dx.loc[:, vn + ["Tliq_C", "cluster_id"]]

# Create transformed versions of all the composition variables
for v in vn:
    dx.loc[:, v + "_x"] = np.sqrt(dx.loc[:, v])
vnx = [v + "_x" for v in vn]
mnx = dx.loc[:, vnx].mean(0)

