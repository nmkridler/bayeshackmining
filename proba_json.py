import pandas as pd
import numpy as np
import json

def main():
	coefs = json.load(open("coef.json", "r"))

	# Mine Names
	cols = ["MINE_ID", "MINE_NAME"]
	mines = pd.read_csv("../data/AddressOfRecord.txt", sep="|", usecols=cols, dtype={"MINE_ID": np.float64})
	df = pd.read_csv("last_data.csv")
	df = pd.merge(df, mines, how="inner", on="MINE_ID", sort=False)
	df = df.drop("ACCIDENT_DT", 1)
	df["risk"] = 0.
	for key, value in coefs.items():
		if key == "Intercept":
			df["risk"] += float(value)

		else:
			df["risk"] += float(value) * df[key]

	df["risk"] = 1./(1. + np.exp(-df["risk"]))
	df = df.sort_index(by="risk", ascending=False)
	f = open("output.json", "w")
	json_rows = []
	for i, row in enumerate(df.iterrows()):
		if i%10 == 0:
			json_rows.append(row[1].to_dict())

	print df.risk.min(), df.risk.max()
	f.write(json.dumps(json_rows))
	f.close()

	df.to_csv("last_data_with_proba.csv", index=False)

if __name__ == "__main__":
	main()