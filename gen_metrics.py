import pandas as pd
import numpy as np

import features.inspections as ins

READ_OPTS = {"sep": "|"}
INS_FILE = "../data/Inspections.txt"
OUT_FILE = "../data/inspection_metrics.csv"

def main():
	df = pd.read_csv(INS_FILE, **READ_OPTS)
	ins.inspection_metrics(df).to_csv(OUT_FILE, index=False)

if __name__ == "__main__":
	main()