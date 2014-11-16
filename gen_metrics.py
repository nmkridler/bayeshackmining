import pandas as pd
import numpy as np

import features.inspections as ins
import features.violations as vio

READ_OPTS = {"sep": "|"}
INS_FILE = "../data/Inspections.txt"
OUT_FILE = "../data/inspection_metrics.csv"

def inspections():
	df = pd.read_csv(INS_FILE, **READ_OPTS)
	ins.inspection_metrics(df).to_csv(OUT_FILE, index=False)

def violations():
	i_dt = "INSPECTIONS_BEGIN_DT"
	types = dict([i,object] for i in [0,25,34,36,37,46])
	vf = pd.read_csv("../data/Violations.txt", sep="|", dtype=types, parse_dates=[i_dt])
	vio.get_violation_data(vf).to_csv("../data/violation_metrics.csv", index=False)

def main():
	violations()

if __name__ == "__main__":
	main()