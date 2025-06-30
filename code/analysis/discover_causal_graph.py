import argparse, yaml, logging, pandas as pd, numpy as np, rasterio
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz
from causallearn.utils.GraphUtils import GraphUtils
from code.utils.logging_config import setup_logging
setup_logging(); log = logging.getLogger(__name__)

def raster_to_df(path, n=10000):
    with rasterio.open(path) as src:
        data = src.read().reshape(src.count, -1).T
        mask = ~np.isnan(data).any(axis=1)
        data = data[mask][:n]
    return pd.DataFrame(data, columns=[f"b{i+1}" for i in range(data.shape[1])])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_stack", required=True)
    p.add_argument("--algorithm", choices=["fci"], required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    config = yaml.safe_load(open(args.config))["causal_discovery"]
    df = raster_to_df(args.feature_stack, config["sample_size"])
    gml = GraphUtils.to_gml(fci(df.to_numpy(), fisherz, alpha=config["significance_level"])[0],
                            labels=df.columns.tolist())
    open(args.output_path, "w").write(gml)

if __name__ == "__main__": main()
