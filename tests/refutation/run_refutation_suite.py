import pandas as pd, json, argparse, logging
from dowhy import CausalModel
logging.basicConfig(level=logging.INFO)

def run(path, out):
    df = pd.DataFrame({"Z": [1,2,3,4,5,6], "X": [2,3,4,5,6,7], "Y": [3,4,5,6,7,8]})
    g = "digraph { Z -> X; Z -> Y; X -> Y; }"
    model = CausalModel(data=df, treatment="X", outcome="Y", graph=g)
    estimand = model.identify_effect()
    est = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
    ptest = model.refute_estimate(estimand, est, method_name="placebo_treatment_refuter")
    rcause = model.refute_estimate(estimand, est, method_name="random_common_cause")
    json.dump({
        "initial": est.value,
        "placebo": str(ptest),
        "random_cause": str(rcause)
    }, open(out, "w"), indent=2)

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--graph_path", required=False)
    a.add_argument("--output_report", required=True)
    args = a.parse_args()
    run(args.graph_path, args.output_report)
