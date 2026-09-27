import argparse
from .pipeline import run_demo

def main():
    p=argparse.ArgumentParser(description="Glucose digital twin research pipeline")
    p.add_argument("--demo",action="store_true",help="run the dataset-free end-to-end pipeline")
    p.add_argument("--output",default="artifacts/demo")
    args=p.parse_args()
    if not args.demo:
        p.error("The materialized pipeline currently exposes the deterministic --demo path; clinical datasets are supplied separately.")
    result=run_demo(args.output)
    print(f"held_out_patient={result.held_out_patient}")
    print(result.metrics)

if __name__=="__main__":
    main()
