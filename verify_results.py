"""复算指标和检查冻结选择。历史结果可用基础检查，新结果使用完整检查。"""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if (args.run / "search_plan.json").is_file():
        from run_experiment import verify_extended
        from har_study.data import prepare_dataset
        result = verify_extended(args.run, prepare_dataset(None, ROOT / "data"))
    else:
        from har_study.verify import verify_run
        result = verify_run(args.run)
    print(result)
    if (args.run / "stage_timing_environment.json").is_file():
        from har_study.stage_timing import verify_stage_timing
        print(verify_stage_timing(args.run))
