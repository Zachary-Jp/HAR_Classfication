"""重画已经完成的结果，不训练、不改变模型选择。"""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="重新生成简洁中文图、混淆矩阵和参数热力图")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    from har_study.plots import draw_all
    paths = draw_all(args.run, args.font)
    if (args.run / "stage_timing_environment.json").is_file():
        from har_study.stage_timing import draw_stage_figures
        paths += draw_stage_figures(args.run, args.font)
    print(f"[Figures] {len(paths)} figures saved in {args.run / 'figures'}")
