import argparse
from experiments.baseline import run_baseline_experiment
from experiments.crformer_demo import run_crformer_experiment
from utils.visualization import plot_comparison


def main():
    parser = argparse.ArgumentParser(description='WRSN级联失效仿真')
    parser.add_argument('--mode', choices=['baseline', 'crformer', 'compare'],
                        default='compare', help='运行模式')

    args = parser.parse_args()

    if args.mode == 'baseline':
        run_baseline_experiment()

    elif args.mode == 'crformer':
        run_crformer_experiment()

    elif args.mode == 'compare':
        print("运行对比实验".center(50, "="))

        # 运行基线
        baseline_result = run_baseline_experiment()

        # 运行CR-Former
        crformer_result = run_crformer_experiment()

        # 显示对比结果
        print("\n" + "对比结果".center(50, "="))
        print(f"基线寿命: {baseline_result['lifetime']} 步")
        print(f"CR-Former寿命: {crformer_result['lifetime']} 步")
        improvement = (crformer_result['lifetime'] - baseline_result['lifetime']) / baseline_result['lifetime'] * 100
        print(f"性能提升: {improvement:.1f}%")

        # 绘制对比图
        plot_comparison(baseline_result, crformer_result)


if __name__ == "__main__":
    main()