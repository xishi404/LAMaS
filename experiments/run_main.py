#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lamas.configs.models_config import ModelsConfig
from lamas.ext.lamas.benchmark.experiment_configs import EXPERIMENT_CONFIGS
from lamas.ext.lamas.scripts.optimizer import Optimizer
from lamas.logs import logger, define_log_level


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Runner")
    parser.add_argument("--dataset", type=str, choices=list(EXPERIMENT_CONFIGS.keys()), required=True)
    parser.add_argument("--sample", type=int, default=4)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--opt_model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--exec_model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--token_weights", type=str, default="0.000001,0.00001,0.0001")
    parser.add_argument("--use_token_penalty", action="store_true")
    parser.add_argument("--virtual_token_rate", type=float, default=50.0, help="Virtual tokens per second (conversion rate from tool execution time to tokens)")
    parser.add_argument("--optimized_path", type=str, default="lamas/ext/lamas/scripts/optimized")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--no_parallel", action="store_true", help="Disable parallel execution within layers")
    parser.add_argument("--no_critical_path", action="store_true", help="Disable critical path credit assignment")
    parser.add_argument("--normalize_rewards", action="store_true")
    return parser.parse_args()


def run_experiment(config, opt_llm_config, exec_llm_config,
                   token_weight, use_tokens, experiment_name, args):
    """Run a single experiment with given parameters"""

    # Log experiment start with config
    logger.info(f"[START] Experiment: {experiment_name}")
    logger.info(f"  Dataset: {config.dataset}, Token weight: {token_weight}, Parallel: {not args.no_parallel}, Critical path: {not args.no_critical_path}, Normalize: {args.normalize_rewards}")

    code_base_path = args.optimized_path
    experiment_results_path = os.path.join(args.optimized_path, "experiments", experiment_name)

    optimizer = Optimizer(
        dataset=config.dataset,
        question_type=config.question_type,
        opt_llm_config=opt_llm_config,
        exec_llm_config=exec_llm_config,
        operators=config.operators,
        optimized_path=experiment_results_path,
        code_path=code_base_path,
        sample=args.sample,
        round=args.round,
        batch_size=args.batch_size,
        lr=args.lr,
        is_textgrad=False,
        token_weight=token_weight,
        use_tokens=use_tokens,
        virtual_token_rate=args.virtual_token_rate,
        use_critical_path=not args.no_critical_path,
        parallel_execution=not args.no_parallel,
        normalize_rewards=args.normalize_rewards,
    )

    train_score = None
    train_cost = None
    train_wall_clock_time = None
    train_prompt_tokens = None
    train_completion_tokens = None
    train_total_tokens = None

    if not args.test_only:
        try:
            logger.info("[TRAIN] Starting training...")
            import time
            train_start_time = time.time()
            optimizer.optimize("Graph")
            train_wall_clock_time = time.time() - train_start_time

            train_results_path = os.path.join(optimizer.root_path, "train", "results.json")
            if os.path.exists(train_results_path):
                with open(train_results_path, "r") as f:
                    results_data = json.load(f)
                    if results_data:
                        last_result = results_data[-1]
                        train_score = last_result.get("score", None)
                        train_cost = last_result.get("avg_cost", None)

            if hasattr(optimizer, 'benchmark'):
                benchmark = optimizer.benchmark
                train_prompt_tokens = getattr(benchmark, 'total_prompt_tokens', None)
                train_completion_tokens = getattr(benchmark, 'total_completion_tokens', None)
                if train_prompt_tokens is not None and train_completion_tokens is not None:
                    train_total_tokens = train_prompt_tokens + train_completion_tokens

            logger.info(f"[TRAIN] Completed in {train_wall_clock_time:.2f}s, Score: {f'{train_score:.5f}' if train_score else 'N/A'}")

        except Exception as train_error:
            logger.error(f"[TRAIN] Failed: {train_error}")
            if not args.test_only:
                raise

    logger.info("[TEST] Starting testing...")
    import asyncio
    import glob
    import pandas as pd

    num_test_runs = 1
    test_scores = []
    test_costs = []
    test_avg_cp_tokens = []
    test_p90_cp_tokens = []
    test_total_cp_tokens = []
    test_prompt_tokens_list = []
    test_completion_tokens_list = []

    for test_run in range(1, num_test_runs + 1):
        try:
            test_score = asyncio.run(optimizer.test())

            test_results_path = os.path.join(optimizer.root_path, "test", "results.json")
            test_cost = None
            if os.path.exists(test_results_path):
                with open(test_results_path, "r") as f:
                    test_results_data = json.load(f)
                    if test_results_data:
                        last_result = test_results_data[-1]
                        test_score = last_result.get("score", test_score)
                        test_cost = last_result.get("avg_cost", None)

            test_round_path = os.path.join(optimizer.root_path, "test", f"round_{args.round}")
            test_avg_cp_token = None
            test_p90_cp_token = None
            test_total_cp_token = None

            if os.path.exists(test_round_path):
                csv_files = glob.glob(os.path.join(test_round_path, "*.csv"))
                if csv_files:
                    if use_tokens:
                        penalty_suffix = f"_tok{token_weight:.4f}".replace(".", "_")
                    else:
                        penalty_suffix = "_0000"

                    matching_csv_files = [f for f in csv_files if penalty_suffix in os.path.basename(f)]
                    if matching_csv_files:
                        latest_csv = max(matching_csv_files, key=os.path.getctime)
                        try:
                            df = pd.read_csv(latest_csv)
                            if "cp_token" in df.columns:
                                test_avg_cp_token = df["cp_token"].mean()
                                test_p90_cp_token = df["cp_token"].quantile(0.9)
                                test_total_cp_token = df["cp_token"].sum()
                        except Exception:
                            pass

            test_prompt_tokens = None
            test_completion_tokens = None
            if hasattr(optimizer, 'test_benchmark'):
                test_benchmark = optimizer.test_benchmark
                test_prompt_tokens = getattr(test_benchmark, 'test_prompt_tokens', None)
                test_completion_tokens = getattr(test_benchmark, 'test_completion_tokens', None)
                test_total_cost = getattr(test_benchmark, 'test_total_cost', None)
                if test_total_cost is not None:
                    test_cost = test_total_cost

            if test_score is not None:
                test_scores.append(test_score)
            if test_cost is not None:
                test_costs.append(test_cost)
            if test_avg_cp_token is not None:
                test_avg_cp_tokens.append(test_avg_cp_token)
            if test_p90_cp_token is not None:
                test_p90_cp_tokens.append(test_p90_cp_token)
            if test_total_cp_token is not None:
                test_total_cp_tokens.append(test_total_cp_token)
            if test_prompt_tokens is not None:
                test_prompt_tokens_list.append(test_prompt_tokens)
            if test_completion_tokens is not None:
                test_completion_tokens_list.append(test_completion_tokens)

        except Exception as test_error:
            logger.error(f"[TEST] Run {test_run} failed: {test_error}")

    test_score = sum(test_scores) / len(test_scores) if test_scores else None
    test_cost = sum(test_costs) / len(test_costs) if test_costs else None
    test_avg_cp_token = sum(test_avg_cp_tokens) / len(test_avg_cp_tokens) if test_avg_cp_tokens else None
    test_p90_cp_token = sum(test_p90_cp_tokens) / len(test_p90_cp_tokens) if test_p90_cp_tokens else None
    test_total_cp_token = sum(test_total_cp_tokens) / len(test_total_cp_tokens) if test_total_cp_tokens else None
    test_prompt_tokens = sum(test_prompt_tokens_list) / len(test_prompt_tokens_list) if test_prompt_tokens_list else None
    test_completion_tokens = sum(test_completion_tokens_list) / len(test_completion_tokens_list) if test_completion_tokens_list else None
    test_total_tokens = test_prompt_tokens + test_completion_tokens if (test_prompt_tokens is not None and test_completion_tokens is not None) else None

    # Log final metrics
    logger.info(f"[RESULT] Score: {f'{test_score:.5f}' if test_score else 'N/A'}, Avg CP Token: {test_avg_cp_token:.2f}" if test_avg_cp_token else f"[RESULT] Score: {f'{test_score:.5f}' if test_score else 'N/A'}")
    logger.info(f"[END] Experiment: {experiment_name}")

    try:
        return {
            "experiment_name": experiment_name,
            "dataset": config.dataset,
            "token_weight": token_weight,
            "use_tokens": use_tokens,
            "parallel_execution": not args.no_parallel,
            "use_critical_path": not args.no_critical_path,
            "normalize_rewards": args.normalize_rewards,
            "training": {
                "score": float(train_score) if train_score is not None else None,
                "avg_cost": float(train_cost) if train_cost is not None else None,
                "wall_clock_time_seconds": float(train_wall_clock_time) if train_wall_clock_time is not None else None,
                "prompt_tokens": int(train_prompt_tokens) if train_prompt_tokens is not None else None,
                "completion_tokens": int(train_completion_tokens) if train_completion_tokens is not None else None,
                "total_tokens": int(train_total_tokens) if train_total_tokens is not None else None,
            },
            "testing": {
                "score": float(test_score) if test_score is not None else None,
                "cost": float(test_cost) if test_cost is not None else None,
                "avg_cp_token": float(test_avg_cp_token) if test_avg_cp_token is not None else None,
                "p90_cp_token": float(test_p90_cp_token) if test_p90_cp_token is not None else None,
                "total_cp_token": float(test_total_cp_token) if test_total_cp_token is not None else None,
                "prompt_tokens": int(test_prompt_tokens) if test_prompt_tokens is not None else None,
                "completion_tokens": int(test_completion_tokens) if test_completion_tokens is not None else None,
                "total_tokens": int(test_total_tokens) if test_total_tokens is not None else None,
            },
            "log_path": optimizer.root_path,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Experiment {experiment_name} failed: {e}")
        return {
            "experiment_name": experiment_name,
            "dataset": config.dataset,
            "token_weight": token_weight,
            "use_tokens": use_tokens,
            "parallel_execution": not args.no_parallel,
            "use_critical_path": not args.no_critical_path,
            "normalize_rewards": args.normalize_rewards,
            "training": {"score": None, "avg_cost": None, "wall_clock_time_seconds": None,
                        "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            "testing": {"score": None, "cost": None, "avg_cp_token": None, "p90_cp_token": None,
                       "total_cp_token": None, "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            "log_path": optimizer.root_path if 'optimizer' in locals() else experiment_results_path,
            "status": "failed",
            "error": str(e)
        }


def main():
    args = parse_args()

    mode_suffix = "sequential" if args.no_parallel else "parallel"
    cp_suffix = "no_cp" if args.no_critical_path else "cp"
    norm_suffix = "norm" if args.normalize_rewards else "no_norm"
    flags = f"{mode_suffix}_{cp_suffix}_{norm_suffix}"

    token_weights = [float(w.strip()) for w in args.token_weights.split(",")]
    primary_weight = token_weights[0] if token_weights else None

    define_log_level(print_level="INFO", logfile_level="DEBUG", dataset=args.dataset, flags=flags, weight=primary_weight)

    config = EXPERIMENT_CONFIGS[args.dataset]
    models_config = ModelsConfig.default()
    opt_llm_config = models_config.get(args.opt_model_name)
    exec_llm_config = models_config.get(args.exec_model_name)

    if opt_llm_config is None or exec_llm_config is None:
        raise ValueError("Model config not found")

    experiments = []
    if args.use_token_penalty:
        for weight in token_weights:
            experiments.append({"token_weight": weight, "use_tokens": True,
                               "name": f"token_weight_{weight:.6f}".replace(".", "_")})
    else:
        experiments.append({"token_weight": 0.0, "use_tokens": False, "name": "baseline"})

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_base_dir = os.path.join(args.optimized_path, args.dataset, "train", f"experiments_{timestamp}")
    os.makedirs(experiment_base_dir, exist_ok=True)

    for exp_config in experiments:
        result = run_experiment(config, opt_llm_config, exec_llm_config,
                               exp_config.get("token_weight", 0.0),
                               exp_config.get("use_tokens", False), exp_config["name"], args)
        results.append(result)

    summary_file = os.path.join(experiment_base_dir, "experiment_summary.json")
    with open(summary_file, "w") as f:
        json.dump({"timestamp": timestamp, "dataset": args.dataset, "experiments": results,
                  "config": {"sample": args.sample, "round": args.round, "batch_size": args.batch_size,
                            "lr": args.lr, "weights_tested": token_weights if args.use_token_penalty else [0.0]}}, f, indent=2)

    logger.info(f"Results saved to: {summary_file}")


if __name__ == "__main__":
    main()
