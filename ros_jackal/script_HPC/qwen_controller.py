#!/usr/bin/env python3
"""
Checkpoint 自动监控和切换控制器

功能：
1. 监控 checkpoint 目录，发现新训练的 checkpoint
2. 检查当前 checkpoint 是否完成评估（查看 signal 文件）
3. 自动切换到下一个未评估的 checkpoint
4. 提交评估任务
"""

import os
import json
import time
import argparse
import subprocess
import requests
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime


class CheckpointController:
    """自动监控和控制 checkpoint 评估流程"""

    def __init__(
        self,
        checkpoint_dir: str,
        qwen_host: str,
        qwen_port: int = 5000,
        start_world: int = 0,
        end_world: int = 299,
        runs_per_world: int = 1,
        check_interval: int = 60,
        watch_mode: bool = False,
        fixed_algorithm: Optional[str] = None,
        fixed_num_params: Optional[int] = None,
        checkpoint_order: Optional[List[int]] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.qwen_host = qwen_host
        self.qwen_port = qwen_port
        self.start_world = start_world
        self.end_world = end_world
        self.runs_per_world = runs_per_world
        self.check_interval = check_interval
        self.watch_mode = watch_mode  # 持续监控模式
        self.fixed_algorithm = fixed_algorithm.upper() if fixed_algorithm else None
        self.fixed_num_params = fixed_num_params
        self.checkpoint_order = checkpoint_order  # 自定义 checkpoint 顺序

        # 状态跟踪
        self.completed_checkpoints: Set[str] = set()
        self.current_checkpoint_path: Optional[Path] = None
        self.current_job_id: Optional[str] = None

        # 初始化：加载已完成的 checkpoint
        self._load_completed_checkpoints()

    def _load_completed_checkpoints(self):
        """加载所有已完成评估的 checkpoint（有 signal 文件的）"""
        if not self.checkpoint_dir.exists():
            return

        for checkpoint_path in self.checkpoint_dir.iterdir():
            if checkpoint_path.is_dir() and checkpoint_path.name.startswith("checkpoint-"):
                signal_file = checkpoint_path / "evaluation_complete.signal"
                if signal_file.exists():
                    self.completed_checkpoints.add(checkpoint_path.name)

        if self.completed_checkpoints:
            print(f"📋 Found {len(self.completed_checkpoints)} already completed checkpoints:")
            for name in sorted(self.completed_checkpoints):
                print(f"   ✓ {name}")

    def find_all_checkpoints(self) -> List[Path]:
        """查找所有 checkpoint 目录"""
        if not self.checkpoint_dir.exists():
            return []

        # 如果指定了自定义顺序，按指定顺序返回
        if self.checkpoint_order:
            checkpoints = []
            for step in self.checkpoint_order:
                cp_path = self.checkpoint_dir / f"checkpoint-{step}"
                if cp_path.is_dir():
                    checkpoints.append(cp_path)
                else:
                    print(f"⚠ Warning: checkpoint-{step} not found, skipping")
            return checkpoints

        # 默认按数字升序排列
        checkpoints = sorted(
            [d for d in self.checkpoint_dir.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1])
        )
        return checkpoints

    def find_next_checkpoint(self) -> Optional[Path]:
        """找到下一个需要评估的 checkpoint（未完成的）"""
        all_checkpoints = self.find_all_checkpoints()

        for checkpoint_path in all_checkpoints:
            if checkpoint_path.name not in self.completed_checkpoints:
                return checkpoint_path

        return None

    def is_checkpoint_completed(self, checkpoint_path: Path) -> bool:
        """检查 checkpoint 是否完成评估"""
        signal_file = checkpoint_path / "evaluation_complete.signal"
        return signal_file.exists()

    def get_current_qwen_checkpoint(self) -> Optional[str]:
        """获取 Qwen 服务当前加载的 checkpoint"""
        url = f"http://{self.qwen_host}:{self.qwen_port}/health"

        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get('current_checkpoint')
        except Exception as e:
            print(f"⚠ Warning: Failed to get current checkpoint from Qwen: {e}")
            return None

    def get_qwen_algorithm(self) -> Optional[str]:
        """从 Qwen 服务的 /health 获取当前算法名（DWA/TEB/MPPI/DDP）"""
        url = f"http://{self.qwen_host}:{self.qwen_port}/health"

        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            alg = data.get('algorithm')
            if isinstance(alg, str) and alg:
                return alg.upper()
        except Exception as e:
            print(f"⚠ Warning: Failed to get algorithm from Qwen: {e}")
        return None

    def _effective_alg_and_num(self, checkpoint_path: Path) -> (str, int):
        """返回应使用的算法与参数数目：优先使用固定值；否则从服务/路径推断"""
        if self.fixed_algorithm and self.fixed_num_params:
            return self.fixed_algorithm, int(self.fixed_num_params)

        alg = self.get_qwen_algorithm()
        if not alg:
            name = checkpoint_path.as_posix().lower()
            if "mppi" in name:
                alg = "MPPI"
            elif "teb" in name:
                alg = "TEB"
            elif "ddp" in name:
                alg = "DDP"
            else:
                alg = "DWA"

        algo_num = {"DWA": 9, "TEB": 9, "MPPI": 10, "DDP": 8}.get(alg, 9)
        return alg, algo_num

    def _get_other_algorithm_nodes(self, current_alg: str) -> str:
        """获取其他算法的robot test作业正在运行或等待的节点，用于排除"""
        try:
            nodes = set()
            pending_jobs = []

            # 第一次查询：获取所有 qwen_robot_test 作业
            result = subprocess.run(
                ['squeue', '-u', os.environ.get('USER', ''), '-h', '-o', '%i %j %T %N'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=5
            )

            if result.returncode != 0:
                return ""

            # 解析输出，收集 RUNNING 作业的节点和 PENDING 作业的 ID
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue

                job_id, job_name, status, node = parts[0], parts[1], parts[2], parts[3]

                # 只关心 qwen_robot_test 作业
                if 'qwen_robot_test' not in job_name:
                    continue

                if status == 'RUNNING' and node and node != '(None)':
                    nodes.add(node)
                elif status == 'PENDING' or status == 'PD':
                    pending_jobs.append(job_id)

            # 对于 PENDING 作业，等待并重试获取节点（最多等待 20 秒）
            if pending_jobs:
                print(f"  Found {len(pending_jobs)} PENDING qwen_robot_test jobs, waiting for node assignment...")
                for retry in range(10):  # 最多重试 10 次，每次 2 秒
                    time.sleep(2)

                    for job_id in pending_jobs[:]:  # 使用切片复制，允许修改原列表
                        result = subprocess.run(
                            ['squeue', '-j', job_id, '-h', '-o', '%T %N'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True,
                            timeout=5
                        )

                        if result.returncode == 0:
                            parts = result.stdout.strip().split()
                            if len(parts) >= 2:
                                status, node = parts[0], parts[1]
                                if status == 'RUNNING' and node and node != '(None)':
                                    nodes.add(node)
                                    pending_jobs.remove(job_id)
                                    print(f"    Job {job_id} assigned to node: {node}")

                    # 如果所有 PENDING 作业都已分配节点，提前退出
                    if not pending_jobs:
                        break

                if pending_jobs:
                    print(f"  Warning: {len(pending_jobs)} jobs still pending after 20s, may cause node conflicts")

            # 返回逗号分隔的节点列表
            return ','.join(sorted(nodes)) if nodes else ""

        except Exception as e:
            print(f"  Warning: Failed to get other algorithm nodes: {e}")
            return ""

    def switch_checkpoint(self, checkpoint_path: Path) -> bool:
        """通过 API 切换 checkpoint"""
        url = f"http://{self.qwen_host}:{self.qwen_port}/switch_checkpoint"
        # 使用固定算法/参数数目（若提供），否则从服务/路径推断
        alg, algo_num = self._effective_alg_and_num(checkpoint_path)

        payload = {
            "checkpoint_path": str(checkpoint_path),
            "algorithm": alg,
            "num_params": algo_num,
        }

        print(f"\n{'='*60}")
        print(f"🔄 Switching to: {checkpoint_path.name}")
        print(f"{'='*60}")

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()

            print(f"✓ Checkpoint switched successfully")
            print(f"  New checkpoint: {checkpoint_path}")
            self.current_checkpoint_path = checkpoint_path
            return True

        except Exception as e:
            print(f"❌ Failed to switch checkpoint: {e}")
            return False

    def submit_evaluation_job(self, checkpoint_path: Path) -> Optional[str]:
        """提交评估任务到 SLURM"""
        # run_hopper_qwen.slurm 在 script_HPC 目录下
        script_path = Path(__file__).parent / "run_hopper_qwen.slurm"

        if not script_path.exists():
            print(f"❌ Evaluation script not found: {script_path}")
            return None

        # 规范化 policy 名称：<algorithm>_qwen（例如 dwa_qwen）
        alg, algo_num = self._effective_alg_and_num(checkpoint_path)
        policy_name = f"{alg.lower()}_qwen"

        # 准备环境变量
        env = os.environ.copy()
        env.update({
            'QWEN_HOST': self.qwen_host,
            'QWEN_PORT': str(self.qwen_port),
            'START_WORLD': str(self.start_world),
            'END_WORLD': str(self.end_world),
            'RUNS_PER_WORLD': str(self.runs_per_world),
            'POLICY_NAME': policy_name,
            'ALGORITHM': alg,
            'NUM_PARAMS': str(algo_num),
        })

        print(f"\n🚀 Submitting evaluation job")
        print(f"  Checkpoint: {checkpoint_path.name}")
        print(f"  Algorithm: {alg}")
        print(f"  Policy: {policy_name}")
        print(f"  Worlds: {self.start_world} - {self.end_world}")

        # 🔧 添加基于算法的固定延迟，确保按顺序提交，避免节点冲突
        # DWA: 0s, TEB: 30s, MPPI: 60s, DDP: 90s（足够 SLURM 分配节点）
        algorithm_delays = {
            'DWA': 0,
            'TEB': 30,
            'MPPI': 60,
            'DDP': 90
        }
        delay = algorithm_delays.get(alg, 0)
        if delay > 0:
            print(f"  Waiting {delay}s (algorithm-based delay to avoid node conflicts)...")
            time.sleep(delay)

        # 🔧 获取其他算法当前占用的节点，避免冲突
        exclude_nodes = self._get_other_algorithm_nodes(alg)
        if exclude_nodes:
            print(f"  Excluding nodes: {exclude_nodes}")

        # 可选：允许通过环境变量覆盖部分 sbatch 资源（不设置则使用 slurm 脚本中的 #SBATCH 默认）
        # 支持变量：EVAL_PARTITION, EVAL_QOS, EVAL_CPUS, EVAL_MEM, EVAL_MEM_PER_CPU, EVAL_TIME
        sbatch_cmd = ['sbatch']

        # 添加节点排除策略
        if exclude_nodes:
            sbatch_cmd += [f'--exclude={exclude_nodes}']
        if os.environ.get('EVAL_PARTITION'):
            sbatch_cmd += ['-p', os.environ['EVAL_PARTITION']]
        if os.environ.get('EVAL_QOS'):
            sbatch_cmd += ['--qos', os.environ['EVAL_QOS']]
        if os.environ.get('EVAL_CPUS'):
            sbatch_cmd += [f"--cpus-per-task={os.environ['EVAL_CPUS']}"]
        # 互斥选择总体内存或每CPU内存
        if os.environ.get('EVAL_MEM'):
            sbatch_cmd += [f"--mem={os.environ['EVAL_MEM']}"]
        elif os.environ.get('EVAL_MEM_PER_CPU'):
            sbatch_cmd += [f"--mem-per-cpu={os.environ['EVAL_MEM_PER_CPU']}"]
        if os.environ.get('EVAL_TIME'):
            sbatch_cmd += [f"--time={os.environ['EVAL_TIME']}"]

        # 最后附上脚本路径
        sbatch_cmd.append(str(script_path))

        try:
            result = subprocess.run(
                sbatch_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=True
            )

            output = result.stdout.strip()
            job_id = output.split()[-1] if output else None

            print(f"✓ Job submitted: {job_id}")

            # 🔧 等待并验证节点分配
            if job_id:
                time.sleep(3)  # 等待SLURM分配节点
                node_result = subprocess.run(
                    ['squeue', '-j', job_id, '-h', '-o', '%N'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    timeout=5
                )
                if node_result.returncode == 0:
                    assigned_node = node_result.stdout.strip()
                    if assigned_node and assigned_node != '(None)':
                        print(f"  Assigned to node: {assigned_node}")
                        if exclude_nodes and assigned_node in exclude_nodes:
                            print(f"  ⚠️ WARNING: Job assigned to excluded node {assigned_node}!")

            self.current_job_id = job_id
            return job_id

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to submit job: {e}")
            print(f"  stderr: {e.stderr}")
            return None

    def wait_for_completion(self, checkpoint_path: Path, timeout: int = 86400) -> bool:
        """等待当前 checkpoint 评估完成"""
        print(f"\n⏳ Waiting for evaluation to complete...")
        print(f"  Checkpoint: {checkpoint_path.name}")
        print(f"  Checking every {self.check_interval} seconds")

        start_time = time.time()
        check_count = 0

        while time.time() - start_time < timeout:
            check_count += 1

            # 检查 signal 文件
            if self.is_checkpoint_completed(checkpoint_path):
                print(f"\n✓ Evaluation completed for {checkpoint_path.name}!")
                self.completed_checkpoints.add(checkpoint_path.name)
                return True

            elapsed = time.time() - start_time
            print(f"  [{elapsed/60:.1f}min] Check #{check_count}: No signal yet...", end='\r')
            time.sleep(self.check_interval)

        print(f"\n❌ Timeout waiting for {checkpoint_path.name}")
        return False

    def process_one_checkpoint(self) -> bool:
        """处理一个 checkpoint 的完整流程，返回是否成功"""
        # 0) 若服务当前已加载本目录下的某个 checkpoint，且尚未完成评估，则视为“当前进行中”，不切换，直接等待完成
        current_loaded_path = self.get_current_qwen_checkpoint()
        try:
            current_path = Path(current_loaded_path) if current_loaded_path else None
        except Exception:
            current_path = None

        if current_path and current_path.exists() and current_path.parent == self.checkpoint_dir:
            if not self.is_checkpoint_completed(current_path):
                print(f"\n▶ Current in-service checkpoint: {current_path.name} (no signal yet). Will not switch.")

                # ✅ 自动提交评估任务
                print(f"🚀 Auto-submitting evaluation for initial checkpoint: {current_path.name}")
                job_id = self.submit_evaluation_job(current_path)
                if job_id:
                    print(f"✓ Evaluation job submitted: {job_id}")
                else:
                    print(f"⚠️ Failed to submit job, will wait anyway (job may already be running)")

                time.sleep(10)  # 等待任务启动

                # 等待完成
                if not self.wait_for_completion(current_path, timeout=86400):
                    print(f"❌ Evaluation timeout or failed for {current_path.name}")
                    return True
                # 完成后继续处理后续
                time.sleep(5)
                return True

        # 1) 查找下一个需要评估的 checkpoint（基于 signal 文件）
        next_checkpoint = self.find_next_checkpoint()

        if not next_checkpoint:
            print("\n✓ All checkpoints have been evaluated!")
            return False  # 没有更多任务

        print(f"\n{'#'*60}")
        print(f"# Next checkpoint: {next_checkpoint.name}")
        print(f"{'#'*60}")

        # 2) 如有必要，切换到目标 checkpoint（避免对同一 checkpoint 重复卸载/加载）
        current_loaded = self.get_current_qwen_checkpoint()
        try:
            current_name = Path(current_loaded).name if current_loaded else None
        except Exception:
            current_name = None

        if current_name == next_checkpoint.name:
            print(f"🔁 Target checkpoint already loaded: {current_name} (skip switching)")
        else:
            if not self.switch_checkpoint(next_checkpoint):
                print(f"❌ Failed to switch, skipping...")
                return True  # 继续尝试其他 checkpoint

        time.sleep(5)

        # 3) 提交评估任务（若已由外部提交，可重复提交保护在脚本层处理；此处按一次提交逻辑）
        job_id = self.submit_evaluation_job(next_checkpoint)
        if not job_id:
            print(f"❌ Failed to submit job, skipping...")
            return True

        time.sleep(10)

        # 4) 等待完成
        if not self.wait_for_completion(next_checkpoint, timeout=86400):
            print(f"❌ Evaluation timeout or failed")
            # 不标记为完成，下次可以重试
            return True

        # 5) 短暂休息后继续
        time.sleep(30)
        return True  # 继续处理下一个

    def run_batch_mode(self):
        """批量模式：处理所有 checkpoint 后退出"""
        print(f"\n{'='*60}")
        print(f"📊 Batch Mode: Evaluate All Checkpoints")
        print(f"{'='*60}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"Qwen service: {self.qwen_host}:{self.qwen_port}")
        print(f"World range: {self.start_world} - {self.end_world}")
        print(f"{'='*60}\n")

        while self.process_one_checkpoint():
            pass  # 持续处理直到没有更多 checkpoint

        self.print_summary()

    def run_watch_mode(self):
        """监控模式：持续监控新 checkpoint 并自动评估"""
        print(f"\n{'='*60}")
        print(f"👁️  Watch Mode: Continuous Monitoring")
        print(f"{'='*60}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"Qwen service: {self.qwen_host}:{self.qwen_port}")
        print(f"World range: {self.start_world} - {self.end_world}")
        print(f"Check interval: {self.check_interval}s")
        print(f"{'='*60}")
        print(f"\n🔍 Watching for new checkpoints...")
        print(f"   (Press Ctrl+C to stop)\n")

        try:
            while True:
                # 查找新 checkpoint
                all_checkpoints = self.find_all_checkpoints()
                pending = [cp for cp in all_checkpoints if cp.name not in self.completed_checkpoints]

                if pending:
                    print(f"\n🆕 Found {len(pending)} pending checkpoint(s):")
                    for cp in pending:
                        print(f"   - {cp.name}")

                    # 处理一个 checkpoint
                    self.process_one_checkpoint()
                else:
                    # 没有新 checkpoint，等待
                    now = datetime.now().strftime("%H:%M:%S")
                    print(f"[{now}] No new checkpoints. Waiting...", end='\r')
                    time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print(f"\n\n⏹️  Watch mode stopped by user")
            self.print_summary()

    def print_summary(self):
        """打印评估总结"""
        all_checkpoints = self.find_all_checkpoints()
        pending = [cp.name for cp in all_checkpoints if cp.name not in self.completed_checkpoints]

        print(f"\n{'='*60}")
        print(f"📊 Summary")
        print(f"{'='*60}")
        print(f"✓ Completed: {len(self.completed_checkpoints)}")
        for name in sorted(self.completed_checkpoints):
            print(f"    - {name}")

        if pending:
            print(f"\n⏸️  Pending: {len(pending)}")
            for name in sorted(pending):
                print(f"    - {name}")

        print(f"\n{'='*60}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="自动监控和控制 checkpoint 评估流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  1. 批量模式（处理所有现有 checkpoint 后退出）:
     python checkpoint_controller.py \\
         --checkpoint_dir /path/to/checkpoints \\
         --qwen_host gpu011

  2. 监控模式（持续监控新 checkpoint）:
     python checkpoint_controller.py \\
         --checkpoint_dir /path/to/checkpoints \\
         --qwen_host gpu011 \\
         --watch

  3. 只评估部分环境:
     python checkpoint_controller.py \\
         --checkpoint_dir /path/to/checkpoints \\
         --qwen_host gpu011 \\
         --start_world 0 \\
         --end_world 49 \\
         --watch
        """
    )

    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        required=True,
        help='Checkpoint 目录路径（包含多个 checkpoint-* 子目录）'
    )
    parser.add_argument(
        '--qwen_host',
        type=str,
        required=True,
        help='Qwen 服务所在的 GPU 节点（如 gpu011）'
    )
    parser.add_argument(
        '--qwen_port',
        type=int,
        default=5000,
        help='Qwen 服务端口（默认 5000）'
    )
    parser.add_argument(
        '--start_world',
        type=int,
        default=0,
        help='起始环境编号（默认 0）'
    )
    parser.add_argument(
        '--end_world',
        type=int,
        default=299,
        help='结束环境编号（默认 299）'
    )
    parser.add_argument(
        '--runs_per_world',
        type=int,
        default=1,
        help='每个环境运行次数（默认 1）'
    )
    parser.add_argument(
        '--check_interval',
        type=int,
        default=60,
        help='检查间隔（秒，默认 60）'
    )
    parser.add_argument(
        '--watch',
        action='store_true',
        help='启用监控模式（持续监控新 checkpoint）'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default=os.environ.get('ALGORITHM', None),
        help='固定算法名称（DWA/TEB/MPPI/DDP）。若提供，则控制器不会更改算法，仅切换同目录下的 checkpoint'
    )
    parser.add_argument(
        '--num_params',
        type=int,
        default=int(os.environ['NUM_PARAMS']) if os.environ.get('NUM_PARAMS') else None,
        help='固定参数数量（DWA=9, TEB=9, MPPI=10, DDP=8）'
    )
    parser.add_argument(
        '--checkpoint_order',
        type=str,
        default=None,
        help='自定义 checkpoint 顺序，逗号分隔（如 17500,15000,12500）。只评估指定的 checkpoint'
    )

    args = parser.parse_args()

    # 解析 checkpoint_order
    checkpoint_order = None
    if args.checkpoint_order:
        try:
            checkpoint_order = [int(x.strip()) for x in args.checkpoint_order.split(',')]
            print(f"📋 Custom checkpoint order: {checkpoint_order}")
        except ValueError as e:
            print(f"❌ Invalid checkpoint_order format: {e}")
            exit(1)

    controller = CheckpointController(
        checkpoint_dir=args.checkpoint_dir,
        qwen_host=args.qwen_host,
        qwen_port=args.qwen_port,
        start_world=args.start_world,
        end_world=args.end_world,
        runs_per_world=args.runs_per_world,
        check_interval=args.check_interval,
        watch_mode=args.watch,
        fixed_algorithm=args.algorithm,
        fixed_num_params=args.num_params,
        checkpoint_order=checkpoint_order,
    )

    if args.watch:
        controller.run_watch_mode()
    else:
        controller.run_batch_mode()


if __name__ == '__main__':
    main()
