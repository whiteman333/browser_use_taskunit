import prompt
import os
import json
import ast
import asyncio
import time
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
import pandas as pd


async def process_cluster_dag_async(
    cluster_id,
    trajectories_in_cluster,
    task_units,
    output_dir,
    mode="formal",
):
    """
    处理单个聚类的DAG生成(异步版本)

    参数:
        cluster_id: 聚类ID
        trajectories_in_cluster: 该聚类中的轨迹数据
        task_units: task_unit数据
        output_dir: 输出目录 (由调用者提供)
        mode: 输出模式（formal为正式模式，其他值为调试模式）
    """
    print(f"正在处理聚类: {cluster_id}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        print("错误：未提供有效的输出目录。")
        return cluster_id, False

    try:
        if not trajectories_in_cluster:
            print(f"⚠️ 聚类 {cluster_id} 没有有效的轨迹数据")
            return cluster_id, False

        if not hasattr(prompt, "taskunit2DAG"):
            print(f"错误：'prompt' 模块中未找到 'taskunit2DAG' 函数。")
            return cluster_id, False

        success = prompt.taskunit2DAG(
            trajectories_in_cluster,
            task_units,
            f"cluster_{cluster_id}",  # 使用聚类ID作为标识
            output_dir,
            mode,
        )
        return cluster_id, success

    except Exception as e:
        print(f"处理聚类 {cluster_id} 时出错: {str(e)}")
        if output_dir:
            try:
                with open(
                    os.path.join(output_dir, "dag_errors.log"), "a", encoding="utf-8"
                ) as f:
                    f.write(f"cluster_{cluster_id}: {str(e)}\n")
            except Exception as log_e:
                print(f"写入错误日志失败: {log_e}")
        return cluster_id, False


async def main_cluster_async(
    clusters_by_distance,
    df,
    taskunit_dir: str,
    output_dir: str,
    max_concurrency=4,
    mode="formal",
):
    """
    基于聚类结果的主函数(异步版本)

    参数:
        clusters_by_distance: 层次聚类的结果 (假设是一个列表或类似数组，索引对应df的行)
        df: 原始数据DataFrame (假设索引与clusters_by_distance的索引对齐)
        taskunit_dir: Task unit 文件所在的目录
        output_dir: 图输出目录
        max_concurrency: 最大并发数
        mode: 输出模式
    """
    os.makedirs(output_dir, exist_ok=True)

    if clusters_by_distance is None:
        print("错误：clusters_by_distance 为 None。")
        return
    if df is None:
        print("错误：df 为 None。")
        return

    if len(clusters_by_distance) > len(df):
        print(
            f"错误：聚类结果数量 ({len(clusters_by_distance)}) 大于 DataFrame 行数 ({len(df)})。无法安全地使用 iloc 索引。"
        )
        return

    print(f"根据 {len(clusters_by_distance)} 个聚类标签准备轨迹数据...")
    cluster_trajectories = {}
    for idx, cluster_id in enumerate(clusters_by_distance):
        try:
            cluster_id_key = int(cluster_id)
        except (ValueError, TypeError):
            print(
                f"警告：聚类ID '{cluster_id}' 无法转换为整数，将按原样使用或可能导致问题。"
            )
            cluster_id_key = cluster_id

        if cluster_id_key not in cluster_trajectories:
            cluster_trajectories[cluster_id_key] = []

        try:
            row = df.iloc[idx]
            action_reprs_raw = row.get("action_reprs")
            actions = []
            if isinstance(action_reprs_raw, str):
                try:
                    actions = ast.literal_eval(action_reprs_raw)
                except (ValueError, SyntaxError, TypeError) as eval_err:
                    print(
                        f"警告：解析 action_reprs 时出错 (DF index {idx}, cluster {cluster_id_key}): {eval_err}. 使用空列表。"
                    )
                    actions = []
            elif isinstance(action_reprs_raw, list):
                actions = action_reprs_raw
            else:
                print(
                    f"警告：未知的 action_reprs 类型 (DF index {idx}, cluster {cluster_id_key}): {type(action_reprs_raw)}. 使用空列表。"
                )
                actions = []

            trajectory = {
                "domain": row.get("domain", "N/A"),
                "subdomain": row.get("subdomain", "N/A"),
                "task": row.get("confirmed_task", "N/A"),
                "action_reprs": actions,
            }
            cluster_trajectories[cluster_id_key].append(trajectory)
        except IndexError:
            print(f"处理轨迹时出错：索引 {idx} 超出 DataFrame 范围。")
            continue
        except Exception as e:
            print(f"处理轨迹 idx={idx}, cluster={cluster_id_key} 时出错: {str(e)}")
            continue

    print(f"为 {len(cluster_trajectories)} 个唯一聚类准备图生成任务...")
    tasks = []
    skipped_clusters_no_traj = 0
    skipped_clusters_no_tu = 0
    for cluster_id, trajectories in cluster_trajectories.items():
        task_unit_filename = f"cluster_{cluster_id}_task_unit.json"
        task_unit_path = os.path.join(taskunit_dir, task_unit_filename)

        cluster_task_units = []
        if os.path.exists(task_unit_path):
            try:
                with open(task_unit_path, "r", encoding="utf-8") as f:
                    task_unit_data = json.load(f)
                    cluster_task_units = task_unit_data.get("task_units", [])
            except Exception as e:
                print(f"读取或解析 task unit 文件 {task_unit_path} 时出错: {e}")
                cluster_task_units = []

        if not trajectories:
            print(f"ℹ️ 聚类 {cluster_id} 没有有效的轨迹数据，跳过图生成。")
            skipped_clusters_no_traj += 1
            continue
        if not cluster_task_units:
            print(
                f"ℹ️ 聚类 {cluster_id} 没有 Task Units 数据 ({task_unit_path}) 或读取失败，跳过图生成。"
            )
            skipped_clusters_no_tu += 1
            continue

        tasks.append((cluster_id, trajectories, cluster_task_units, output_dir, mode))

    if skipped_clusters_no_traj > 0:
        print(f"总计 {skipped_clusters_no_traj} 个聚类因缺少轨迹数据被跳过。")
    if skipped_clusters_no_tu > 0:
        print(f"总计 {skipped_clusters_no_tu} 个聚类因缺少Task Unit数据被跳过。")

    if not tasks:
        print("没有准备任何有效的图生成任务。")
        return

    print(f"开始执行 {len(tasks)} 个图生成任务...")
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_with_semaphore(args):
        async with semaphore:
            return await process_cluster_dag_async(*args)

    start_time = time.time()
    coroutines = [process_with_semaphore(task) for task in tasks]
    results = await tqdm_asyncio.gather(*coroutines)

    success_count = sum(1 for _, success in results if success)
    elapsed_time = time.time() - start_time

    print(f"处理完成: 成功 {success_count}/{len(tasks)} 个聚类")
    print(f"总耗时: {elapsed_time:.2f}秒")

    failed_clusters = [
        str(cluster_id) for cluster_id, success in results if not success
    ]
    if failed_clusters:
        print(f"处理失败的聚类: {', '.join(failed_clusters)}")


async def run_cluster_async_main(
    clusters_by_distance, df, max_concurrency=12, mode="formal"
):
    """
    运行基于聚类的异步主函数（适配Jupyter环境）
    这里定义默认路径作为统一接口。

    参数:
        clusters_by_distance: 层次聚类的结果
        df: 原始数据DataFrame
        max_concurrency: 最大并发数
        mode: 输出模式
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_data_dir = os.path.join(script_dir, "data")
        default_output_dir = os.path.join(
            base_data_dir, "cluster_graph_outputs", "deepseek"
        )
        default_taskunit_dir = os.path.join(
            base_data_dir, "cluster_task_unit_outputs", "deepseek"
        )
    except NameError:
        print("警告：无法自动确定脚本目录，将使用相对于当前工作目录的路径。")
        cwd = os.getcwd()
        base_data_dir_rel = "browser_use_myown/agent/Task_unit_utils/data"
        default_output_dir = os.path.join(
            cwd, base_data_dir_rel, "cluster_graph_outputs", "deepseek"
        )
        default_taskunit_dir = os.path.join(
            cwd, base_data_dir_rel, "cluster_task_unit_outputs", "deepseek"
        )

    print(f"使用 Task Unit 目录: {default_taskunit_dir}")
    print(f"使用图输出目录: {default_output_dir}")

    return await main_cluster_async(
        clusters_by_distance=clusters_by_distance,
        df=df,
        taskunit_dir=default_taskunit_dir,
        output_dir=default_output_dir,
        max_concurrency=max_concurrency,
        mode=mode,
    )
