import os
import json
import openai
from openai import AzureOpenAI
import requests
import pandas as pd
from tqdm import tqdm
import ast
import asyncio
import numpy as np
from tqdm.asyncio import tqdm as async_tqdm

task_unit_prompt = (
    "You will receive user interaction trajectories, which represent sequences of atomic actions performed by users when completing web-based tasks. "
    "Your goal is to identify and define generalized task units from these trajectories clearly and concisely.\n\n"
    "Task units represent **human-level goals** and are higher-level, semantically meaningful abstractions. Each task unit should:\n"
    "- Group atomic actions based on semantic similarity and functional objectives.\n"
    "- Be named to reflect a general, reusable **task goal** rather than task-specific details.\n"
    "- Exhibit generalizability across different tasks within similar domains and subdomains.\n"
    "- A task unit may also consist of smaller task units, representing hierarchical steps within larger human-level goals.\n\n"
    "### Sub-action Abstraction Guidelines\n"
    "Sub-actions represent **webpage interaction steps** taken by the user to achieve the task unit. For each task unit, provide a list of **abstracted sub-actions** that:\n"
    "- Generalize specific content (e.g., replace city names, dates with placeholders like 'TYPE: Location' or 'SELECT: Date').\n"
    "- Focus on **web interaction steps** rather than human-level task goals (e.g., `TYPE`, `CLICK`, `SELECT` actions).\n\n"
    "### Steps to Follow\n"
    "1. **Identify Task Units:**\n"
    "   - Analyze the trajectory and group atomic actions into higher-level **human-level task units**.\n"
    "   - Each task unit should represent a meaningful **task goal** in the task completion process.\n\n"
    "2. **List Abstracted Sub-actions:**\n"
    "   - List the abstracted sub-actions that represent **web interaction steps** required to achieve the task unit.\n\n"
    "3. **Use the EXACT JSON structure below for output:**\n"
    "{{\n"
    '  "task_units": [\n'
    "    {\n"
    '      "id": "task_unit_id(eg.TU1)",\n'
    '      "name": "Generalized Task Unit Name (e.g., Book a flight from A to B)",\n'
    '      "sub_actions": [\n'
    '        "Generalized sub-action 1 (e.g., Select departure city)",\n'
    '        "Generalized sub-action 2 (e.g., Select arrival city)",\n'
    '        "Generalized sub-action 3 (e.g., View calendar)",\n'
    '        "Generalized sub-action 4 (e.g., Select date)",\n'
    '        "Generalized sub-action 5 (e.g., Click search)",\n'
    "        ...\n"
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}}\n\n"
    "### Important Notes:\n"
    "- **Do NOT include specific values** like 'Chicago' or 'April 19'—replace them with generalized placeholders (e.g., 'TYPE: Location', 'SELECT: Date').\n"
    "- **Do NOT reference specific UI components** (e.g., '[textbox]', '[button]')—focus on **abstracted web actions**.\n"
    "- **A task unit may consist of smaller task units**, which should be represented as part of a hierarchical structure reflecting the human-level goal.\n\n"
    "Here is the user interaction trajectory:\n\n"
    "{trajectory}"
)


Graph_generating_prompt = """
### Instruction

You are provided with a series of tasks and a set of task units extracted from their trajectories. Each task unit represents a high-level, semantically meaningful sub-goal abstracted from atomic user interactions on a website.

Your task is to analyze the provided tasks and their corresponding user interaction trajectories to construct a Directed Acyclic Graph (DAG) that accurately reflects the hierarchical and compositional dependencies among task units.

### Clarification on Hierarchical Relationships:

- **Parent task units:**  
  Represent broader, higher-level goals that implicitly or explicitly include sub-steps performed in child task units. These are typically overarching tasks that require completion of smaller, specific actions.

- **Child task units:**  
  Represent narrower, lower-level tasks that are necessary steps or components of the broader parent task unit. These steps are performed to reach the overall goal.

**Important Note:**  
- Parent task units may occur **before or after** child task units in user trajectories. For example, the broader goal of "Create Flight Alert" might include the step "Search Flights" as a prerequisite, even though it comes later in the user's flow.
- Do **NOT** treat the task sequence as the only factor in defining parent-child relationships. Focus on **semantic containment** (e.g., one task goal is a necessary part of completing another) rather than simple task order.

### Steps

1. **Identify Hierarchical Dependencies:**
   - Group task units based on semantic relationships, with broader goals being the parent and narrower tasks being the child.
   - Ensure dependencies reflect logical steps and task objectives, not just the order of completion.

2. **Ensure DAG Structure:**
   - Construct a directed acyclic graph (DAG) to represent the flow of task units, ensuring no cyclic dependencies exist.

3. **Provide Clear Reasoning:**
   - Justify the relationships between task units based on the user trajectories and how each task unit is a sub-goal within the broader task.

### Output Format (JSON):

{
  "task_unit_dag": [
    {
      "parent_task_unit_id": "parent_task_unit_id",
      "child_task_unit_ids": ["child_task_unit_id_1", "child_task_unit_id_2"],
      "relationship_reasoning": "Reasoning based on task objectives and user trajectory, showing that the parent task unit includes the child task units."
    },
    {
      "parent_task_unit_id": "...",
      "child_task_unit_ids": ["...", "..."],
      "relationship_reasoning": "..."
    }
  ],
  "validation": {
    "is_acyclic": true,
    "acyclic_reasoning": "Provide justification to confirm there are no cycles in the DAG, ensuring task dependencies reflect logical hierarchies."
  }
}

### Example:

{
  "task_unit_dag": [
    {
      "parent_task_unit_id": "TU6:Create Flight Alert",
      "child_task_unit_ids": ["TU1:Search Flights"],
      "relationship_reasoning": "Creating a flight alert encompasses searching and selecting a specific flight. This makes 'Search Flights' a subordinate action within 'Create Flight Alert', based on task dependencies in the user trajectories."
    },
    ...
  ],
  "validation": {
    "is_acyclic": true,
    "acyclic_reasoning": "All relationships strictly follow a hierarchical structure of task objectives, with no cycles or contradictions in the dependencies."
  }
}

Please analyze tasks and trajectories carefully to produce an accurate hierarchical DAG structure following these clarified guidelines.

Task units are provided here:
{task_units}

Trajectories are provided here:
{trajectories}
"""

task_unit_prompt = (
    task_unit_prompt.replace("{", "{{")
    .replace("}", "}}")
    .replace("{{trajectory}}", "{trajectory}")
)
Graph_generating_prompt = (
    Graph_generating_prompt.replace("{", "{{")
    .replace("}", "}}")
    .replace("{{trajectories}}", "{trajectories}")
    .replace("{{task_units}}", "{task_units}")
)


class AIServiceManager:
    def __init__(self, config_path=None, config_dict=None):
        """
        初始化服务管理器
        可以通过文件路径或直接传入字典来加载配置

        Args:
            config_path: JSON配置文件的路径
            config_dict: 直接传入的配置字典
        """
        if config_dict:
            self.config = config_dict
        elif config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            raise ValueError("需要提供配置文件路径或配置字典")

        self.openai_client = openai.OpenAI(
            api_key="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",
            base_url="https://openrouter.ai/api/v1"
        )

    def get_deepseek_completion(self, prompt, use_backup=False):
        """调用 OpenAI 客户端 API"""
        try:
            response = self.openai_client.chat.completions.create(
                model="deepseek/deepseek-chat",  # 根据实际可用模型调整
                messages=[
                {
                    "role": "system",
                    "content": "You are a JSON generator. Always respond with only pure, valid JSON without markdown formatting, code blocks, or any explanations. Your output should start with '{' and end with '}' and be directly parseable by JSON.parse().",
                },
                {"role": "user", "content": prompt},
            ],
                response_format={"type": "json_object"},
            )
            
            return {
                "choices": [
                    {
                        "message": {
                            "content": response.choices[0].message.content
                        }
                    }
                ]
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "choices": []
            }

    def filter_unique_entries_by_website(self, dataset, website):
        """
        按网站筛选数据条目，并确保annotation_id独特

        参数:
            dataset: 数据集
            website: 目标网站

        返回:
            筛选后的数据条目列表，annotation_id独特
        """
        seen_annotation_ids = set()
        unique_entries = []

        for entry in dataset:
            if (
                entry["website"] == website
                and entry["annotation_id"] not in seen_annotation_ids
            ):
                unique_entries.append(entry)
                seen_annotation_ids.add(entry["annotation_id"])

        return unique_entries


def trajectories2taskunit(
    df,
    service_manager,
    task_unit_prompt,
    output_dir="task_unit_outputs",
    mode="formal",
):
    """
    处理指定网站的数据，生成元操作并保存结果

    参数:
        df: 包含网站数据的DataFrame
        service_manager: AIServiceManager实例
        task_unit_prompt: 元操作提示模板
        output_dir: 输出目录
    """
    try:
        model_dir = os.path.join(output_dir, "deepseek")
        os.makedirs(model_dir, exist_ok=True)

        website = df["website"].iloc[0]

        # 构建清晰结构的轨迹字符串
        trajectory = ""
        for idx, (_, row) in enumerate(df.iterrows()):
            try:
                actions = (
                    ast.literal_eval(row["action_reprs"])
                    if isinstance(row["action_reprs"], str)
                    else row["action_reprs"]
                )
            except Exception:
                actions = row["action_reprs"]

            if isinstance(actions, list):
                actions_text = "\n".join(f"- {a}" for a in actions)
            else:
                actions_text = str(actions)

            trajectory += (
                f"=== Trajectory {idx + 1} ===\n"
                f"[Domain]: {row['domain']}\n"
                f"[Subdomain]: {row['subdomain']}\n"
                f"[Website]: {row['website']}\n"
                f"[Task Description]: {row['confirmed_task']}\n"
                f"[Action Representations]:\n{actions_text}\n"
                f"===========================\n\n"
            )

        # print("开始构建prompt")
        try:
            prompt = task_unit_prompt.format(trajectory=trajectory)
            # print("prompt构建完成")
        except Exception as e:
            print("❌ Prompt 构建失败:", str(e))
            with open(
                f"{output_dir}/debug_prompt_error.txt", "w", encoding="utf-8"
            ) as f:
                f.write(task_unit_prompt)
            raise

        # 保存构建的 prompt 供检查
        if mode != "formal":
            with open(
                os.path.join(model_dir, f"{website}_prompt.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(prompt)

        # 调用模型
        # print(f"生成元操作...")
        deepseek_response = service_manager.get_deepseek_completion(prompt)
        response_content = deepseek_response["choices"][0]["message"]["content"]
        # print("模型输出已获取")

        # 保存原始响应
        if mode != "formal":
            with open(
                os.path.join(model_dir, f"{website}_model_raw.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(response_content)

        # 尝试解析或清理 JSON
        try:
            parsed_json = json.loads(response_content)
            json_content = json.dumps(parsed_json, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            print("⚠️ JSON 解析失败，尝试清理内容...")
            json_content = clean_model_response(response_content)
            try:
                parsed_json = json.loads(json_content)
                json_content = json.dumps(parsed_json, ensure_ascii=False, indent=2)
            except json.JSONDecodeError as je:
                print(f"最终仍无法解析: {str(je)}")
                with open(
                    os.path.join(model_dir, f"{website}_raw_response.txt"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(response_content)
                return False

        # 保存清理后的 JSON
        output_file = os.path.join(model_dir, f"{website}_task_unit.json")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json_content)

        print(f"✓ 已保存结果到: {output_file}")
        return True

    except Exception as e:
        print(f"处理网站 {website} 时出错: {str(e)}")
        with open(os.path.join(output_dir, "errors.log"), "a", encoding="utf-8") as f:
            f.write(f"{website}: {str(e)}\n")
        return False


def taskunit2DAG(
    trajectories, task_units, catagory: str, output_dir="graph_outputs", mode="formal"
):
    """
    使用传入的trajectories和task_units生成DAG并保存结果

    参数:
        trajectories: 轨迹列表
        task_units: 元操作列表
        website: 网站名称
        output_dir: 输出目录
        mode: 输出模式（formal为正式模式，其他值为调试模式）

    返回:
        处理成功返回True，失败返回False
    """
    try:
        # 格式化轨迹和元操作为字符串
        trajectories_str = json.dumps(trajectories, ensure_ascii=False, indent=2)
        task_units_str = json.dumps(task_units, ensure_ascii=False, indent=2)

        # 在debug模式下保存输入数据
        if mode == "debug":
            # 保存trajectories数据
            trajectories_file = os.path.join(
                output_dir, f"{catagory}_trajectories.json"
            )
            with open(trajectories_file, "w", encoding="utf-8") as f:
                f.write(trajectories_str)
            print(f"✓ 已保存 {catagory} 的轨迹数据到: {trajectories_file}")

            # 保存task_units数据
            task_units_file = os.path.join(output_dir, f"{catagory}_task_units.json")
            with open(task_units_file, "w", encoding="utf-8") as f:
                f.write(task_units_str)
            print(f"✓ 已保存 {catagory} 的元操作数据到: {task_units_file}")

        # 初始化AI服务管理器
        service_manager = AIServiceManager(config_path="api_config.json")

        # 准备prompt
        try:
            final_prompt = Graph_generating_prompt.format(
                trajectories=trajectories_str, task_units=task_units_str
            )
        except Exception as e:
            print("❌ Prompt 构建失败:", str(e))
            with open(
                f"{output_dir}/debug_graph_prompt_error.txt", "w", encoding="utf-8"
            ) as f:
                f.write(Graph_generating_prompt)
            raise

        # 保存构建的 prompt 供检查（调试模式）
        if mode != "formal":
            with open(
                os.path.join(output_dir, f"{catagory}_graph_prompt.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(final_prompt)

        # 调用deepseek API
        print(f"正在为 {catagory} 调用Deepseek API生成图...")
        deepseek_response = service_manager.get_deepseek_completion(final_prompt)
        response_content = deepseek_response["choices"][0]["message"]["content"]

        # 保存原始响应（调试模式）
        if mode != "formal":
            with open(
                os.path.join(output_dir, f"{catagory}_graph_raw.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(response_content)

        # 尝试解析或清理 JSON
        try:
            parsed_json = json.loads(response_content)
            json_content = json.dumps(parsed_json, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            print("⚠️ JSON 解析失败，尝试清理内容...")
            json_content = clean_model_response(response_content)
            try:
                parsed_json = json.loads(json_content)
                json_content = json.dumps(parsed_json, ensure_ascii=False, indent=2)
            except json.JSONDecodeError as je:
                print(f"最终仍无法解析JSON: {str(je)}")
                with open(
                    os.path.join(output_dir, f"{catagory}_graph_raw_response.txt"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(response_content)
                return False

        # 保存清理后的 JSON
        output_file = os.path.join(output_dir, f"{catagory}_graph.json")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json_content)

        print(f"✓ 已保存 {catagory} 的图结构到: {output_file}")
        return True

    except Exception as e:
        print(f"处理聚类 {catagory} 的图生成时出错: {str(e)}")
        with open(
            os.path.join(output_dir, "graph_errors.log"), "a", encoding="utf-8"
        ) as f:
            f.write(f"{catagory}: {str(e)}\n")
        return False


def clean_model_response(response_content):
    """
    清理模型返回的响应，提取有效的JSON部分

    参数:
        response_content: 模型的原始响应内容

    返回:
        清理后的JSON内容
    """
    print(
        "原始响应内容:",
        (
            repr(response_content[:100]) + "..."
            if len(response_content) > 100
            else repr(response_content)
        ),
    )

    # 尝试寻找完整的JSON结构
    if response_content.strip().startswith("{") and response_content.strip().endswith(
        "}"
    ):
        json_content = response_content.strip()
    else:
        # 检查是否缺少了开头的 {
        if (
            '"task_units"' in response_content
            and not response_content.strip().startswith("{")
        ):
            json_content = "{" + response_content.strip()
        else:
            # 尝试找到JSON的开始和结束位置
            start_idx = response_content.find("{")
            end_idx = response_content.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_content = response_content[start_idx:end_idx]
            else:
                # 如果没有找到完整的括号，尝试构建一个有效的JSON
                if '"task_units"' in response_content:
                    # 提取task_units部分
                    meta_start = response_content.find('"task_units"')
                    content = response_content[meta_start:]
                    # 确保有效的JSON格式
                    json_content = "{" + content

                    # 确保结尾有}
                    if not json_content.strip().endswith("}"):
                        json_content = json_content.strip() + "}"
                else:
                    # 无法修复，返回原始内容
                    json_content = response_content

    # 验证JSON是否有效
    try:
        parsed = json.loads(json_content)
        print("JSON清理成功!")
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except json.JSONDecodeError as e:
        print(f"JSON清理后仍无法解析: {str(e)}")
        print("尝试进一步修复...")

        # 进一步尝试修复常见问题
        # 1. 检查是否缺少引号
        fixed_content = json_content
        if (
            '"task_units"' in fixed_content
            and '"id"' not in fixed_content
            and 'id":' in fixed_content
        ):
            fixed_content = fixed_content.replace('id":', '"id":')

        # 2. 确保task_units是一个数组
        if (
            '"task_units":' in fixed_content
            and '["' not in fixed_content
            and "[{" not in fixed_content
        ):
            fixed_content = fixed_content.replace('"task_units":', '"task_units": [')
            if not fixed_content.strip().endswith("}"):
                fixed_content = fixed_content.strip() + "]}"
            else:
                fixed_content = fixed_content[:-1] + "]}"

        # 再次验证
        try:
            parsed = json.loads(fixed_content)
            print("JSON进一步修复成功!")
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            # 最后的尝试：构建一个最小的有效JSON
            minimal_json = '{"task_units": []}'
            print("无法修复JSON，返回空的元操作数组")
            return minimal_json


async def generate_cluster_task_units(
    clusters_by_distance,
    df,
    service_manager,
    output_dir="cluster_task_unit_outputs",
    mode="formal",
):
    """
    基于聚类结果生成task_unit (添加进度条)

    参数:
        clusters_by_distance: 层次聚类的结果
        df: 原始数据DataFrame
        service_manager: AIServiceManager实例
        output_dir: 输出目录
        mode: 输出模式（formal为正式模式，其他值为调试模式）
    """
    try:
        model_dir = os.path.join(output_dir, "deepseek")
        os.makedirs(model_dir, exist_ok=True)

        # 按聚类ID组织数据
        cluster_data = {}
        print("组织聚类数据...")
        for idx, cluster_id in enumerate(clusters_by_distance):
            if cluster_id not in cluster_data:
                cluster_data[cluster_id] = []
            cluster_data[cluster_id].append(df.iloc[idx])
        print(f"数据组织完成，共 {len(cluster_data)} 个聚类。")

        results = {}
        # --- 添加 tqdm 包装器到这个循环 ---
        print("开始处理各个聚类...")
        # 使用 tqdm 包装 cluster_data.items()
        for cluster_id, rows in tqdm(
            cluster_data.items(), desc="Processing Clusters", unit="cluster"
        ):
            # print(f"正在处理聚类 {cluster_id}...") # tqdm 会显示进度，可以注释掉这个

            # 构建轨迹字符串
            trajectory = ""
            for idx, row in enumerate(rows):  # 内部循环可以不加进度条，避免过多输出
                try:
                    actions = (
                        ast.literal_eval(row["action_reprs"])
                        if isinstance(row["action_reprs"], str)
                        else row["action_reprs"]
                    )
                except Exception:
                    actions = row["action_reprs"]

                if isinstance(actions, list):
                    actions_text = "\n".join(f"- {a}" for a in actions)
                else:
                    actions_text = str(actions)

                trajectory += (
                    f"=== Trajectory {idx + 1} ===\n"
                    f"[Domain]: {row['domain']}\n"
                    f"[Subdomain]: {row['subdomain']}\n"
                    f"[Website]: {row['website']}\n"
                    f"[Task Description]: {row['confirmed_task']}\n"
                    f"[Action Representations]:\n{actions_text}\n"
                    f"===========================\n\n"
                )

            # 构建prompt
            try:
                prompt = task_unit_prompt.format(trajectory=trajectory)
            except Exception as e:
                print(f"❌ 聚类 {cluster_id} 的Prompt构建失败:", str(e))
                # 在tqdm循环中，最好显式更新描述或添加后缀来指示错误
                # tqdm.write(f"❌ Cluster {cluster_id}: Prompt build failed: {e}")
                continue

            # 保存prompt（调试模式）
            if mode != "formal":
                prompt_file = os.path.join(
                    model_dir, f"cluster_{cluster_id}_prompt.txt"
                )
                try:
                    with open(prompt_file, "w", encoding="utf-8") as f:
                        f.write(prompt)
                except Exception as e_write:
                    tqdm.write(
                        f"⚠️ Cluster {cluster_id}: Failed to write prompt file: {e_write}"
                    )

            # 调用模型
            try:
                # --- 这里调用的是同步的 get_deepseek_completion，如果它是异步的，需要不同的处理 ---
                # 可以在这里添加一个子进度描述，但对于快速API调用可能意义不大
                # tqdm.set_description_str(f"Cluster {cluster_id} - Calling API")
                deepseek_response = service_manager.get_deepseek_completion(prompt)
                # tqdm.set_description_str(f"Cluster {cluster_id} - Processing response") # 处理响应

                if (
                    "choices" not in deepseek_response
                    or not deepseek_response["choices"]
                ):
                    tqdm.write(
                        f"❌ Cluster {cluster_id}: Invalid API response (no choices)."
                    )
                    continue  # 跳到下一个聚类

                response_content = deepseek_response["choices"][0]["message"]["content"]

                # 保存原始响应（调试模式）
                if mode != "formal":
                    raw_file = os.path.join(
                        model_dir, f"cluster_{cluster_id}_model_raw.txt"
                    )
                    try:
                        with open(raw_file, "w", encoding="utf-8") as f:
                            f.write(response_content)
                    except Exception as e_write:
                        tqdm.write(
                            f"⚠️ Cluster {cluster_id}: Failed to write raw response file: {e_write}"
                        )

                # 解析JSON
                try:
                    parsed_json = json.loads(response_content)
                    # 在保存之前修改task_unit的id
                    if "task_units" in parsed_json and isinstance(
                        parsed_json["task_units"], list
                    ):
                        for task_unit in parsed_json["task_units"]:
                            if "id" in task_unit:
                                # 如果ID已经有前缀就不添加
                                if not task_unit["id"].startswith(
                                    f"CLUSTER{cluster_id}"
                                ):
                                    task_unit["id"] = (
                                        f"CLUSTER{cluster_id}_{task_unit['id']}"
                                    )
                except json.JSONDecodeError:
                    json_content_cleaned = clean_model_response(response_content)
                    try:
                        parsed_json = json.loads(json_content_cleaned)
                        # 同样在这里也要修改ID
                        if "task_units" in parsed_json and isinstance(
                            parsed_json["task_units"], list
                        ):
                            for task_unit in parsed_json["task_units"]:
                                if "id" in task_unit:
                                    if not task_unit["id"].startswith(
                                        f"CLUSTER{cluster_id}"
                                    ):
                                        task_unit["id"] = (
                                            f"CLUSTER{cluster_id}_{task_unit['id']}"
                                        )
                    except json.JSONDecodeError as je:
                        tqdm.write(
                            f"❌ Cluster {cluster_id}: JSON clean & parse failed: {je}"
                        )
                        continue

                # 保存结果 (直接使用解析后的对象)
                output_file = os.path.join(
                    model_dir, f"cluster_{cluster_id}_task_unit.json"
                )
                try:
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(parsed_json, f, ensure_ascii=False, indent=2)
                    results[cluster_id] = parsed_json
                except Exception as e_write:
                    tqdm.write(
                        f"⚠️ Cluster {cluster_id}: Failed to write final JSON file: {e_write}"
                    )

            except Exception as e:
                tqdm.write(
                    f"❌ Cluster {cluster_id}: Error during API call or processing: {e}"
                )
                with open(
                    os.path.join(output_dir, "errors.log"), "a", encoding="utf-8"
                ) as f:
                    f.write(f"cluster_{cluster_id}: {str(e)}\n")
                continue  # 确保出错时继续下一个循环

        print("\n所有聚类处理完成。")
        return results

    except Exception as e:
        print(f"生成聚类task_unit时发生顶层错误: {str(e)}")
        import traceback

        print(traceback.format_exc())
        return None


async def run_cluster_task_units(
    clusters_by_distance, df, max_concurrency=12, mode="formal"
):
    """
    运行聚类task_unit生成的入口函数 (异步版本)

    参数:
        clusters_by_distance: 层次聚类的结果
        df: 原始数据DataFrame
        max_concurrency: 最大并发数 (在这个实现中未使用，因为generate_cluster_task_units内部是串行的)
        mode: 输出模式
    """
    # 初始化服务管理器
    service_manager = AIServiceManager(config_path="api_config.json")

    # 直接 await 异步函数，而不是调用 asyncio.run()
    results = await generate_cluster_task_units(
        clusters_by_distance=clusters_by_distance,
        df=df,
        service_manager=service_manager,
        mode=mode,
    )

    return results


def test_single_cluster_task_unit(
    cluster_id=1,  # 想要测试的簇的ID
    clusters_by_distance=None,
    df=None,
    output_dir="test_cluster_outputs",
    mode="debug",  # 默认使用debug模式以便查看详细信息
):
    """
    测试单个簇的task_unit生成 (增加调试信息)

    参数:
        cluster_id: 要测试的簇的ID
        clusters_by_distance: 聚类结果
        df: 原始数据DataFrame
        output_dir: 输出目录
        mode: 输出模式（debug/formal）
    """
    print(f"[Debug] 开始测试簇 {cluster_id} 的task_unit生成...")

    # 1. 初始化服务管理器
    try:
        service_manager = AIServiceManager(config_path="api_config.json")
        print("[Debug] ✓ 服务管理器初始化成功")
    except Exception as e:
        print(f"[Debug] ✗ 服务管理器初始化失败: {str(e)}")
        return

    # 2. 创建输出目录
    model_dir = os.path.join(output_dir, "deepseek")
    os.makedirs(model_dir, exist_ok=True)
    print(f"[Debug] ✓ 创建输出目录成功: {model_dir}")

    try:
        # 3. 获取指定簇的数据
        print(f"[Debug] 正在根据 cluster_id={cluster_id} 筛选数据...")
        if clusters_by_distance is None or df is None:
            print("[Debug] ✗ 错误：clusters_by_distance 或 df 为 None。")
            return
        cluster_indices = np.where(clusters_by_distance == cluster_id)[0]
        print(f"[Debug] 找到 {len(cluster_indices)} 个属于簇 {cluster_id} 的索引。")

        if len(cluster_indices) == 0:
            print(f"[Debug] ✗ 错误：簇 {cluster_id} 没有样本，无法继续。")
            return

        cluster_df = df.iloc[cluster_indices]
        print(
            f"[Debug] ✓ 已成功提取簇 {cluster_id} 的 DataFrame，行数: {len(cluster_df)}"
        )

        print(f"\n簇 {cluster_id} 的基本信息:")
        print(f"样本数量: {len(cluster_df)}")
        print("\n任务描述示例:")
        for i, task in enumerate(cluster_df["confirmed_task"].head(5), 1):
            print(f"{i}. {task}")

        # 4. 构建轨迹字符串
        print(f"\n[Debug] 开始构建簇 {cluster_id} 的轨迹字符串...")
        trajectory = ""
        processed_rows = 0
        for idx, (df_idx, row) in enumerate(
            cluster_df.iterrows()
        ):  # 使用 df_idx 获取原始索引
            try:
                actions = (
                    ast.literal_eval(row["action_reprs"])
                    if isinstance(row["action_reprs"], str)
                    else row["action_reprs"]
                )
            except Exception as e:
                print(
                    f"[Debug] ⚠️ 解析 action_reprs 时出错 (DF index {df_idx}, 簇内第 {idx+1} 条): {e}. 使用原始数据。"
                )
                actions = row["action_reprs"]

            if isinstance(actions, list):
                actions_text = "\n".join(f"- {a}" for a in actions)
            else:
                actions_text = str(actions)

            # 检查关键字段是否存在
            domain = row.get("domain", "N/A")
            subdomain = row.get("subdomain", "N/A")
            website = row.get("website", "N/A")
            confirmed_task = row.get("confirmed_task", "N/A")

            trajectory += (
                f"=== Trajectory {idx + 1} (DataFrame Index: {df_idx}) ===\n"
                f"[Domain]: {domain}\n"
                f"[Subdomain]: {subdomain}\n"
                f"[Website]: {website}\n"
                f"[Task Description]: {confirmed_task}\n"
                f"[Action Representations]:\n{actions_text}\n"
                f"====================================================\n\n"  # 增加分隔符长度
            )
            processed_rows += 1

        print(f"[Debug] ✓ 轨迹字符串构建完成，处理了 {processed_rows} 行。")
        print(f"[Debug] 轨迹字符串总长度: {len(trajectory)}")
        if len(trajectory) > 0:
            print(f"[Debug] 轨迹字符串开头片段:\n---\n{trajectory[:500]}...\n---")
        else:
            print("[Debug] ⚠️ 警告：构建的轨迹字符串为空！")

        # 5. 构建prompt
        print("\n[Debug] 开始构建最终 Prompt...")
        try:
            # 打印模板信息以确认占位符存在
            print(f"[Debug] task_unit_prompt 模板长度: {len(task_unit_prompt)}")
            if "{trajectory}" not in task_unit_prompt:
                print(
                    "[Debug] ✗ 错误：task_unit_prompt 模板中缺少 {{trajectory}} 占位符！"
                )
                return

            prompt = task_unit_prompt.format(trajectory=trajectory)
            print("[Debug] ✓ Prompt 格式化成功")
            print(f"[Debug] 最终 Prompt 总长度: {len(prompt)}")
            if len(prompt) > len(task_unit_prompt):
                print("[Debug] ✓ 最终 Prompt 长度大于模板长度，看起来轨迹已加入。")
            else:
                print(
                    "[Debug] ⚠️ 警告：最终 Prompt 长度不大于模板长度，轨迹可能未正确加入！"
                )

            # 保存prompt供检查
            prompt_file = os.path.join(model_dir, f"cluster_{cluster_id}_prompt.txt")
            print(f"[Debug] 正在保存 Prompt 到: {prompt_file}")
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(prompt)
            print(f"[Debug] ✓ Prompt已保存至: {prompt_file}")

        except KeyError as e:
            print(
                f"[Debug] ✗ Prompt 构建失败 (KeyError): {str(e)}。很可能是模板中的占位符不匹配。"
            )
            return
        except Exception as e:
            print(f"[Debug] ✗ Prompt 构建失败 (其他错误): {str(e)}")
            return

        # 6. 调用模型 (添加 try...except 块并报告 prompt 长度)
        print("\n[Debug] 准备调用模型生成 task_unit...")
        print(f"[Debug] 发送给模型的 Prompt 长度: {len(prompt)}")
        try:
            deepseek_response = service_manager.get_deepseek_completion(prompt)
            print("[Debug] ✓ 模型调用成功。")
            if "choices" not in deepseek_response or not deepseek_response["choices"]:
                print("[Debug] ✗ 错误：模型响应中缺少 'choices' 或 'choices' 为空。")
                print(f"[Debug] 完整响应: {deepseek_response}")
                return None  # 或者抛出错误

            response_content = deepseek_response["choices"][0]["message"]["content"]
            print(f"[Debug] ✓ 获取到模型响应内容，长度: {len(response_content)}")
            if len(response_content) > 0:
                print(
                    f"[Debug] 模型响应内容开头片段:\n---\n{response_content[:200]}...\n---"
                )

            # 保存原始响应
            raw_response_file = os.path.join(
                model_dir, f"cluster_{cluster_id}_raw_response.txt"
            )
            print(f"[Debug] 正在保存原始响应到: {raw_response_file}")
            with open(raw_response_file, "w", encoding="utf-8") as f:
                f.write(response_content)
            print(f"[Debug] ✓ 原始响应已保存至: {raw_response_file}")

            # 7. 解析和保存结果
            print("\n[Debug] 开始解析模型响应...")
            try:
                parsed_json = json.loads(response_content)
                print("[Debug] ✓ JSON 直接解析成功。")
                json_content = json.dumps(parsed_json, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                print("[Debug] ⚠️ JSON 直接解析失败，尝试调用 clean_model_response...")
                json_content = clean_model_response(response_content)
                try:
                    parsed_json = json.loads(json_content)
                    print("[Debug] ✓ JSON 清理和解析成功。")
                except json.JSONDecodeError as je:
                    print(f"[Debug] ✗ 错误：清理后 JSON 解析仍然失败: {je}")
                    print(
                        f"[Debug] 清理后的内容 (前 500 字符):\n---\n{json_content[:500]}...\n---"
                    )
                    # 即使解析失败也尝试保存清理后的内容
                    failed_parse_file = os.path.join(
                        model_dir, f"cluster_{cluster_id}_cleaned_failed_parse.txt"
                    )
                    with open(failed_parse_file, "w", encoding="utf-8") as f:
                        f.write(json_content)
                    print(
                        f"[Debug] 清理后但解析失败的内容已保存至: {failed_parse_file}"
                    )
                    return None  # 解析失败则返回

            # 保存最终结果
            output_file = os.path.join(
                model_dir, f"cluster_{cluster_id}_task_unit.json"
            )
            print(f"[Debug] 正在保存最终 JSON 结果到: {output_file}")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(json_content)
            print(f"\n✓ 最终结果已保存至: {output_file}")

            # 8. 打印task_unit概要
            print("\n生成的Task Units概要:")
            if "task_units" in parsed_json and isinstance(
                parsed_json["task_units"], list
            ):
                for task_unit in parsed_json["task_units"]:
                    tu_id = task_unit.get("id", "N/A")
                    tu_name = task_unit.get("name", "N/A")
                    sub_actions_count = len(task_unit.get("sub_actions", []))
                    print(f"\nID: {tu_id}")
                    print(f"名称: {tu_name}")
                    print(f"子操作数量: {sub_actions_count}")
            else:
                print(
                    "[Debug] ⚠️ 警告：解析后的 JSON 中缺少 'task_units' 列表或格式不正确。"
                )

            return parsed_json

        except Exception as e:
            print(f"[Debug] ✗ 模型调用或结果处理过程中发生未预料的错误: {str(e)}")
            import traceback

            print(traceback.format_exc())  # 打印完整堆栈跟踪
            return None

    except Exception as e:
        print(f"[Debug] ✗ 测试过程发生未预料的错误: {str(e)}")
        import traceback

        print(traceback.format_exc())  # 打印完整堆栈跟踪
        return None
