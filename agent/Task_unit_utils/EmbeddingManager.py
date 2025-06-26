import faiss
import numpy as np
import pandas as pd
import os
import json
import logging
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from scipy.datasets import electrocardiogram
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Optional
import joblib
import re

# --- 添加 logger ---
logger = logging.getLogger(__name__)

# 定义中间结果文件的路径 (现在在 __init__ 中配置)
# INTERMEDIATE_PATH = "intermediate_data"
# CONTENTS_FILE = os.path.join(INTERMEDIATE_PATH, "task_contents.json")
# EMBEDDINGS_FILE = os.path.join(INTERMEDIATE_PATH, "embeddings.npy")
# IDS_FILE = os.path.join(INTERMEDIATE_PATH, "task_ids.json")

# --- Prompts (保持不变) ---
searching_prompt = """
### Instruction

You are provided with a **task description**, a set of **task units**, and a **task unit DAG**. The task units represent key steps in completing a task, while the DAG shows their relationships. Your goal is to identify the relevant task units needed to accomplish the task based on the description and DAG dependencies.

### Task Description
{task_description}

### Task Units
{task_units}

### Task Unit DAG
{task_unit_dag}

### Steps to Follow:

1. **Analyze Task Description**:
   - Identify which **task units** are relevant to the task, focusing on those that directly contribute to completing the task.

2. **Use the Task Unit DAG**:
   - Leverage the DAG's parent-child relationships to ensure logical flow and select task units in the correct order.
   - Only select task units that are logically connected to the task.

3. **Identify Relevant Task Units**:
   - Select **task units** that are necessary to complete the task, ensuring dependencies are respected.

4. **Return the Relevant Task Units**:
   - Provide the list of relevant task units with their **reason** for selection.

### Output Format (JSON):

{
  "relevant_task_units": [
    {
      "id": "task_unit_id",
      "name": "Task Unit Name",
      "sub_actions": ["Sub-action 1", "Sub-action 2", ...],
      "reason": "Explanation why this task unit is relevant"
    },
    ...
  ]
}

### Example:

For the task "Find the nearest store and set it as my preferred store":

{
  "relevant_task_units": [
    {
      "id": "CLUSTER1_TU1",
      "name": "Locate a store in a specified location",
      "sub_actions": [
        "CLICK on store locator",
        "ENTER location details",
        "CLICK submit search",
        "SELECT store"
      ],
      "reason": "This step is required to find a store before setting it as preferred."
    },
    {
      "id": "CLUSTER1_TU2",
      "name": "Set a store as preferred",
      "sub_actions": [
        "CLICK on set as my store",
        "CONFIRM selection"
      ],
      "reason": "This step makes the selected store the user's preferred one."
    }
  ]
}
"""

extract_traintask_prompt = """
You will be given a task description formatted as follows:

task_info = "{task}"

Where:
- **task**: A brief description of the task (e.g., "Find flights from Chicago to London").
- **website**: The website where the task is performed (e.g., "aa").
- **domain**: The general domain of the task (e.g., "Travel").
- **subdomain**: The specific subdomain of the task (e.g., "Airlines").
- **action_reprs**: The actions involved in the task, typically a list of interactions.

Your goal is to extract the following key details from the input:

1. **Task Name**: A brief title or type of the task (e.g., "Flight Search").
2. **Domain**: The general domain of the task (e.g., "Travel").
3. **Subdomain**: The specific subdomain of the task (e.g., "Airlines").
4. **Generalized Task Representation**: A high-level description of the task with placeholders for specific details (e.g., "Find flights from [departure city] to [destination city] on [departure date]").
5. **Sub-actions**: Key steps of the task.
6. **Website**: The website where the task is performed (e.g., Provided in the task_info).

Provide your answer in the following format without any other text:

{
  "task_name": "Flight Search",
  "domain": "Travel",
  "subdomain": "Airlines",
  "website": "aa",
  "generalized_task_representation": "Find flights from [departure city] to [destination city] on [departure date]",
  "sub_actions": ["Select departure city", "Select return date", "Click Search"]
}
"""

extract_searchtask_prompt = """
You will be given a task description formatted as follows:

task_info = "{task}"

Where:
- **task**: A description of the task (e.g., "Find flights from Chicago to London").

Your goal is to extract the following key details from the input:

1. **Task Name**: A brief title or type of the task (e.g., "Flight Search").
2. **Domain**: Summarize the general domain of the task (e.g., "Travel").
3. **Subdomain**: Summarize the specific subdomain of the task (e.g., "Airlines").
4. **Generalized Task Representation**: A high-level description of the task with placeholders for specific details (e.g., "Find flights from [departure city] to [destination city] on [departure date]").

Provide your answer in the following format without any other text:

{
  "task_name": "Flight Search",
  "domain": "Travel",
  "subdomain": "Airlines",
  "generalized_task_representation": "Find flights from [departure city] to [destination city] on [departure date]",
  "website": "aa"
}
"""


class TaskEmbeddingManager:
    """
    任务嵌入管理器 (初始化时不自动加载 DataFrame)
    """

    def __init__(
        self,
        service_manager,
        dataset: Optional[pd.DataFrame] = None,  # Allow passing df directly
        dataset_path: Optional[
            str
        ] = "/home/douzc/users/yuyao/browser_use_taskunit/agent/Task_unit_utils/dataset/mind2web_dataset.parquet",  # Store path for later use
        embedding_dim=1536,
        intermediate_path="/home/douzc/users/yuyao/browser_use_taskunit/agent/Task_unit_utils/data/intermediate_data",
        task_unit_base_path="/home/douzc/users/yuyao/browser_use_taskunit/agent/Task_unit_utils/data/cluster_task_unit_outputs",
        graph_base_path="/home/douzc/users/yuyao/browser_use_taskunit/agent/Task_unit_utils/data/cluster_graph_outputs",
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = "https://aihubmix.com/v1",
    ):
        """
        初始化任务嵌入管理器，定义路径，尝试加载中间结果，但不自动加载主数据集。
        """
        # 基础设置
        self.service_manager = service_manager
        self.embedding_dim = embedding_dim
        self.intermediate_path = intermediate_path
        self.task_unit_base_path = task_unit_base_path
        self.graph_base_path = graph_base_path
        self.dataset_path = dataset_path  # Store the path for potential later loading

        # --- 定义所有中间文件路径 ---
        os.makedirs(self.intermediate_path, exist_ok=True)  # Ensure dir exists early
        # 嵌入相关
        self.contents_file = os.path.join(self.intermediate_path, "task_contents.json")
        self.embeddings_file = os.path.join(self.intermediate_path, "embeddings.npy")
        self.ids_file = os.path.join(self.intermediate_path, "task_ids.json")
        # FAISS 索引与映射
        self.final_index_file = os.path.join(
            self.intermediate_path, "final_task_embeddings.index"
        )
        self.final_mapping_file = os.path.join(
            self.intermediate_path, "final_task_mapping.json"
        )
        # 聚类结果统一保存文件（使用 joblib，简化读写）
        self.cluster_data_file = os.path.join(
            self.intermediate_path, "clustering_data.joblib"
        )

        # --- 初始化状态变量 ---
        self.df = dataset  # Initialize with passed dataset if provided
        self.df_index_map = {}  # Initialize as empty
        self.index = None
        self.data_index = {}
        self.reverse_index = {}
        self.task_contents = {}
        self.embeddings = None
        self.raw_cluster_labels = None
        self.df_cluster_labels = None
        self.cluster_centroids = None
        self.openai_client = None

        # --- 如果直接传入了 dataset，则创建映射 ---
        if self.df is not None:
            logger.info("接收到直接传入的 DataFrame，将创建索引映射。")
            self._create_df_index_mapping()

        # --- 初始化 OpenAI 客户端 ---
        self._initialize_openai_client(openai_api_key, openai_base_url)

        # --- 尝试加载所有已存在的中间结果 (但不加载 df) ---
        self.load_all_results()

    def _initialize_openai_client(self, api_key, base_url):
        """Helper to initialize OpenAI client"""
        api_key_to_use = api_key
        if api_key_to_use is None:
            api_key_to_use = os.getenv("OPENAI_API_KEY_FOR_EMBEDDING")
            if not api_key_to_use:
                logger.warning(
                    "OpenAI API Key 未提供（传入值为 None）且环境变量 OPENAI_API_KEY_FOR_EMBEDDING 未设置。嵌入计算将失败。"
                )
                self.openai_client = None
                return
            else:
                logger.info("使用环境变量中的 OpenAI API Key 进行嵌入计算。")
        elif not api_key_to_use:
            logger.warning("传入的 OpenAI API Key 为空字符串。嵌入计算将失败。")
            self.openai_client = None
            return

        try:
            self.openai_client = OpenAI(api_key=api_key_to_use, base_url=base_url)
            logger.info("OpenAI 客户端初始化成功。")
        except Exception as e:
            logger.error(f"初始化 OpenAI 客户端失败: {e}", exc_info=True)
            self.openai_client = None

    # --- 数据集加载与映射 ---
    @staticmethod
    def load_dataset(dataset_path: str) -> Optional[pd.DataFrame]:
        """从指定路径加载数据集 (支持 .joblib 和 .parquet)"""
        if not os.path.exists(dataset_path):
            logger.error(f"数据集文件不存在: {dataset_path}")
            return None
        try:
            logger.info(f"从 {dataset_path} 加载数据集...")
            if dataset_path.endswith(".joblib"):
                df = joblib.load(dataset_path)
            elif dataset_path.endswith(".parquet"):
                # 尝试导入 pyarrow
                try:
                    import pyarrow
                except ImportError:
                    logger.error(
                        "加载 Parquet 需要 'pyarrow' 库。请运行 'pip install pyarrow'。"
                    )
                    raise  # 让调用者知道依赖缺失
                df = pd.read_parquet(dataset_path, engine="pyarrow")
            else:
                logger.error(
                    f"不支持的数据集文件格式: {dataset_path}. 请使用 .joblib 或 .parquet"
                )
                return None
            logger.info(f"数据集加载成功，大小: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"加载数据集 {dataset_path} 时出错: {e}", exc_info=True)
            return None

    def _create_df_index_mapping(self):
        """创建 annotation_id 到 DataFrame 行索引的映射"""
        if self.df is None:
            logger.warning("DataFrame 未提供或加载失败，无法创建索引映射。")
            self.df_index_map = {}
            return
        if "annotation_id" not in self.df.columns:
            logger.error("提供的 DataFrame 中缺少 'annotation_id' 列。")
            self.df_index_map = {}  # Reset map on error
            return

        logger.info("创建 DataFrame annotation_id 到行索引的映射...")
        try:
            # 先重置索引，保证 0..N-1 连续，防止后续 NumPy 赋值越界
            self.df = self.df.reset_index(drop=True)
            
            # 使用 enumerate 生成连续行索引映射，确保索引连续
            self.df_index_map = {
                str(ann_id): idx 
                for idx, ann_id in enumerate(self.df["annotation_id"].tolist())
            }
            logger.info(f"映射创建完成，包含 {len(self.df_index_map)} 个条目。")
        except Exception as e:
            logger.error(f"创建 DataFrame 索引映射时出错: {e}", exc_info=True)
            self.df_index_map = {}  # Reset map on error

    # --- 统一加载函数 ---
    def load_all_results(self):
        """尝试加载所有可用的中间结果。"""
        logger.info("尝试加载所有可用的中间结果...")
        loaded_embeddings = self._load_embeddings_data()
        loaded_index = self._load_faiss_index()
        loaded_clustering = self._load_clustering_results()

        if loaded_embeddings:
            logger.info("成功加载嵌入相关数据。")
        if loaded_index:
            logger.info("成功加载FAISS索引和映射。")
        if loaded_clustering:
            logger.info("成功加载聚类结果。")

        # 简单一致性检查
        if self.embeddings is not None and self.index is not None:
            if len(self.embeddings) != self.index.ntotal:
                logger.warning(
                    f"加载后发现不一致：嵌入数量 ({len(self.embeddings)}) 与索引大小 ({self.index.ntotal}) 不匹配。可能需要重新处理。"
                )
        if self.embeddings is not None and self.raw_cluster_labels is not None:
            if len(self.embeddings) != len(self.raw_cluster_labels):
                logger.warning(
                    f"加载后发现不一致：嵌入数量 ({len(self.embeddings)}) 与原始聚类标签数量 ({len(self.raw_cluster_labels)}) 不匹配。可能需要重新聚类。"
                )

    # --- 嵌入数据 保存/加载 ---
    def _load_embeddings_data(self) -> bool:
        """加载嵌入、ID、内容。"""
        if not all(
            os.path.exists(f)
            for f in [self.contents_file, self.ids_file, self.embeddings_file]
        ):
            # logger.info("未找到完整的嵌入相关文件，跳过加载。") # Less verbose
            return False
        try:
            logger.info("加载嵌入相关数据...")
            with open(self.contents_file, "r", encoding="utf-8") as f:
                loaded_contents = json.load(f)
            with open(self.ids_file, "r", encoding="utf-8") as f:
                loaded_ids = json.load(f)  # List
            loaded_embeddings = np.load(self.embeddings_file)

            # 基本验证
            if not (
                isinstance(loaded_contents, dict)
                or isinstance(loaded_ids, list)
                or isinstance(loaded_embeddings, np.ndarray)
            ):
                logger.warning(
                    "嵌入相关文件格式不正确。{}".format(type(loaded_contents))
                )
                return False
            if len(loaded_ids) != loaded_embeddings.shape[0]:
                logger.warning(
                    f"ID数量 ({len(loaded_ids)}) 与嵌入数量 ({loaded_embeddings.shape[0]}) 不匹配。"
                )
                return False

            # 更新实例状态
            self.task_contents = loaded_contents
            self.embeddings = loaded_embeddings
            # IDs 不直接存储，但在构建索引时需要，或者可以从 self.embeddings 重建
            logger.info(f"成功加载 {len(loaded_ids)} 个嵌入及相关数据。")
            return True
        except Exception as e:
            logger.error(f"加载嵌入相关数据时出错: {e}", exc_info=True)
            # 重置可能部分加载的状态
            self.task_contents = {}
            self.embeddings = None
            return False

    def _save_embeddings_data(
        self, task_ids: list, contents: dict, embeddings: np.ndarray
    ):
        """保存嵌入、ID、内容。"""
        if embeddings is None or not task_ids or not contents:
            logger.warning("尝试保存空的嵌入数据，已跳过。")
            return False
        try:
            logger.info("保存嵌入相关数据...")
            os.makedirs(self.intermediate_path, exist_ok=True)
            with open(self.contents_file, "w", encoding="utf-8") as f:
                json.dump(contents, f, ensure_ascii=False, indent=2)
            with open(self.ids_file, "w", encoding="utf-8") as f:
                json.dump(task_ids, f, ensure_ascii=False, indent=2)
            np.save(self.embeddings_file, embeddings)
            logger.info(f"成功保存 {len(task_ids)} 个嵌入及相关数据。")
            return True
        except Exception as e:
            logger.error(f"保存嵌入相关数据时出错: {e}", exc_info=True)
            return False

    # --- FAISS索引 保存/加载 ---
    def _load_faiss_index(self) -> bool:
        """加载FAISS索引及 annotation_id 顺序列表。"""
        if not os.path.exists(self.final_index_file) or not os.path.exists(
            self.final_mapping_file
        ):
            return False
        try:
            logger.info("加载FAISS索引...")
            self.index = faiss.read_index(self.final_index_file)

            with open(self.final_mapping_file, "r", encoding="utf-8") as f:
                mapping_data = json.load(f)
                self.faiss_ids = mapping_data.get("faiss_ids", [])
                # task_contents 仍然可能需要，用于快速 prompt 构建
                loaded_contents = mapping_data.get("task_contents", {})
                if loaded_contents:
                    self.task_contents.update({str(k): v for k, v in loaded_contents.items()})

            if self.index is None or not self.faiss_ids or self.index.ntotal != len(
                self.faiss_ids
            ):
                logger.warning("FAISS索引或映射文件内容不完整或不匹配。")
                self.index = None
                self.faiss_ids = []
                return False

            # 构建 reverse_index 动态映射
            self.reverse_index = {ann: idx for idx, ann in enumerate(self.faiss_ids)}

            logger.info(
                f"成功加载FAISS索引 (大小: {self.index.ntotal}) 和映射 ({len(self.faiss_ids)} 条)。"
            )
            return True
        except Exception as e:
            logger.error(f"加载FAISS索引或映射时出错: {e}", exc_info=True)
            self.index = None
            self.faiss_ids = []
            self.reverse_index = {}
            return False

    def _save_faiss_index(self):
        """保存FAISS索引和 annotation_id 顺序映射。"""
        if self.index is None or not self.faiss_ids:
            logger.warning("FAISS索引或映射未完全构建，无法保存。")
            return False
        try:
            logger.info("保存FAISS索引...")
            os.makedirs(self.intermediate_path, exist_ok=True)
            faiss.write_index(self.index, self.final_index_file)

            with open(self.final_mapping_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "faiss_ids": self.faiss_ids,
                        "task_contents": self.task_contents,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(
                f"成功保存FAISS索引 (大小: {self.index.ntotal}) 和映射 ({len(self.faiss_ids)} 条)。"
            )
            return True
        except Exception as e:
            logger.error(f"保存FAISS索引时出错: {e}", exc_info=True)
            return False

    # --- 聚类结果 保存/加载 ---
    def _load_clustering_results(self) -> bool:
        """从单一 joblib 文件加载聚类标签与质心。"""
        if not os.path.exists(self.cluster_data_file):
            return False

        try:
            logger.info("加载聚类结果 (joblib)...")
            data: dict = joblib.load(self.cluster_data_file)

            self.raw_cluster_labels = data.get("raw_cluster_labels")
            self.df_cluster_labels = data.get("df_cluster_labels")
            self.cluster_centroids = data.get("cluster_centroids")

            if (
                self.raw_cluster_labels is None
                or self.cluster_centroids is None
            ):
                logger.warning("聚类结果文件缺少必要字段，加载失败。")
                return False

            logger.info(
                f"成功加载聚类结果：标签 {len(self.raw_cluster_labels)} 条，"
                f"质心 {len(self.cluster_centroids)} 个。"
            )
            return True
        except Exception as e:
            logger.error(f"加载聚类结果时发生错误: {e}", exc_info=True)
            self.raw_cluster_labels = None
            self.df_cluster_labels = None
            self.cluster_centroids = None
            return False

    def _save_clustering_results(self):
        """将聚类标签与质心统一保存到单一 joblib 文件。"""
        if self.raw_cluster_labels is None or self.cluster_centroids is None:
            logger.warning("聚类结果或质心为空，跳过保存。")
            return False

        try:
            logger.info("保存聚类结果 (joblib)...")
            os.makedirs(self.intermediate_path, exist_ok=True)
            joblib.dump(
                {
                    "raw_cluster_labels": self.raw_cluster_labels,
                    "df_cluster_labels": self.df_cluster_labels,
                    "cluster_centroids": self.cluster_centroids,
                },
                self.cluster_data_file,
            )
            logger.info("聚类结果保存成功。")
            return True
        except Exception as e:
            logger.error(f"保存聚类结果时发生错误: {e}", exc_info=True)
            return False

    # --- 核心计算函数 (修改后) ---
    def process_dataset_parallel(self):
        """
        处理数据集，计算嵌入并构建FAISS索引。
        会先尝试加载现有嵌入数据，如果失败则尝试加载 DataFrame 并计算。
        """
        # 1. 尝试加载现有嵌入数据
        if self.embeddings is not None and self.task_contents:
            logger.info("发现已加载的嵌入数据，跳过嵌入计算。")
            # We need task_ids to build the index later if it's missing
            if not self.faiss_ids:  # If index wasn't loaded, we need IDs
                if os.path.exists(self.ids_file):
                    try:
                        with open(self.ids_file, "r", encoding="utf-8") as f:
                            task_ids_from_embeddings = json.load(f)
                        logger.info("从文件加载了 task_ids 用于后续索引构建。")
                    except Exception as e:
                        logger.error(
                            f"无法从 {self.ids_file} 加载 task_ids，可能无法构建索引。错误: {e}"
                        )
                        task_ids_from_embeddings = list(
                            self.task_contents.keys()
                        )  # Fallback
                else:
                    logger.warning(
                        f"未找到 task_ids 文件 ({self.ids_file})，将从 task_contents 推断，顺序可能不保证。"
                    )
                    task_ids_from_embeddings = list(
                        self.task_contents.keys()
                    )  # Fallback
            else:
                # If index *was* loaded, we don't strictly need task_ids_from_embeddings here
                pass

        else:
            # 需要计算嵌入: 确保 DataFrame 已加载
            if not self._ensure_df_loaded():
                logger.error("无法进行嵌入计算，因为 DataFrame 加载失败。")
                return
            # DataFrame 已加载，继续计算嵌入
            logger.info("开始计算嵌入和提取内容...")
            all_task_ids = self.df["annotation_id"].tolist()
            tasks_to_process = self.df["confirmed_task"].tolist()
            final_task_ids = []
            final_embeddings_list = []
            final_contents = {}
            extraction_errors = 0
            embedding_errors = 0
            # (Parallel extraction)
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(self.extract_task_content, task, mode="train"): i
                    for i, task in enumerate(tasks_to_process)
                }
                extracted_contents_list = [None] * len(tasks_to_process)
                for future in tqdm(
                    futures, total=len(tasks_to_process), desc="提取任务内容"
                ):
                    idx = futures[future]
                    try:
                        content = future.result()
                        extracted_contents_list[idx] = content
                    except Exception as exc:
                        logger.error(f"提取任务 {all_task_ids[idx]} 内容时出错: {exc}")
                        extraction_errors += 1
            # (Parallel embedding)
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(
                        self.compute_embedding, extracted_contents_list[i]
                    ): i
                    for i, content in enumerate(extracted_contents_list)
                    if content
                }
                embeddings_list = [None] * len(tasks_to_process)
                for future in tqdm(futures, total=len(futures), desc="计算嵌入向量"):
                    idx = futures[future]
                    try:
                        embedding = future.result()
                        embeddings_list[idx] = embedding
                    except Exception as exc:
                        logger.error(f"计算任务 {all_task_ids[idx]} 嵌入时出错: {exc}")
                        embedding_errors += 1
            # (Combine results)
            for i, task_id in enumerate(all_task_ids):
                if extracted_contents_list[i] and embeddings_list[i] is not None:
                    final_task_ids.append(task_id)
                    final_embeddings_list.append(embeddings_list[i])
                    final_contents[str(task_id)] = extracted_contents_list[i]

            if not final_task_ids:
                logger.error("未能成功处理任何任务的嵌入。")
                return
            self.embeddings = np.array(final_embeddings_list, dtype=np.float32)
            self.task_contents = final_contents
            task_ids_from_embeddings = final_task_ids
            self._save_embeddings_data(
                final_task_ids, self.task_contents, self.embeddings
            )

        # 2. 尝试加载现有 FAISS 索引 (或者如果上面刚计算了嵌入，就构建索引)
        if self.index is not None:
            logger.info("发现已加载的 FAISS 索引，跳过索引构建。")
            # Ensure task_contents from mapping file is merged if index was loaded separately
            if self.task_contents and hasattr(self, "_loaded_contents_from_mapping"):
                self.task_contents.update(self._loaded_contents_from_mapping)
                del self._loaded_contents_from_mapping  # Clean up temporary attribute
            return  # Index is ready

        # 3. 构建 FAISS 索引 (如果未加载)
        if self.embeddings is None:
            logger.error("错误：无法构建 FAISS 索引，因为嵌入向量不可用。")
            return
        if "task_ids_from_embeddings" not in locals() or not task_ids_from_embeddings:
            logger.error("错误：无法构建 FAISS 索引，因为任务 ID 列表不可用。")
            return

        logger.info("开始构建 FAISS 索引...")
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.faiss_ids = []
        self.reverse_index = {}
        processed_count = 0
        error_add_index_count = 0
        if len(task_ids_from_embeddings) != len(self.embeddings):
            logger.error(
                f"内部错误：任务 ID 数量 ({len(task_ids_from_embeddings)}) 与嵌入数量 ({len(self.embeddings)}) 不匹配。无法构建索引。"
            )
            self.index = None
            return
        for i, (annotation_id, embedding) in enumerate(
            tqdm(
                zip(task_ids_from_embeddings, self.embeddings),
                total=len(self.embeddings),
                desc="添加到索引",
            )
        ):
            try:
                annotation_id_str = str(annotation_id)
                faiss_id = self.index.ntotal
                embedding_to_add = np.ascontiguousarray(embedding.reshape(1, -1))
                self.index.add(embedding_to_add)
                self.faiss_ids.append(annotation_id_str)
                self.reverse_index[annotation_id_str] = faiss_id
                processed_count += 1
            except Exception as e:
                error_add_index_count += 1
                logger.error(
                    f"添加到索引时出错 (ID: {annotation_id_str}): {str(e)}",
                    exc_info=True,
                )
        if self.index is None or self.index.ntotal == 0:
            logger.error("FAISS 索引构建失败或为空。")
            return
        logger.info(
            f"FAISS 索引构建完成。大小: {self.index.ntotal}, 添加错误: {error_add_index_count}"
        )
        self._save_faiss_index()

    def perform_hierarchical_clustering(
        self, method="ward", metric="euclidean", criterion="distance", threshold=0.9
    ):
        """
        执行层次聚类。会先尝试加载现有聚类结果。
        """
        # 1. 尝试加载现有聚类结果
        if self.raw_cluster_labels is not None and self.cluster_centroids is not None:
            # 验证加载的标签数量是否与当前嵌入匹配
            if self.embeddings is not None and len(self.raw_cluster_labels) == len(
                self.embeddings
            ):
                logger.info("发现已加载且匹配的聚类结果，跳过聚类计算。")
                # 确保 df_cluster_labels 也加载了
                if self.df_cluster_labels is None:
                    # Try to map loaded raw labels if df exists
                    self.map_clusters_to_dataframe()
                return self.raw_cluster_labels
            else:
                logger.warning("加载的聚类结果与当前嵌入不匹配，将重新计算。")
                self.raw_cluster_labels = None
                self.df_cluster_labels = None
                self.cluster_centroids = None

        # 2. 确保嵌入向量可用
        if self.embeddings is None:
            if self.index is not None and self.index.ntotal > 0:
                logger.info("聚类需要嵌入向量，正在从FAISS索引重建...")
                try:
                    n_vectors = self.index.ntotal
                    self.embeddings = np.zeros(
                        (n_vectors, self.embedding_dim), dtype=np.float32
                    )
                    self.index.reconstruct_n(0, n_vectors, self.embeddings)
                    logger.info("向量重建完成。")
                except Exception as e:
                    logger.error(f"从 FAISS 重建向量失败: {e}", exc_info=True)
                    return None
            else:
                logger.error(
                    "错误: 无法执行聚类，嵌入向量和FAISS索引都不可用。请先运行 process_dataset_parallel。"
                )
                return None
        elif self.index is not None and len(self.embeddings) != self.index.ntotal:
            # This case should ideally be caught earlier, but double-check
            logger.error(
                f"错误：嵌入数量 ({len(self.embeddings)}) 与索引大小 ({self.index.ntotal}) 不匹配，无法聚类。"
            )
            return None

        # 3. 执行聚类计算
        logger.info("\n开始执行层次聚类...")
        try:
            logger.info(
                f"计算链接矩阵 (方法: {method}, 度量: {metric}) for {self.embeddings.shape[0]} vectors..."
            )
            linked = linkage(self.embeddings, method=method, metric=metric)
            logger.info("链接矩阵计算完成。")

            logger.info(
                f"根据标准 '{criterion}' 和阈值 {threshold} 获取扁平聚类结果..."
            )
            self.raw_cluster_labels = fcluster(linked, threshold, criterion=criterion)
            num_clusters = (
                self.raw_cluster_labels.max() if len(self.raw_cluster_labels) > 0 else 0
            )
            logger.info(f"聚类完成，得到 {num_clusters} 个簇。")

            # 4. 计算质心
            self.calculate_centroids()

            # 5. 映射到 DataFrame (如果 df 可用)
            self.map_clusters_to_dataframe()  # Safe to call even if df is None

            # 6. 保存聚类结果
            self._save_clustering_results()

            return self.raw_cluster_labels

        except MemoryError:
            logger.error(
                "错误：内存不足，无法计算链接矩阵。请考虑降维或采样。", exc_info=True
            )
            self.raw_cluster_labels = None
            self.cluster_centroids = None
            return None
        except Exception as e:
            logger.error(f"执行层次聚类时发生错误: {e}", exc_info=True)
            self.raw_cluster_labels = None
            self.cluster_centroids = None
            return None

    def map_clusters_to_dataframe(self):
        """
        将原始聚类标签映射到DataFrame。
        如果 df 不可用，会尝试加载它。
        """
        # 确保 DataFrame 已加载
        if not self._ensure_df_loaded():
            logger.warning("DataFrame 不可用，无法执行聚类标签到 DataFrame 的映射。")
            self.df_cluster_labels = None
            return None

        # --- 后续检查与之前类似 ---
        if self.raw_cluster_labels is None:
            logger.warning("原始聚类标签不可用，无法映射到 DataFrame。")
            self.df_cluster_labels = None
            return None
        if self.index is None or not self.faiss_ids:
            logger.warning("FAISS 反向映射不可用，无法映射聚类标签到 DataFrame。")
            self.df_cluster_labels = None
            return None
        if len(self.raw_cluster_labels) != self.index.ntotal:
            logger.warning(
                f"原始聚类标签数量 ({len(self.raw_cluster_labels)}) 与 FAISS 索引大小 ({self.index.ntotal}) 不匹配，无法映射。"
            )
            self.df_cluster_labels = None
            return None
        if not self.df_index_map:  # Check if map was created successfully
            logger.warning("DataFrame 索引映射不可用，无法映射聚类标签。")
            self.df_cluster_labels = None
            return None

        # --- 映射逻辑与之前类似 ---
        logger.info("开始将聚类标签映射到DataFrame顺序...")
        mapped_labels = np.full(len(self.df), -1, dtype=self.raw_cluster_labels.dtype)
        map_success_count = 0
        map_fail_count = 0
        for faiss_id, cluster_label in enumerate(
            tqdm(self.raw_cluster_labels, desc="映射聚类标签")
        ):
            if faiss_id < len(self.faiss_ids):
                annotation_id = self.faiss_ids[faiss_id]
            else:
                annotation_id = None
            if annotation_id is None:
                map_fail_count += 1
                continue
            df_index = self.df_index_map.get(annotation_id)
            if df_index is None or df_index >= len(mapped_labels):
                map_fail_count += 1
                continue
            mapped_labels[df_index] = cluster_label
            map_success_count += 1
        self.df_cluster_labels = mapped_labels
        logger.info(
            f"聚类标签映射完成。成功映射 {map_success_count} 个，失败 {map_fail_count} 个。"
        )
        return self.df_cluster_labels

    def calculate_centroids(self):
        """计算簇质心。"""
        if self.embeddings is None or self.raw_cluster_labels is None:
            logger.warning("嵌入向量或原始聚类标签不可用，无法计算质心。")
            self.cluster_centroids = None
            return
        if len(self.embeddings) != len(self.raw_cluster_labels):
            logger.warning("嵌入向量数量和聚类标签数量不匹配，无法计算质心。")
            self.cluster_centroids = None
            return

        logger.info("开始计算簇质心...")
        self.cluster_centroids = {}
        unique_labels = np.unique(self.raw_cluster_labels)
        if -1 in unique_labels:
            unique_labels = unique_labels[
                unique_labels != -1
            ]  # Exclude potential placeholders

        if len(unique_labels) == 0:
            logger.warning("没有有效的聚类标签，无法计算质心。")
            return

        for label in tqdm(unique_labels, desc="计算质心"):
            indices = np.where(self.raw_cluster_labels == label)[0]
            if len(indices) > 0:
                cluster_vectors = self.embeddings[indices]
                centroid = np.mean(cluster_vectors, axis=0)
                self.cluster_centroids[int(label)] = centroid  # Ensure key is int
            else:
                # This shouldn't happen if unique_labels is derived correctly
                logger.warning(f"簇 {label} 没有找到对应的向量，无法计算质心。")

        logger.info(
            f"质心计算完成，共计算了 {len(self.cluster_centroids)} 个簇的质心。"
        )

    def extract_task_content(self, task_description: str, mode="train") -> str:
        """
        使用DeepSeek模型提取任务的核心内容 (假定 service_manager 是线程安全的)
        """
        if mode == "train":
            formatted_prompt = extract_traintask_prompt.replace(
                "{task}", task_description
            )
        elif mode == "search":
            formatted_prompt = extract_searchtask_prompt.replace(
                "{task}", task_description
            )
        else:
            logger.warning(f"未知的提取模式: {mode}")
            return ""
        try:
            # 确保 service_manager 可用
            if self.service_manager is None:
                logger.error("Service manager 未初始化，无法提取任务内容。")
                return ""
            response = self.service_manager.get_deepseek_completion(formatted_prompt)
            # 检查响应结构
            if (
                response
                and "choices" in response
                and response["choices"]
                and "message" in response["choices"][0]
                and "content" in response["choices"][0]["message"]
            ):
                content = response["choices"][0]["message"]["content"]
                # 简单的 JSON 格式检查 (可以根据需要增强)
                if (
                    isinstance(content, str)
                    and content.strip().startswith("{")
                    and content.strip().endswith("}")
                ):
                    return content
                elif isinstance(content, str):
                    logger.warning(
                        f"提取到的内容似乎不是有效的JSON: {content[:100]}..."
                    )
                    return content  # 返回非 JSON 字符串，调用者可能需要处理
                else:
                    logger.warning(f"提取到的内容类型不是字符串: {type(content)}")
                    return ""
            else:
                logger.error(f"从 DeepSeek 收到的响应格式无效: {response}")
                return ""
        except AttributeError as e:
            logger.error(
                f"提取任务内容时出错：Service Manager 可能没有 'get_deepseek_completion' 方法或未正确初始化。错误：{e}",
                exc_info=True,
            )
            return ""
        except Exception as e:
            logger.error(f"提取任务内容时发生意外错误: {str(e)}", exc_info=True)
            return ""

    def compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        使用OpenAI模型计算文本的嵌入向量 (假定 openai_client 是线程安全的)
        """
        if self.openai_client is None:
            logger.error("OpenAI 客户端未初始化，无法计算嵌入向量。")
            return None

        if not text or not isinstance(text, str):
            # logger.debug("输入文本为空或类型不正确，返回 None。") # Can be noisy
            return None
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002",  # Consider making model name configurable
            )
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            # Log specific OpenAI errors if possible
            logger.error(f"使用 OpenAI 计算嵌入向量时出错: {str(e)}", exc_info=True)
            return None

    def visualize_clusters_tsne(
        self,
        perplexity=30,
        n_iter=300,
        figsize=(12, 10),
        point_size=10,
        alpha=0.6,
        title="t-SNE Visualization of Clusters",
    ):
        """使用 t-SNE 可视化聚类结果。"""
        logger.info("\n开始生成 t-SNE 可视化图...")

        # 1. 检查所需数据是否可用
        if self.embeddings is None:
            logger.error(
                "错误: 嵌入向量 (self.embeddings) 不可用，无法进行可视化。请先运行 process_dataset_parallel 或加载数据。"
            )
            return
        if self.raw_cluster_labels is None:
            logger.error(
                "错误: 原始聚类标签 (self.raw_cluster_labels) 不可用，无法进行可视化。请先运行 perform_hierarchical_clustering 或加载数据。"
            )
            return
        if len(self.embeddings) != len(self.raw_cluster_labels):
            logger.error(
                f"错误：嵌入向量数量 ({len(self.embeddings)}) 与原始聚类标签数量 ({len(self.raw_cluster_labels)}) 不匹配，无法可视化。"
            )
            return

        vectors_to_plot = self.embeddings
        labels_to_plot = self.raw_cluster_labels

        # 2. 检查样本数量
        num_samples = len(vectors_to_plot)
        if num_samples <= 1:
            logger.error(f"错误: 样本数量 ({num_samples}) 不足以进行 t-SNE 可视化。")
            return

        # 3. 调整 Perplexity
        adjusted_perplexity = min(perplexity, num_samples - 1)
        if adjusted_perplexity <= 0:
            logger.error(f"错误: 调整后的 Perplexity ({adjusted_perplexity}) 无效。")
            return
        if adjusted_perplexity != perplexity:
            logger.warning(
                f"Perplexity ({perplexity}) 大于样本数减一 ({num_samples - 1}) 或小于等于0，已调整为 {adjusted_perplexity}。"
            )

        # 4. 执行 t-SNE
        logger.info(
            f"正在对 {num_samples} 个向量执行 t-SNE (perplexity={adjusted_perplexity}, n_iter={n_iter})..."
        )
        try:
            tsne = TSNE(
                n_components=2,
                random_state=42,
                max_iter=n_iter,
                perplexity=adjusted_perplexity,
                n_jobs=-1,
                init="pca",
                learning_rate="auto",
            )
            data_tsne = tsne.fit_transform(vectors_to_plot)
            logger.info("t-SNE 计算完成。")
        except Exception as e:
            logger.error(f"执行 t-SNE 时出错: {e}", exc_info=True)
            return

        # 5. 绘制散点图
        logger.info("正在绘制散点图...")
        plt.figure(figsize=figsize)
        try:
            scatter = plt.scatter(
                data_tsne[:, 0],
                data_tsne[:, 1],
                c=labels_to_plot,
                cmap="viridis",
                s=point_size,
                alpha=alpha,
            )
            unique_labels = np.unique(labels_to_plot)
            num_clusters = (
                len(unique_labels)
                if -1 not in unique_labels
                else len(unique_labels) - 1
            )
            plt.title(f"{title} (k={num_clusters})")
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
            if 0 < num_clusters < 30:
                handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label=f"Cluster {i}",
                        markerfacecolor=plt.cm.viridis(i / max(1, num_clusters)),
                        markersize=5,
                    )
                    for i in unique_labels
                    if i != -1
                ]
                plt.legend(
                    handles=handles,
                    title="Clusters",
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                )
            elif num_clusters >= 30:
                cbar = plt.colorbar(scatter, label="Cluster ID")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            save_path = os.path.join(self.intermediate_path, "tsne_visualization.png")
            plt.savefig(save_path, bbox_inches="tight")
            logger.info(f"t-SNE 可视化图已保存到: {save_path}")
        except Exception as e:
            logger.error(f"绘制或保存 t-SNE 图像时出错: {e}", exc_info=True)
        finally:
            plt.close()

    def search_nearest_cluster(self, query_text: str, k: int = 5) -> list[int]:
        """
        查找与查询文本最相似的 k 个簇 (基于与簇质心的距离)。
        """
        # 1. 检查质心是否可用，如果不可用尝试自动聚类
        if self.cluster_centroids is None or not self.cluster_centroids:
            logger.warning("簇质心不可用，尝试自动执行聚类...")
            if self.perform_hierarchical_clustering() is None:  # If clustering fails
                logger.error("自动聚类失败，无法搜索。")
                return []
            if (
                self.cluster_centroids is None or not self.cluster_centroids
            ):  # Check again after clustering
                logger.error("聚类后质心仍然不可用，无法搜索。")
                return []
            logger.info("自动聚类完成，继续搜索。")
        current_centroids = self.cluster_centroids

        # 2. 提取和计算查询嵌入
        query_embedding = None
        try:
            extracted_content = self.extract_task_content(query_text, mode="search")
            if extracted_content:
                query_embedding = self.compute_embedding(extracted_content)
            if query_embedding is None:
                logger.error(f"无法为查询文本计算嵌入向量: '{query_text[:50]}...'")
                return []
        except Exception as e:
            logger.error(f"处理查询文本或计算嵌入时出错: {e}", exc_info=True)
            return []

        # 3. 计算与质心的距离
        distances_to_centroids = []
        for cluster_id, centroid in current_centroids.items():
            try:
                if not isinstance(centroid, np.ndarray):
                    logger.warning(
                        f"簇 {cluster_id} 的质心类型不正确 ({type(centroid)})，跳过。"
                    )
                    continue
                distance = np.linalg.norm(query_embedding - centroid)
                distances_to_centroids.append((distance, cluster_id))
            except Exception as e:
                logger.warning(f"计算与簇 {cluster_id} 质心距离时出错: {e}")
                continue
        if not distances_to_centroids:
            logger.warning("无法计算与任何簇质心的距离。")
            return []

        # 4. 排序并返回结果
        distances_to_centroids.sort(key=lambda x: x[0])
        k = min(k, len(distances_to_centroids))
        nearest_k_clusters = [cid for dist, cid in distances_to_centroids[:k]]
        logger.info(
            f"对于查询 '{query_text[:50]}...' 找到最近的 {k} 个簇标签: {nearest_k_clusters}"
        )
        return nearest_k_clusters

    def search_nearest_taskunit(
        self, query_text: str, k: int = 4, max_task_units: int = 200, min_k: int = 1
    ) -> str:
        """
        查找与查询文本最相似的 task units。
        如果从 k 个簇加载的 task units 超过 max_task_units，则尝试减少 k。
        """
        logger.info(
            f"开始搜索与 '{query_text[:50]}...' 相关的 task units (初始 k={k}, max_units={max_task_units})..."
        )

        if k < min_k:
            logger.warning(f"初始 k ({k}) 小于最小 k ({min_k})，将使用 min_k。")
            k = min_k

        current_k = k
        final_task_units = []
        final_graphs = []
        nearest_clusters = []

        while current_k >= min_k:
            logger.info(f"尝试使用 k={current_k} 搜索簇...")
            # 1. 找到最近的 current_k 个簇
            nearest_clusters = self.search_nearest_cluster(query_text, current_k)
            if not nearest_clusters:
                logger.warning(f"使用 k={current_k} 未找到相近的簇。")
                # If initial k failed, return empty. If reduced k failed, maybe return previous result?
                # Let's return empty for simplicity now.
                return "{}"

            logger.info(
                f"使用 k={current_k} 找到 {len(nearest_clusters)} 个相关簇: {nearest_clusters}"
            )

            # 2. 加载这些簇的 task_unit 和 graph
            all_task_units = []
            all_graphs = []
            task_unit_dir = os.path.join(self.task_unit_base_path, "deepseek")
            graph_dir = os.path.join(self.graph_base_path, "deepseek")

            for cluster_id in nearest_clusters:
                task_unit_path = os.path.join(
                    task_unit_dir, f"cluster_{cluster_id}_task_unit.json"
                )
                graph_path = os.path.join(graph_dir, f"cluster_{cluster_id}_graph.json")

                # 读取 task_unit
                if os.path.exists(task_unit_path):
                    try:
                        with open(task_unit_path, "r", encoding="utf-8") as f:
                            task_unit_data = json.load(f)
                        if (
                            isinstance(task_unit_data, dict)
                            and "task_units" in task_unit_data
                            and isinstance(task_unit_data["task_units"], list)
                        ):
                            all_task_units.extend(task_unit_data["task_units"])
                        # else: logger.warning(...) # Keep logs less verbose inside loop
                    except Exception as e:
                        logger.warning(
                            f"读取或解析 Task unit 文件 {task_unit_path} 时出错: {e}"
                        )

                # 读取 graph (optional based on prompt needs, but good to load)
                if os.path.exists(graph_path):
                    try:
                        with open(graph_path, "r", encoding="utf-8") as f:
                            graph_data = json.load(f)
                        if (
                            isinstance(graph_data, dict)
                            and "task_unit_dag" in graph_data
                            and isinstance(graph_data["task_unit_dag"], list)
                        ):
                            all_graphs.extend(graph_data["task_unit_dag"])
                        # else: logger.warning(...)
                    except Exception as e:
                        logger.warning(
                            f"读取或解析 Graph 文件 {graph_path} 时出错: {e}"
                        )

            # 3. 检查数量是否超限
            num_task_units = len(all_task_units)
            logger.info(f"使用 k={current_k} 加载了 {num_task_units} 个 Task Units。")

            if num_task_units <= max_task_units:
                logger.info(
                    f"Task units 数量 ({num_task_units}) 在限制 ({max_task_units}) 内，使用当前 k={current_k}。"
                )
                final_task_units = all_task_units
                final_graphs = all_graphs  # Use the graphs corresponding to this k
                break  # 找到合适的 k，跳出循环
            else:
                # 超出限制，尝试减小 k
                if current_k > min_k:
                    logger.warning(
                        f"Task units 数量 ({num_task_units}) 超过限制 ({max_task_units})，尝试减小 k。"
                    )
                    current_k -= 1
                    # 继续循环
                else:
                    # 已经是最小 k，仍然超限，只能截断
                    logger.warning(
                        f"即使使用最小 k={min_k}，Task units 数量 ({num_task_units}) 仍然超过限制 ({max_task_units})，将进行截断。"
                    )
                    final_task_units = all_task_units[:max_task_units]
                    # Optionally truncate graphs too, though less critical for prompt size
                    # final_graphs = all_graphs[:max_task_units] # Or some other logic for graphs
                    final_graphs = all_graphs  # Keep all loaded graphs for k=1 for now
                    break  # 必须跳出循环

        # --- 循环结束 ---

        if not final_task_units:
            logger.warning("最终未能加载到任何 task unit 数据。")
            return "{}"

        logger.info(
            f"最终使用 {len(final_task_units)} 个 Task Units (来自 k={current_k} 个簇) 构建 prompt。"
        )

        # 4. 构建 prompt 并调用模型 (使用 final_task_units 和 final_graphs)
        try:
            task_units_str = json.dumps(
                {"task_units": final_task_units}, ensure_ascii=False, indent=2
            )
            # Decide how to handle graphs - use the graphs loaded with the final 'k'
            task_unit_dag_str = json.dumps(
                {"task_unit_dag": final_graphs}, ensure_ascii=False, indent=2
            )
            final_prompt = (
                searching_prompt.replace("{task_description}", query_text)
                .replace("{task_units}", task_units_str)
                .replace("{task_unit_dag}", task_unit_dag_str)
            )
            logger.info(f"final_prompt: {final_prompt}")
            logger.info("正在调用模型分析相关 task units...")
            if self.service_manager is None:
                logger.error("Service manager 未初始化，无法调用模型。")
                return "{}"

            response = self.service_manager.get_deepseek_completion(final_prompt)

            if not response or "choices" not in response or not response["choices"]:
                logger.error(f"模型响应格式不正确: {response}")
                return "{}"

            content = response["choices"][0]["message"]["content"]
            # ... (JSON validation logic remains the same) ...
            try:
                json.loads(content)
                logger.info(f"模型返回的JSON: {response}")
                return content
            except json.JSONDecodeError:
                logger.error(f"模型返回的不是有效的JSON: {content}")
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    try:
                        json.loads(match.group(0))
                        logger.warning("提取并返回了模型响应中的 JSON 部分。")
                        return match.group(0)
                    except json.JSONDecodeError:
                        logger.error("提取的部分仍然不是有效的 JSON。")
                return json.dumps(
                    {
                        "relevant_task_units": [],
                        "reason": "Model response was not valid JSON",
                    }
                )

        except AttributeError as e:
            logger.error(
                f"调用模型时出错：Service Manager 可能没有 'get_deepseek_completion' 方法或未正确初始化。错误：{e}",
                exc_info=True,
            )
            return "{}"
        except Exception as e:
            logger.error(f"调用模型或处理结果时发生错误: {e}", exc_info=True)
            return "{}"

    def _ensure_df_loaded(self) -> bool:
        """内部方法：确保 self.df 已加载，如果需要的话。"""
        if self.df is not None:
            return True  # Already loaded (likely passed in __init__)

        if self.dataset_path:
            logger.info("DataFrame 未加载，尝试从路径加载...")
            try:
                self.df = self.load_dataset(self.dataset_path)
                if self.df is not None:
                    self._create_df_index_mapping()  # Create map after loading
                    return True
                else:
                    logger.error("尝试从路径加载 DataFrame 失败。")
                    return False
            except Exception as e:
                logger.error(
                    f"尝试从路径 {self.dataset_path} 加载 DataFrame 时发生意外错误: {e}",
                    exc_info=True,
                )
                return False
        else:
            logger.error(
                "错误：需要 DataFrame，但未在初始化时提供，且未设置 dataset_path。"
            )
            return False
