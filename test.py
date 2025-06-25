DEEPSEEK_API_KEY = "sk-99dc62cf7a0e4016bd01a79249b195f9"
import json
import logging
import asyncio
import os

from langchain_openai import ChatOpenAI
from agent.service import Agent
from pydantic import BaseModel
from typing import List
from controller.service import Controller
from browser.browser import Browser, BrowserConfig
from browser.context import BrowserContext
from openai import OpenAI
from tqdm import tqdm
from agent.webwiki.loader import WebWiki

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

controller = Controller()

llm = ChatOpenAI(
    model="/fs/fast/u2023100837/Qwen2.5-VL-72B-Instruct",
    temperature=0,
    base_url="http://10.10.252.11:23041/v1",
    api_key="empty",
    max_tokens=4096,
)

world_model_client = OpenAI(base_url="http://10.10.252.11:23041/v1", api_key="empty")


async def main():
    tasks = {
        "online_mind2web": "/home/douzc/users/yuyao/browser_use_naive/datasets/online_mind2web_tasks.json",
        "webvoyager": "/home/douzc/users/yuyao/browser_use_naive/datasets/webvoyager_tasks.json",
        "webwalker": "/home/douzc/users/yuyao/browser_use_naive/datasets/webwalker_final_tasks.json",
        "webwalker_edu": "/home/douzc/users/yuyao/browser_use_naive/datasets/top_12_education_tasks.json",
        "webwalker_conf": "/home/douzc/users/yuyao/browser_use_naive/datasets/top_15_conference_tasks.json",
    }
    dataset = "webvoyager"
    task_path = tasks[dataset]
    log_dir = f"/home/douzc/users/yuyao/browser_use_naive/logs/{dataset}/qwen2.5_72b_vl_react_webwiki_v1"
    # log_dir = f'/home/zyy/browser_use_naive/logs/{dataset}/qwen2.5_72b_vl_v1'

    with open(task_path, "r") as f:
        tasks = json.load(f)

    # Load WebWiki
    webwiki = WebWiki(
        meta_data_file_path=f"/home/douzc/users/yuyao/browser_use_naive/datasets/meta/{dataset}.json",
        cache_file_path=f"/home/douzc/users/yuyao/browser_use_naive/cache/webwiki_{dataset}.json",
    )

    tasks = [
        task
        for task in tasks
        if not os.path.exists(f"{log_dir}/{task['task_id']}/result.json")
    ]
    tasks = tasks[45:]

    # Run tasks
    for id in tqdm(range(0, len(tasks))):
        try:
            task = tasks[id]["confirmed_task"]
            task_target_url = tasks[id]["website"]
            task_id = tasks[id]["task_id"]

            initial_actions = [{"go_to_url": {"url": task_target_url}}]

            if os.path.exists(f"{log_dir}/{task_id}/result.json"):
                logger.info(f"Task {task_id} already exists, skipping")
                continue

            webwiki.load_sitemap(url=task_target_url)

            agent = Agent(
                task=task,
                task_target_url=task_target_url,
                llm=llm,
                webwiki=webwiki,
                use_vision=True,
                controller=controller,
                max_failures=5,
                save_conversation_path=f"{log_dir}/{task_id}/{task_id}",
                initial_actions=initial_actions,
                max_actions_per_step=1,
                use_react=True if "react" in log_dir else False,
                use_webwiki=True if "webwiki" in log_dir else False,
                use_world_model=True if "world_model" in log_dir else False,
                world_model_client=world_model_client,
            )
            _ = await agent.run(max_steps=20)

        except Exception as e:
            logger.error(f"Error running agent: {e}", exc_info=True)


asyncio.run(main())
