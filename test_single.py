import os

os.environ["https_proxy"] = "http://127.0.0.1:7895"
os.environ["http_proxy"] = "http://127.0.0.1:7895"
os.environ["OPENAI_API_KEY"] = (
    "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"
)
os.environ["base_url"] = "https://openrouter.ai/api/v1"

from langchain_openai import ChatOpenAI
from agent.service import Agent
from pydantic import BaseModel
from typing import List
from controller.service import Controller
from browser.browser import Browser, BrowserConfig
from browser.context import BrowserContext
from openai import OpenAI
import json
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

controller = Controller()

llm = ChatOpenAI(
    # model = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8",
    model="/home/zyy/LLMs/Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
    temperature=0,
    base_url="http://localhost:12111/v1",
    api_key="empty",
    max_tokens=512,
)

world_model_client = OpenAI(base_url="http://localhost:12111/v1", api_key="empty")
task_id = ["Allrecipes--1"]


async def main():

    tasks = json.load(open("/home/zyy/webvoyager/data/formatted_WebVoyager_data.json"))
    task = None
    for id in range(0, len(tasks)):
        if tasks[id]["task_id"] in task_id:
            this_id = tasks[id]["task_id"]
            task = tasks[id]
            initial_actions = [{"go_to_url": {"url": task["website"]}}]
            if os.path.exists(
                f"/home/zyy/browser_use_naive/logs/test/20250506/{this_id}/result.json"
            ):
                logger.info(f"Task {this_id} already exists, skipping")
                continue
            agent = Agent(
                world_model_client=world_model_client,
                task_target_url=task["website"],
                task=task["confirmed_task"],
                llm=llm,
                use_vision=False,
                controller=controller,
                max_failures=3,
                save_conversation_path=f"/home/zyy/browser_use_naive/logs/test/20250506/{this_id}/{this_id}",
                initial_actions=initial_actions,
                max_actions_per_step=1,
                use_world_model=True,
                use_react=True,
            )

            await agent.run(max_steps=30)


asyncio.run(main())
