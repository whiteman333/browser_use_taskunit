from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from model import Node, Tree
import argparse
import base64
import json
import os
from tqdm import tqdm
from prompt import PAGE_SUMMARIZER_PROMPT

client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1",
)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_json_from_response(response):
    try:
        content = response.choices[0].message.content
        try:
            json_obj = json.loads(content)
            if "error" in content:
                json_obj["page_summary"] = None
            return json_obj
        except json.JSONDecodeError:
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
                json_obj = json.loads(json_str)
                if "error" in content:
                    json_obj["page_summary"] = None
                return json_obj
            elif "```" in content:
                # Try to extract from generic code block
                json_str = content.split("```")[1].split("```")[0]
                json_obj = json.loads(json_str)
                if "error" in content:
                    json_obj["page_summary"] = None
                return json_obj
            else:
                return None
    except Exception as e:
        print(f"Error extracting JSON: {str(e)}")
        return None


def summarize_single_page(node: Node):
    img_path = node.snapshot_path
    url = node.url
    try:
        if img_path:
            response = client.chat.completions.create(
                model="openai/gpt-4o-2024-11-20",
                messages=[
                    {"role": "system", "content": PAGE_SUMMARIZER_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Here is the screenshot of the webpage with url: {url}",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encode_image(img_path)}"
                                },
                            },
                        ],
                    },
                ],
            )
            node.content = extract_json_from_response(response)
            return True
        else:
            node.content = {"page_summary": None, "blocks": []}
            return False
    except Exception as e:
        print(f"Error summarizing page: {str(e)}")
        node.content = {"page_summary": None, "blocks": []}
        return False


def main():
    parser = argparse.ArgumentParser(description="Summarize web pages from URL tree")

    parser.add_argument(
        "--max_workers",
        type=int,
        default=15,
        help="Maximum number of concurrent workers",
    )
    # /home/zyy/web_analyzer/output/low_count_sites.json
    parser.add_argument(
        "--file_paths",
        type=str,
        default="/home/zyy/web_analyzer/webvoyager_output/meta/successful_urls_v2.json",
        help="Path to the URL tree file",
    )

    args = parser.parse_args()

    with open(args.file_paths, "r") as f:
        data = json.load(f)

    file_paths = [d["tree_file"] for d in data]

    for file_path in file_paths:
        if os.path.exists(file_path):
            print(f"Summarizing pages from {file_path}")
            tree = Tree.load_from_json(file_path)
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_to_task = {
                    executor.submit(summarize_single_page, node): node
                    for node in tree.all_nodes
                }
                with tqdm(total=len(tree.all_nodes), desc="Summarizing pages") as pbar:
                    for future in as_completed(future_to_task):
                        pbar.update(1)
        else:
            with open("failed_paths.txt", "a") as f:
                f.write(f"{file_path}\n")

            print(f"File not found: {file_path}")

        tree.save_to_json(file_path)


if __name__ == "__main__":
    main()
