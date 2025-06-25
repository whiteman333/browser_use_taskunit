from agent.webwiki.model import Node, Tree
from openai import OpenAI
import json
import logging
import numpy as np
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebWiki:

    def __init__(
        self,
        meta_data_file_path: str,
        api_key: str = "sk-j7XqJIWtO34tdPAa249f980f1d2145CdA0B594Fa55094dFd",
        base_url: str = "https://aihubmix.com/v1",
        embedding_model: str = "jina-embeddings-v3",
        cache_file_path: str = "",
    ):
        with open(meta_data_file_path, "r") as f:
            data = json.load(f)

        self.metadata = {d["url"]: d for d in data}

        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.embedding_model = embedding_model

        self._tree = None
        self._semantics_cache = {}
        self._cache_file_path = cache_file_path
        self._load_cache(cache_file_path)

    def load_sitemap(self, url: str):
        url = Node._normalize_url(url)

        meta_data = self.metadata.get(url, None)

        if meta_data is None:
            logger.warning(f"No sitemap found for {url}")
            self._tree = None
        else:
            self._tree = Tree.load_from_json(meta_data["tree_file"])

        self._embeddings = np.load(meta_data["embedding_file"])

    def retrieve_from_url(self, url: str):
        url = Node._normalize_url(url)

        if self._tree is None:
            return None

        candidates = []

        for node in self._tree.all_nodes:
            node_url = self._tree.format_url(node)
            if node_url == url:
                return node

        for node in self._tree.all_nodes:
            node_url = self._tree.format_url(node)
            parts1 = node_url.split("/")
            parts2 = url.split("/")
            similarity = len(set(parts1) & set(parts2)) / len(set(parts1) | set(parts2))
            candidates.append((node, similarity))

        if candidates:
            for node, similarity in candidates:
                if node.content and node.content.get("page_summary"):
                    return node

            return None

    def retrieve_from_semantics(self, task: str, topk: int = 5):
        if self._tree is None:
            return None

        cache_key = task
        if cache_key in self._semantics_cache:
            return self._semantics_cache[cache_key]

        try:
            task_embedding = (
                self.client.embeddings.create(
                    model=self.embedding_model,
                    input="Show me the page that is most related to the task: " + task,
                )
                .data[0]
                .embedding
            )
        except Exception as e:
            logger.error(f"Error embedding task: {e}")
            return None

        # Calculate cosine similarity
        # all_embeddings = np.array(self._embeddings['embeddings'])[1:, :]
        # all_ids = self._embeddings['ids'][1:]
        all_embeddings = np.array(self._embeddings["embeddings"])
        all_ids = self._embeddings["ids"]

        dot_products = np.dot(all_embeddings, task_embedding)
        task_norm = np.linalg.norm(task_embedding)

        norms = np.linalg.norm(all_embeddings, axis=1)
        similarities_array = dot_products / (task_norm * norms)

        similarities = list(zip(all_ids, similarities_array))
        similarities.sort(key=lambda x: x[1], reverse=True)

        format_output = "## Retrieved Pages that may be related to the task\n\n"

        for id, _ in similarities[:topk]:
            node = self._tree.all_nodes[id]
            node_url = self._tree.format_url(node)
            format_output += f"### Page: {node_url}\nTitle: {node.title}\n"
            if node.content and node.content.get("page_summary"):
                format_output += f"Summary: {node.content['page_summary']}\n\n"
            else:
                format_output += "\n"

        format_output += "**Hints**: You should think first if you can use 'go_to_url' action to navigate quickly to the task-releated page retrieved above, and then use other interaction actions (click, input_text, etc.) to complete the task.\n"
        format_output += (
            "**Stop Using 'scroll_down' if you have met [End of page] !!**\n"
        )

        self._semantics_cache[cache_key] = format_output

        if self._cache_file_path is not None:
            os.makedirs(os.path.dirname(self._cache_file_path), exist_ok=True)
            with open(self._cache_file_path, "w", encoding="utf-8") as f:
                json.dump(self._semantics_cache, f, ensure_ascii=False, indent=4)

        return format_output

    def retrieve_all_links(self, use_title: bool = True):
        if self._tree is None:
            return None

        links = "# Available Links from this website\n"

        add_urls = set()

        if self._tree.all_nodes:
            # Group children by template URL
            url_groups = {}
            for node in self._tree.all_nodes:
                template_url = node.url
                if template_url not in url_groups:
                    url_groups[template_url] = []
                url_groups[template_url].append(node)

            # Sort template URLs
            sorted_urls = sorted(url_groups.keys())

            # Output each template URL and its parts
            for template_url in sorted_urls:
                if template_url not in add_urls:
                    add_urls.add(template_url)
                    if len(url_groups[template_url]) == 1:
                        links += f"{template_url}"
                        if url_groups[template_url][0].title:
                            links += (
                                f" (Page title: {url_groups[template_url][0].title})"
                            )
                        links += "\n"
                    else:
                        links += f"{template_url}\n"

                        for child in url_groups[template_url]:
                            part_info = ""
                            if use_title:
                                if child.title:
                                    for part in child.normalized_parts:
                                        part_info += (
                                            f"[{part['type']}]: {part['value']}\t"
                                        )
                                    part_info += f" (page title: {child.title})"
                                    part_info += "\n"
                                    links += part_info
                            else:
                                for part in child.normalized_parts:
                                    part_info += f"[{part['type']}]: {part['value']}\t"
                                if child.title:
                                    part_info += f" (page title: {child.title})"
                                    part_info += "\n"
                                links += part_info
        links += "\n"
        links += "**TIPS**: you can use the links to navigate to the task-releated page by 'go_to_url' action.\n"

        return links

    def retrieve_important_links(self, threshold: int = 2):
        if self._tree is None:
            return None

        nodes = [
            node
            for node in self._tree.all_nodes
            if self._tree.score_function(node, threshold=threshold) > 0
        ]

        format_output = ""

        for node in nodes:
            format_output += (
                f"URL: {self._tree.format_url(node)}\nTitle: {node.title}\n"
            )
            if node.content and node.content.get("page_summary"):
                format_output += f"Summary: {node.content['page_summary']}\n\n"
            else:
                format_output += "\n"

        return format_output

    def reset(self):
        self._tree = None
        self._embeddings = None
        self._semantics_cache = {}

    def _load_cache(self, cache_file_path: str):
        if os.path.exists(cache_file_path):
            with open(cache_file_path, "r") as f:
                self._semantics_cache = json.load(f)
        else:
            self._semantics_cache = {}


if __name__ == "__main__":
    webwiki = WebWiki()
    webwiki.load_sitemap(url="http://www.cs.zju.edu.cn/csen")
    print(webwiki.retrieve_all_links())
