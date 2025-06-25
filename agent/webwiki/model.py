from urllib import parse
from typing import List
import re


class Node:
    """存储每个page的信息"""

    def __init__(
        self, url: str, title=None, parent=None, snapshot_path=None, content=None
    ):
        self.url, self.normalized_parts = self._normalize_url_with_parts(url)
        self.title = title
        self.parent = parent
        self.children = {}
        self.visited = False
        self.snapshot_path = snapshot_path
        self.content = content
        self.depth = 0 if parent is None else parent.depth + 1

    @staticmethod
    def _normalize_url(url: str) -> str:
        """规范化URL，移除参数和锚点"""
        parsed = parse.urlparse(url)
        # 重建URL，但不包含参数和锚点
        normalized = parse.urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                "",  # params
                "",  # query
                "",  # fragment
            )
        )
        return normalized.rstrip("/")

    @staticmethod
    def _normalize_url_with_parts(url: str):
        parsed = parse.urlparse(url)
        norm_path, normalized_parts = normalize_url_path_with_parts(parsed.path)
        normalized = parse.urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                norm_path,
                "",  # params
                "",  # query
                "",  # fragment
            )
        )
        return normalized.rstrip("/"), normalized_parts

    @staticmethod
    def _format_title(title: str) -> str:
        if not title:
            return ""
        format = title.strip()
        format = re.sub(r"[^a-zA-Z0-9\s]", "", format)
        format = " ".join(format.split())
        return format

    def __repr__(self) -> str:
        format_sitemap = f"The sitemap of current or nearest parent page: {Tree.format_url(self)}\n\n"

        if self.content is not None:
            page_summary = self.content.get("page_summary", None)
            key_user_flows = self.content.get("key_user_flows", None)
            block_content = self.content.get("blocks", None)

            if page_summary:
                format_sitemap += f"## Current Page-Level Summary\n\n{page_summary}\n\n"
                if key_user_flows:
                    format_sitemap += f"### Key User Flows\n\n"
                    for key_user_flow in key_user_flows:
                        format_sitemap += f"* {key_user_flow}\n"

                    format_sitemap += "\n"

                if block_content:
                    format_sitemap += f"## Current Page Block-Level Content\n\n"
                    for item in block_content:
                        format_sitemap += f"### {item['name']}\nContent: {item['content']}\nPosition: {item['vertical_position']}\nInteraction Guide: {item['interaction_guide']}\n\n"

        # if self.children:
        #     format_sitemap += f"## Available Links from Current Page\n"
        #     childs = sorted(self.children.values(), key = lambda x: len(x.children), reverse = True)

        #     for child in childs[:10]:
        #         format_sitemap += f"* {Tree.format_url(child)}\n"
        #         if child.title:
        #             format_sitemap += f"  - Title: {child.title}\n"
        #         if child.content and child.content.get('page_summary'):
        #             format_sitemap += f"  - Page Summary: {child.content['page_summary']}\n"
        #         if child.content and child.content.get('key_user_flows'):
        #             format_sitemap += f"  - Key User Flows: {child.content['key_user_flows']}\n"
        #         format_sitemap += "\n"

        # format_sitemap += "**Thinking carefully according to the page summary and key user flows to go to the most likely page to complete the task.**\n"

        # format_sitemap += "--------------------------------\n"

        return format_sitemap


class Tree:
    def __init__(self, base_url: str, domain: str = None):
        self.base_url = Node._normalize_url(base_url)
        self.root = Node(self.base_url)
        self.all_nodes = [self.root]
        self.base_domain = domain

    def add_nodes(
        self,
        url: str,
        title: str = None,
    ):
        # Parse and normalize URL
        parsed_url = parse.urlsplit(url)
        current_domain = parsed_url.netloc.split("www.")[-1]
        url = Node._normalize_url(url)

        # 如果URL已经在树中，返回
        if self._find_node(self.root, url):
            return 0

        # 获取路径组件
        path_parts = [p for p in parsed_url.path.split("/") if p.strip()]

        # 确定父节点
        added_nodes = 0
        if current_domain == self.base_domain:
            # 如果是主域名下的页面
            if not path_parts:
                # 如果是主域名本身，不需要添加
                return 0

            parent_node = self.root
            # 逐级构建路径
            current_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            for part in path_parts[:-1]:
                current_url = f"{current_url}/{part}"
                node = self._find_node(self.root, current_url)

                if not node:
                    node = Node(current_url, parent=parent_node)
                    parent_node.children[current_url] = node
                    self.all_nodes.append(node)
                    added_nodes += 1
                parent_node = node
                # if self.score_function(node) <= 0:
                #     return added_nodes
        else:
            parent_node = self.root
            if path_parts:
                current_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                for part in path_parts[:-1]:
                    current_url = f"{current_url}/{part}"
                    node = self._find_node(self.root, current_url)

                    if not node:
                        node = Node(current_url, parent=parent_node)
                        parent_node.children[current_url] = node
                        self.all_nodes.append(node)
                        added_nodes += 1
                    parent_node = node

        if url in parent_node.children:
            return added_nodes

        new_node = Node(url, title=title, parent=parent_node)
        parent_node.children[url] = new_node
        self.all_nodes.append(new_node)
        added_nodes += 1

        return added_nodes

    def mark_visited(self, url: str):
        url = url.rstrip("/")
        node = self._find_node(self.root, url)
        if node:
            node.visited = True

    def _find_node(self, current_node: Node, target_url: str):
        url = self.format_url(current_node)
        if url == target_url:
            return current_node

        for child in current_node.children.values():
            result = self._find_node(child, target_url)
            if result:
                return result

        return None

    def get_next_urls(self, max_urls: int = 10):
        non_visited_nodes = [node for node in self.all_nodes if not node.visited]
        node_scores = [self.score_function(node) for node in non_visited_nodes]
        sorted_pairs = [
            (node, score)
            for node, score in sorted(
                zip(non_visited_nodes, node_scores), key=lambda x: x[1], reverse=True
            )
        ]

        sorted_pairs = [pair for pair in sorted_pairs if pair[1] >= 0]
        return [self.format_url(pair[0]) for pair in sorted_pairs[:max_urls]]

    def score_function(self, node, threshold: int = 2):
        return self._score_function(node, threshold=threshold)

    def _score_function(self, node, threshold: int = 2):
        if node.depth >= threshold:
            return -1
        return 100 - node.depth

    def select_unrepeated_nodes(self, use_title: bool = True) -> List[Node]:
        """
        Select nodes that have unique URL patterns by checking normalized_parts.
        Returns nodes without normalized_parts and one representative node for each pattern.
        """
        unique_patterns = set()
        filtered_result = []
        if use_title:
            for node in self.all_nodes:
                pattern = (node.url, node.title)
                if pattern not in unique_patterns:
                    unique_patterns.add(pattern)
                    filtered_result.append(node)
        else:
            for node in self.all_nodes:
                pattern = node.url
                if pattern not in unique_patterns:
                    unique_patterns.add(pattern)
                    filtered_result.append(node)
        return filtered_result

    def save_to_json(self, file_path):
        # 嵌套结构储存，保存所有节点信息到json
        import json

        def node_to_dict(node: Node):
            return {
                "url": node.url,
                "title": node.title,
                "visited": node.visited,
                "depth": node.depth,
                "children": {
                    url: node_to_dict(child) for url, child in node.children.items()
                },
                "content": node.content,
                "snapshot_path": node.snapshot_path,
                "normalized_parts": node.normalized_parts,
            }

        tree_data = {"base_url": self.base_url, "root": node_to_dict(self.root)}

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(tree_data, f, ensure_ascii=False, indent=4)

    @classmethod
    def load_from_json(cls, file_path):
        import json

        def dict_to_node(node_dict, parent=None):
            node = Node(
                url=node_dict["url"],
                title=node_dict["title"],
                parent=parent,
                snapshot_path=(
                    node_dict["snapshot_path"] if "snapshot_path" in node_dict else None
                ),
                content=node_dict["content"] if "content" in node_dict else None,
            )
            node.visited = node_dict["visited"]
            node.depth = node_dict["depth"]
            node.normalized_parts = node_dict["normalized_parts"]

            for url, child_dict in node_dict["children"].items():
                child_node = dict_to_node(child_dict, parent=node)
                node.children[url] = child_node
            return node

        with open(file_path, "r", encoding="utf-8") as f:
            tree_data = json.load(f)

        tree = cls(tree_data["base_url"])

        # 重建根节点及其所有子节点
        tree.root = dict_to_node(tree_data["root"])

        # 重建 all_nodes 列表
        tree.all_nodes = []

        def collect_nodes(node):
            tree.all_nodes.append(node)
            for child in node.children.values():
                collect_nodes(child)

        collect_nodes(tree.root)
        return tree

    @classmethod
    def format_url(self, node: Node):
        return node.url.format(
            **{part["type"]: part["value"] for part in node.normalized_parts}
        )


def normalize_url_path_with_parts(path):
    normalized_parts = []

    # UUID
    def uuid_repl(match):
        normalized_parts.append({"type": "uuid", "value": match.group(0)})
        return "{uuid}"

    path = re.sub(
        r"[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}",
        uuid_repl,
        path,
    )

    # 年份 (移到数字匹配之前)
    def year_repl(match):
        normalized_parts.append({"type": "year", "value": match.group(0)})
        return "{year}"

    path = re.sub(r"(?<=/)(19|20)\d{2}(?=/|\.|$)", year_repl, path)

    # 长数字
    def num_repl(match):
        normalized_parts.append({"type": "num", "value": match.group(0)})
        return "{num}"

    path = re.sub(r"(?<=/)\d{4,}(?=/|$)", num_repl, path)

    # hash
    def hash_repl(match):
        normalized_parts.append({"type": "hash", "value": match.group(0)})
        return "{hash}"

    path = re.sub(r"(?<=/)[a-fA-F0-9]{16,}(?=/|$)", hash_repl, path)

    # 字母+数字
    def alphanum_repl(match):
        normalized_parts.append({"type": "alphanum", "value": match.group(0)})
        return "{alphanum}"

    path = re.sub(r"(?<=/)[a-zA-Z]\d+[a-zA-Z0-9]*(?=/|$)", alphanum_repl, path)

    # 数字+字母
    def numalpha_repl(match):
        normalized_parts.append({"type": "numalpha", "value": match.group(0)})
        return "{numalpha}"

    path = re.sub(r"(?<=/)\d+[a-zA-Z]+[a-zA-Z0-9]*(?=/|$)", numalpha_repl, path)

    # 短数字
    def short_num_repl(match):
        normalized_parts.append({"type": "short_num", "value": match.group(0)})
        return "{short_num}"

    path = re.sub(r"(?<=/)\d{1,3}(?=/|$)", short_num_repl, path)

    # 连续重复字符
    def repeated_repl(match):
        normalized_parts.append({"type": "repeated", "value": match.group(0)})
        return "{repeated}"

    path = re.sub(r"(?<=/)([a-zA-Z0-9])\1+(?=/|$)", repeated_repl, path)

    return path, normalized_parts
