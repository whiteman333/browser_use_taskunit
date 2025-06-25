from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Optional

from dom.history_tree_processor.views import (
    CoordinateSet,
    HashedDomElement,
    ViewportInfo,
)

# Avoid circular import issues
if TYPE_CHECKING:
    from .views import DOMElementNode


@dataclass(frozen=False)
class DOMBaseNode:
    is_visible: bool  # 是否可见
    parent: Optional["DOMElementNode"]  # 双亲节点


@dataclass(frozen=False)
class DOMTextNode(DOMBaseNode):
    text: str  # 文本内容
    type: str = "TEXT_NODE"  # 节点类型

    def has_parent_with_highlight_index(self) -> bool:
        current = self.parent
        while current is not None:
            if current.highlight_index is not None:
                return True
            current = current.parent
        return False


@dataclass(frozen=False)
class DOMElementNode(DOMBaseNode):
    """
    xpath: the xpath of the element from the last root node (shadow root or iframe OR document if no shadow root or iframe).
    To properly reference the element we need to recursively switch the root node until we find the element (work you way up the tree with `.parent`)
    """

    tag_name: str  # 标签名
    xpath: str  # xpath路径
    attributes: Dict[str, str]  # 属性
    children: List[DOMBaseNode]  # 子节点
    is_interactive: bool = False  # 是否可交互
    is_top_element: bool = False  # 是否是顶部元素
    shadow_root: bool = False  # 是否是影子根节点
    highlight_index: Optional[int] = None  # 高亮索引
    viewport_coordinates: Optional[CoordinateSet] = None  # 视图坐标
    page_coordinates: Optional[CoordinateSet] = None  # 页面坐标
    viewport_info: Optional[ViewportInfo] = None  # 视图信息

    def __repr__(self) -> str:
        """
        重写repr方法，用于表示DOMElementNode对象
        """
        tag_str = f"<{self.tag_name}"

        # Add attributes
        for key, value in self.attributes.items():
            tag_str += f' {key}="{value}"'
        tag_str += ">"

        # Add extra info
        extras = []
        if self.is_interactive:
            extras.append("interactive")
        if self.is_top_element:
            extras.append("top")
        if self.shadow_root:
            extras.append("shadow-root")
        if self.highlight_index is not None:
            extras.append(f"highlight:{self.highlight_index}")

        if extras:
            tag_str += f' [{", ".join(extras)}]'

        return tag_str

    @cached_property
    def hash(self) -> HashedDomElement:
        from dom.history_tree_processor.service import (
            HistoryTreeProcessor,
        )

        return HistoryTreeProcessor._hash_dom_element(self)

    def get_all_text_till_next_clickable_element(self, max_depth: int = -1) -> str:
        """
        获取从当前节点到下一个可点击元素之间的所有文本
        """
        text_parts = []

        def collect_text(node: DOMBaseNode, current_depth: int) -> None:
            """
            收集文本
            """
            if max_depth != -1 and current_depth > max_depth:
                return

            # Skip this branch if we hit a highlighted element (except for the current node)
            if (
                isinstance(node, DOMElementNode)
                and node != self
                and node.highlight_index is not None
            ):
                return

            if isinstance(node, DOMTextNode):
                text_parts.append(node.text)
            elif isinstance(node, DOMElementNode):
                for child in node.children:
                    collect_text(child, current_depth + 1)

        collect_text(self, 0)
        return "\n".join(text_parts).strip()

    def clickable_elements_to_string(self, include_attributes: list[str] = []) -> str:
        """
        将可点击的DOM元素转换为字符串
        """
        formatted_text = []

        def process_node(node: DOMBaseNode, depth: int) -> None:
            """
            处理节点的函数
            如果节点是DOMElementNode，则添加元素；只添加有高亮索引的元素；只添加在include_attributes中的属性
            如果节点是DOMTextNode，则添加文本；只添加没有高亮父节点的文本，并且[]内没有索引
            """
            if isinstance(node, DOMElementNode):
                # Add element with highlight_index
                if node.highlight_index is not None:
                    attributes_str = ""
                    if include_attributes:
                        attributes_str = " " + " ".join(
                            f'{key}="{value}"'
                            for key, value in node.attributes.items()
                            if key in include_attributes
                        )
                    formatted_text.append(
                        f"[{node.highlight_index}]<{node.tag_name}{attributes_str}>{node.get_all_text_till_next_clickable_element()}</{node.tag_name}>"
                    )

                # Process children regardless
                for child in node.children:
                    process_node(child, depth + 1)

            elif isinstance(node, DOMTextNode):
                # Add text only if it doesn't have a highlighted parent
                if not node.has_parent_with_highlight_index():
                    formatted_text.append(f"[]{node.text}")

        process_node(self, 0)
        return "\n".join(formatted_text)

    def get_file_upload_element(
        self, check_siblings: bool = True
    ) -> Optional["DOMElementNode"]:
        # Check if current element is a file input
        if self.tag_name == "input" and self.attributes.get("type") == "file":
            return self

        # Check children
        for child in self.children:
            if isinstance(child, DOMElementNode):
                result = child.get_file_upload_element(check_siblings=False)
                if result:
                    return result

        # Check siblings only for the initial call
        if check_siblings and self.parent:
            for sibling in self.parent.children:
                if sibling is not self and isinstance(sibling, DOMElementNode):
                    result = sibling.get_file_upload_element(check_siblings=False)
                    if result:
                        return result

        return None

    def get_advanced_css_selector(self) -> str:
        from browser.context import BrowserContext

        return BrowserContext._enhanced_css_selector_for_element(self)


SelectorMap = dict[int, DOMElementNode]


@dataclass
class DOMState:
    element_tree: DOMElementNode
    selector_map: SelectorMap
