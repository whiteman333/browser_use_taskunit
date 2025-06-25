from __future__ import annotations

from curses import use_default_colors
import json
import logging
from typing import Dict, List, Optional, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agent.message_manager.views import MessageHistory, MessageMetadata
from agent.prompt import AgentMessagePrompt, SystemPrompt
from agent.views import ActionResult, AgentOutput, AgentStepInfo
from browser.views import BrowserState
from agent.webwiki.loader import WebWiki

logger = logging.getLogger(__name__)


class MessageManager:
    def __init__(
        self,
        llm: BaseChatModel,
        task: str,
        action_descriptions: str,
        system_prompt_class: Type[SystemPrompt],
        max_input_tokens: int = 128000,
        estimated_characters_per_token: int = 3,
        image_tokens: int = 800,
        include_attributes: list[str] = [],
        max_error_length: int = 400,
        max_actions_per_step: int = 10,
        message_context: Optional[str] = None,
        sensitive_data: Optional[Dict[str, str]] = None,
        reference_task_units: str = None,
        webwiki: WebWiki = None,
        use_world_model: bool = False,
        use_react: bool = False,
    ):
        self.llm = llm
        self.task = task
        self.action_descriptions = action_descriptions
        self.system_prompt_class = system_prompt_class
        self.max_input_tokens = max_input_tokens
        self.estimated_characters_per_token = estimated_characters_per_token
        self.IMG_TOKENS = image_tokens
        self.include_attributes = include_attributes

        self.max_error_length = max_error_length
        self.message_context = message_context
        self.sensitive_data = sensitive_data
        self.webwiki = webwiki

        self.history = MessageHistory()

        # 添加 system prompt
        system_message = self.system_prompt_class(
            self.action_descriptions,
            max_actions_per_step=max_actions_per_step,
            reference_task_units=reference_task_units,
            use_world_model=use_world_model,
            use_react=use_react,
        ).get_system_message()
        self._add_message_with_tokens(system_message)

        self.system_prompt = system_message

        if self.message_context:
            context_message = HumanMessage(
                content="Context for the task" + self.message_context
            )
            self._add_message_with_tokens(context_message)

        # 添加 task prompt
        task_message = self.task_instructions(task)
        self._add_message_with_tokens(task_message)

        if self.sensitive_data:
            info = f"Here are placeholders for sensitve data: {list(self.sensitive_data.keys())}"
            info += "To use them, write <secret>the placeholder name</secret>"
            info_message = HumanMessage(content=info)
            self._add_message_with_tokens(info_message)

        placeholder_message = HumanMessage(content="Example output:")
        self._add_message_with_tokens(placeholder_message)

        # 添加 tool prompt
        self.tool_id = 1
        tool_calls = [
            {
                "name": "AgentOutput",
                "args": {
                    "current_state": {
                        "page_summary": "On the page are company a,b,c wtih their revenue 1,2,3.",
                        "evaluation_previous_goal": "Success - I opend the first page",
                        "memory": "Starting with the new task. I have completed 1/10 steps",
                        "next_goal": "Click on company a",
                    },
                    "action": [{"click_element": {"index": 0}}],
                },
                "id": str(self.tool_id),
                "type": "tool_call",
            }
        ]

        example_tool_call = AIMessage(
            content=f"",
            tool_calls=tool_calls,
        )

        self._add_message_with_tokens(example_tool_call)
        tool_message = ToolMessage(
            content=f"Browser started",
            tool_call_id=str(self.tool_id),
        )
        self._add_message_with_tokens(tool_message)

        self.tool_id += 1

        placeholder_message = HumanMessage(
            content="[Your task history memory starts here]"
        )
        self._add_message_with_tokens(placeholder_message)

    @staticmethod
    def task_instructions(task: str) -> HumanMessage:
        content = f'Your ultimate task is: """{task}""". If you achieved your ultimate task, stop everything and use the done action in the next step to complete the task. If not, continue as usual.'
        return HumanMessage(content=content)

    def add_file_paths(self, file_paths: list[str]) -> None:
        content = f"Here are file paths you can use: {file_paths}"
        msg = HumanMessage(content=content)
        self._add_message_with_tokens(msg)

    def add_new_task(self, new_task: str) -> None:
        content = f'Your new ultimate task is: """{new_task}""". Take the previous context into account and finish your new ultimate task. '
        msg = HumanMessage(content=content)
        self._add_message_with_tokens(msg)

    def add_plan(self, plan: Optional[str], position: Optional[int] = None) -> None:
        if plan:
            msg = AIMessage(content=plan)
            self._add_message_with_tokens(msg, position)

    def add_state_message(
        self,
        task: str,
        state: BrowserState,
        result: Optional[List[ActionResult]] = None,
        step_info: Optional[AgentStepInfo] = None,
        use_vision=True,
        use_webwiki=True,
    ) -> None:

        if result:
            for r in result:
                if r.include_in_memory:
                    if r.extracted_content:
                        msg = HumanMessage(
                            content="Action result: " + str(r.extracted_content)
                        )
                        self._add_message_with_tokens(msg)
                    if r.error:
                        msg = HumanMessage(
                            content="Action error: "
                            + str(r.error)[-self.max_error_length :]
                        )
                        self._add_message_with_tokens(msg)
                        result = None

        # Update
        state_message = AgentMessagePrompt(
            task=task,
            state=state,
            webwiki=self.webwiki,
            include_attributes=self.include_attributes,
            max_error_length=self.max_error_length,
            step_info=step_info,
        ).get_user_message(use_vision, use_webwiki)

        self._add_message_with_tokens(state_message)

    def add_meta_action(self, action: str) -> None:
        msg = HumanMessage(content=action)
        self._add_message_with_tokens(msg)

    def _remove_last_state_message(self) -> None:
        if len(self.history.messages) > 2 and isinstance(
            self.history.messages[-1].message, HumanMessage
        ):
            self.history.remove_message()

    def add_model_output(self, model_output: AgentOutput) -> None:
        tool_calls = [
            {
                "name": "AgentOutput",
                "args": model_output.model_dump(mode="json", exclude_unset=True),
                "id": str(self.tool_id),
                "type": "tool_call",
            }
        ]

        observation = ""
        if hasattr(model_output, "current_state") and hasattr(
            model_output.current_state, "memory"
        ):
            observation = model_output.current_state.memory
        elif hasattr(model_output, "current_state") and hasattr(
            model_output.current_state, "observation"
        ):
            observation = model_output.current_state.observation
        else:
            observation = ""

        msg = AIMessage(
            content=observation,
            tool_calls=tool_calls,
        )
        self._add_message_with_tokens(msg)

        tool_message = ToolMessage(
            content="",
            tool_call_id=str(self.tool_id),
        )
        self._add_message_with_tokens(tool_message)
        self.tool_id += 1

    def get_messages(self) -> List[BaseMessage]:
        """Get current message list, potentially trimmed to max tokens"""

        msg = [m.message for m in self.history.messages]
        # debug which messages are in history with token count # log
        total_input_tokens = 0
        logger.debug(f"Messages in history: {len(self.history.messages)}:")
        for m in self.history.messages:
            total_input_tokens += m.metadata.input_tokens
            logger.debug(
                f"{m.message.__class__.__name__} - Token count: {m.metadata.input_tokens}"
            )
        logger.debug(f"Total input tokens: {total_input_tokens}")

        return msg

    def _add_message_with_tokens(
        self, message: BaseMessage, position: Optional[int] = None
    ) -> None:
        """Add message with token count metadata"""

        if self.sensitive_data:
            message = self._filter_sensitive_data(message)

        token_count = self._count_tokens(message)
        metadata = MessageMetadata(input_tokens=token_count)
        self.history.add_message(message, metadata, position)

    def _count_tokens(self, message: BaseMessage) -> int:

        tokens = 0
        if isinstance(message.content, list):
            for item in message.content:
                if "image_url" in item:
                    tokens += self.IMG_TOKENS
                elif isinstance(item, dict) and "text" in item:
                    tokens += self._count_text_tokens(item["text"])
        else:
            msg = message.content
            if hasattr(message, "tool_calls"):
                msg += str(message.tool_calls)
            tokens += self._count_text_tokens(msg)
        return tokens

    def _filter_sensitive_data(self, message: BaseMessage) -> BaseMessage:
        """Filter out sensitive data from the message"""

        def replace_sensitive(value: str) -> str:
            if not self.sensitive_data:
                return value
            for key, val in self.sensitive_data.items():
                if not val:
                    continue
                value = value.replace(val, f"<secret>{key}</secret>")
            return value

        if isinstance(message.content, str):
            message.content = replace_sensitive(message.content)
        elif isinstance(message.content, list):
            for i, item in enumerate(message.content):
                if isinstance(item, dict) and "text" in item:
                    item["text"] = replace_sensitive(item["text"])
                    message.content[i] = item
        return message

    def _count_text_tokens(self, text: str) -> int:
        """Count tokens in a text string"""

        tokens = (
            len(text) // self.estimated_characters_per_token
        )  # Rough estimate if no tokenizer available
        return tokens

    def cut_messages(self):
        """Get current message list, potentially trimmed to max tokens"""
        diff = self.history.total_tokens - self.max_input_tokens
        if diff <= 0:
            return None

        msg = self.history.messages[-1]

        # if list with image remove image
        if isinstance(msg.message.content, list):
            text = ""
            for item in msg.message.content:
                if "image_url" in item:
                    msg.message.content.remove(item)
                    diff -= self.IMG_TOKENS
                    msg.metadata.input_tokens -= self.IMG_TOKENS
                    self.history.total_tokens -= self.IMG_TOKENS
                    logger.debug(
                        f"Removed image with {self.IMG_TOKENS} tokens - total tokens now: {self.history.total_tokens}/{self.max_input_tokens}"
                    )
                elif "text" in item and isinstance(item, dict):
                    text += item["text"]
            msg.message.content = text
            self.history.messages[-1] = msg

        if diff <= 0:
            return None

        # if still over, remove text from state message proportionally to the number of tokens needed with buffer
        # Calculate the proportion of content to remove
        proportion_to_remove = diff / msg.metadata.input_tokens
        if proportion_to_remove > 0.99:
            raise ValueError(
                f"Max token limit reached - history is too long - reduce the system prompt or task. "
                f"proportion_to_remove: {proportion_to_remove}"
            )
        logger.debug(
            f"Removing {proportion_to_remove * 100:.2f}% of the last message  {proportion_to_remove * msg.metadata.input_tokens:.2f} / {msg.metadata.input_tokens:.2f} tokens)"
        )

        content = msg.message.content
        characters_to_remove = int(len(content) * proportion_to_remove)
        content = content[:-characters_to_remove]

        # remove tokens and old long message
        self.history.remove_message(index=-1)

        # new message with updated content
        msg = HumanMessage(content=content)
        self._add_message_with_tokens(msg)

        last_msg = self.history.messages[-1]

        logger.debug(
            f"Added message with {last_msg.metadata.input_tokens} tokens - total tokens now: {self.history.total_tokens}/{self.max_input_tokens} - total messages: {len(self.history.messages)}"
        )

    def convert_messages_for_non_function_calling_models(
        self, input_messages: list[BaseMessage]
    ) -> list[BaseMessage]:
        """Convert messages for non-function-calling models"""
        output_messages = []
        for message in input_messages:
            if isinstance(message, HumanMessage):
                output_messages.append(message)
            elif isinstance(message, SystemMessage):
                output_messages.append(message)
            elif isinstance(message, ToolMessage):
                output_messages.append(HumanMessage(content=message.content))
            elif isinstance(message, AIMessage):
                # check if tool_calls is a valid JSON object
                if message.tool_calls:
                    tool_calls = json.dumps(message.tool_calls)
                    output_messages.append(AIMessage(content=tool_calls))
                else:
                    output_messages.append(message)
            else:
                raise ValueError(f"Unknown message type: {type(message)}")
        return output_messages

    def merge_successive_messages(
        self, messages: list[BaseMessage], class_to_merge: Type[BaseMessage]
    ) -> list[BaseMessage]:
        """Some models like deepseek-reasoner dont allow multiple human messages in a row. This function merges them into one."""
        merged_messages = []
        streak = 0
        for message in messages:
            if isinstance(message, class_to_merge):
                streak += 1
                if streak > 1:
                    if isinstance(message.content, list):
                        merged_messages[-1].content += message.content[0]["text"]
                    else:
                        merged_messages[-1].content += message.content
                else:
                    merged_messages.append(message)
            else:
                merged_messages.append(message)
                streak = 0
        return merged_messages

    def extract_json_from_model_output(self, content: str) -> dict:
        """Extract JSON from model output, handling both plain JSON and code-block-wrapped JSON."""
        cleaned_content = content.strip()
        try:
            # If content is wrapped in code blocks, extract just the JSON part
            if cleaned_content.startswith("<tool_call>") and cleaned_content.endswith(
                "</tool_call>"
            ):
                # Remove tags and potentially leading/trailing whitespace/newlines between tags and JSON
                json_str = cleaned_content[
                    len("<tool_call>") : -len("</tool_call>")
                ].strip()
                outer_json = json.loads(json_str)
                if "arguments" in outer_json:
                    # The actual desired JSON is nested in 'arguments'
                    # We assume 'arguments' itself contains the correctly structured JSON
                    # If 'arguments' contains a string that needs parsing, use json.loads again:
                    # if isinstance(outer_json['arguments'], str):
                    #     return json.loads(outer_json['arguments'])
                    # else:
                    #     return outer_json['arguments']
                    # Assuming arguments is already the desired dict:
                    if isinstance(outer_json.get("arguments"), dict):
                        return outer_json["arguments"]
                    else:
                        # If arguments isn't a dict, try parsing it if it's a string
                        try:
                            return json.loads(str(outer_json.get("arguments", "{}")))
                        except json.JSONDecodeError:
                            raise ValueError(
                                "Could not parse 'arguments' field within <tool_call>."
                            )

                else:
                    raise ValueError("'<tool_call>' content missing 'arguments' field.")
            elif "```" in content:
                # Find the JSON content between code blocks
                content = content.split("```")[1]
                # Remove language identifier if present (e.g., 'json\n')
                if "\n" in content:
                    content = content.split("\n", 1)[1]
            # Parse the cleaned content
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse model output: {content} {str(e)}")
            raise ValueError("Could not parse response.")
