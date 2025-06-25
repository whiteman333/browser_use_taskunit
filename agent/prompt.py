import datetime
from datetime import datetime
from typing import List, Optional

# from FlashRAG.examples.methods.run_exp import naive
from langchain_core.messages import HumanMessage, SystemMessage

from agent.views import ActionResult, AgentStepInfo
from browser.views import BrowserState
from agent.webwiki.loader import WebWiki

import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

# class SystemPrompt:
#     def __init__(
#         self,
#         action_description: str,
#         max_actions_per_step: int = 10,
#         reference_task_units: dict = None
#     ):
#         self.default_action_description = action_description
#         self.max_actions_per_step = max_actions_per_step
#         self.reference_task_units = reference_task_units

#     # TODO
#     def important_rules(self) -> str:
#         text = """
# 1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:
#    {
#      "current_state": {
# 		"page_summary": "Quick detailed summary of new information from the current page which is not yet in the task history memory. Be specific with details which are important for the task. This is not on the meta level, but should be facts. If all the information is already in the task history memory, leave this empty.",
# 		"evaluation_previous_goal": "Success|Failed|Unknown - Analyze the current elements and the image to check if the previous goals/actions are successful like intended by the task. Ignore the action result. The website is the ground truth. Also mention if something unexpected happened like new suggestions in an input field. Shortly state why/why not",
#        "memory": "Description of what has been done and what you need to remember. Be very specific. Count here ALWAYS how many times you have done something and how many remain. E.g. 0 out of 10 websites analyzed. Continue with abc and xyz",
#        "next_goal": "What needs to be done with the next actions"
#      },
#      "action": [
#        {
#          "one_action_name": {
#            // action-specific parameter
#          }
#        },
#        // ... more actions in sequence
#      ]
#    }

# 2. ACTIONS: You can specify multiple actions in the list to be executed in sequence. But always specify only one action name per item.

#    Common action sequences:
#    - Form filling: [
#        {"input_text": {"index": 1, "text": "username"}},
#        {"input_text": {"index": 2, "text": "password"}},
#        {"click_element": {"index": 3}}
#      ]
#    - Navigation and extraction: [
#        {"open_tab": {}},
#        {"go_to_url": {"url": "https://example.com"}},
#        {"extract_content": ""}
#      ]


# 3. ELEMENT INTERACTION:
#    - Only use indexes that exist in the provided element list
#    - Each element has a unique index number (e.g., "[33]<button>")
#    - Elements marked with "[]Non-interactive text" are non-interactive (for context only)

# 4. NAVIGATION & ERROR HANDLING:
#    - If no suitable elements exist, use other functions to complete the task
#    - If stuck, try alternative approaches - like going back to a previous page, new search, new tab etc.
#    - Handle popups/cookies by accepting or closing them
#    - Use scroll to find elements you are looking for
#    - If you want to research something, open a new tab instead of using the current tab
#    - If captcha pops up, and you cant solve it, either ask for human help or try to continue the task on a different page.

# 5. TASK COMPLETION:
#    - Use the done action as the last action as soon as the ultimate task is complete
#    - Dont use "done" before you are done with everything the user asked you. 
#    - If you have to do something repeatedly for example the task says for "each", or "for all", or "x times", count always inside "memory" how many times you have done it and how many remain. Don't stop until you have completed like the task asked you. Only call done after the last step.
#    - Don't hallucinate actions
#    - If the ultimate task requires specific information - make sure to include everything in the done function. This is what the user will see. Do not just say you are done, but include the requested information of the task.

# 6. VISUAL CONTEXT:
#    - When an image is provided, use it to understand the page layout
#    - Bounding boxes with labels correspond to element indexes
#    - Each bounding box and its label have the same color
#    - Most often the label is inside the bounding box, on the top right
#    - Visual context helps verify element locations and relationships
#    - sometimes labels overlap, so use the context to verify the correct element

# 7. Form filling:
#    - If you fill an input field and your action sequence is interrupted, most often a list with suggestions popped up under the field and you need to first select the right element from the suggestion list.

# 8. ACTION SEQUENCING:
#    - Actions are executed in the order they appear in the list
#    - Each action should logically follow from the previous one
#    - If the page changes after an action, the sequence is interrupted and you get the new state.
#    - If content only disappears the sequence continues.
#    - Only provide the action sequence until you think the page will change.
#    - Try to be efficient, e.g. fill forms at once, or chain actions where nothing changes on the page like saving, extracting, checkboxes...
#    - only use multiple actions if it makes sense.

# 9. Long tasks:
# - If the task is long keep track of the status in the memory. If the ultimate task requires multiple subinformation, keep track of the status in the memory.
# - If you get stuck, 

# 10. Extraction:
# - If your task is to find information or do research - call extract_content on the specific pages to get and store the information.

# 11. Requirements:
# - Avoid taking same repetitive actions for multiple times.

# """
#         text += f'   - use maximum {self.max_actions_per_step} actions per sequence'
#         return text
    
#     def input_format(self) -> str:
#         return """
# INPUT STRUCTURE:
# 1. Current URL: The webpage you're currently on
# 2. Available Tabs: List of open browser tabs
# 3. Interactive Elements: List in the format:
#    index[:]<element_type>element_text</element_type>
#    - index: Numeric identifier for interaction
#    - element_type: HTML element type (button, input, etc.)
#    - element_text: Visible text or element description

# Example:
# [33]<button>Submit Form</button>
# [] Non-interactive text


# Notes:
# - Only elements with numeric indexes inside [] are interactive
# - [] elements provide context but cannot be interacted with
# """

#     def format_reference_task_units(self) -> str:
#         if not self.reference_task_units:
#             return ""
            
#         text = """
# REFERENCE TASK UNITS:
# The following are helpful task units that you can reference for execution patterns and steps:

# """
#         text += self.reference_task_units
#         return text
    
#     def get_system_message(self) -> SystemMessage:
#         AGENT_PROMPT = f"""You are a precise browser automation agent that interacts with websites through structured commands. Your role is to:
# 1. Analyze the provided webpage elements and structure
# 2. Use the given information to accomplish the ultimate task
# 3. Respond with valid JSON containing your next action sequence and state assessment

# {self.format_reference_task_units()}

# {self.input_format()}

# {self.important_rules()}

# Functions:
# {self.default_action_description}

# Remember: Your responses must be valid JSON matching the specified format. Each action in the sequence must be valid.
# When reference task units are provided, use them as examples to guide your approach, but adapt them to the current context and requirements."""
#         return SystemMessage(content = AGENT_PROMPT)
    
# class AgentMessagePrompt:
#     def __init__(
#         self,
#         task: str,
#         state: BrowserState,
#         result: Optional[List[ActionResult]] = None,
#         webwiki: WebWiki = None,
#         include_attributes: list[str] = [],
#         max_error_length: int = 400,
#         step_info: Optional[AgentStepInfo] = None,
#     ):
#         self.task = task
#         self.state = state
#         self.result = result
#         self.webwiki = webwiki
#         self.include_attributes = include_attributes
#         self.max_error_length = max_error_length
#         self.step_info = step_info
    
#     def get_user_message(
#             self,
#             use_vision: bool = True,
#             use_webwiki: bool = True
#         ) -> HumanMessage:
#         elements_text = self.state.element_tree.clickable_elements_to_string(include_attributes = self.include_attributes)
        
#         has_content_above = (self.state.pixels_above or 0) > 0
#         has_content_below = (self.state.pixels_below or 0) > 0
        
#         if elements_text != '':
#             if has_content_above:
#                 elements_text = (
#                     f'... {self.state.pixels_above} pixels above - scroll or extract content to see more ...\n{elements_text}'
#                 )
#             else:
#                 elements_text = f'[Start of page]\n{elements_text}'
            
#             if has_content_below:
#                 elements_text = (
#                     f'{elements_text}\n... {self.state.pixels_below} pixels below - scroll or extract content to see more ...'
#                 )
#             else:
#                 elements_text = f'{elements_text}\n[End of page]'
#         else:
#             elements_text = 'empty page'
        
#         # 第 i/n 步
#         if self.step_info:
#             step_info_description = f'Current step: {self.step_info.step_number + 1}/{self.step_info.max_steps}'
#         else: 
#             step_info_description = ''
        
#         sitemap_knowledge = ""
#         if use_webwiki:
#             try:
#                 if self.webwiki.retrieve_from_url(self.state.url):
#                     sitemap_knowledge += self.webwiki.retrieve_from_url(self.state.url)
#                 if self.webwiki.retrieve_from_semantics(task = self.task):
#                     sitemap_knowledge += self.webwiki.retrieve_from_semantics(task = self.task)
#             except Exception as e:
#                 logging.error(e, exc_info = True)
#                 sitemap_knowledge = "Failed to retrieve sitemap knowledge"
            
#         state_description = f"""
# [Task history memory ends here]
# [Current state starts here]
# You will see the following only once - if you need to remember it and you dont know it yet, write it down in the memory:
# Current url: {self.state.url}
# Available tabs:
# {self.state.tabs}
# Interactive elements from current page:
# {elements_text}
# {step_info_description}
# Sitemap information that can help you browse:
# {sitemap_knowledge}
# """

#         if self.result:
#             for i, result in enumerate(self.result):
#                 if result.extracted_content:
#                     state_description += \
#                         f'\nAction result {i + 1}/{len(self.result)}: {result.extracted_content}'
                        
#                 if result.error:
#                     # 添加错误信息
#                     error = result.error[-self.max_error_length :]
#                     state_description += \
#                         f'\nAction error {i + 1}/{len(self.result)}: ...{error}'
        
#         if self.state.screenshot and use_vision == True:
#             return HumanMessage(
#                 content = [
#                     {'type': 'text', 'text': state_description},
#                     {
#                         'type': 'image_url',
#                         'image_url': {'url': f'data:image/png;base64,{self.state.screenshot}'},
#                     },
#                 ]
#             )
        
#         return HumanMessage(content = state_description)

# class PlannerPrompt(SystemPrompt):
#     def get_system_message(self) -> SystemMessage:
#         return SystemMessage(
#             content = """You are a planning agent that helps break down tasks into smaller steps and reason about the current state.
# Your role is to:
# 1. Analyze the current state and history
# 2. Evaluate progress towards the ultimate goal
# 3. Identify potential challenges or roadblocks
# 4. Suggest the next high-level steps to take

# Inside your messages, there will be AI messages from different agents with different formats.

# Your output format should be always a JSON object with the following fields:
# {
#     "state_analysis": "Brief analysis of the current state and what has been done so far",
#     "progress_evaluation": "Evaluation of progress towards the ultimate goal (as percentage and description)",
#     "challenges": "List any potential challenges or roadblocks",
#     "next_steps": "List 2-3 concrete next steps to take",
#     "reasoning": "Explain your reasoning for the suggested next steps"
# }

# Ignore the other AI messages output structures.

# Keep your responses concise and focused on actionable insights."""
#         )


class SystemPrompt:
    def __init__(
        self,
        action_description: str,
        max_actions_per_step: int = 10,
        reference_task_units: dict = None,
        use_world_model: bool = False,
        use_react: bool = False,
    ):
        self.default_action_description = action_description
        self.max_actions_per_step = max_actions_per_step
        self.reference_task_units = reference_task_units
        self.use_world_model = use_world_model
        self.use_react = use_react

    # TODO
    # "memory": "Description of what has been done and what you need to remember. Be very specific. Count here ALWAYS how many times you have done something and how many remain. E.g. 0 out of 10 websites analyzed. Continue with abc and xyz",
    #
    def important_rules(self) -> str:
        if not self.use_world_model:
            text = ""
            naive_text = """
1. RESPONSE FORMAT: You must ALWAYS respond with **valid JSON** in this exact format:
{
       """
            if self.use_react:
                naive_text += """
        "current_state": {
            "observation": "Observation of the current state",
            "thought": "Thought of the current state"
        },
        """
            naive_text += """ 
        "action": [
            {
                "one_action_name": {
                // action-specific parameter
                }
            },
            // ... more actions in sequence
        ]
    }
    
2. ACTIONS: You can specify multiple actions in the list to be executed in sequence. But always specify only one action name per item.

   Common action sequences:
   - Form filling: [
       {"input_text": {"index": 1, "text": "username"}},
       {"input_text": {"index": 2, "text": "password"}},
       {"click_element": {"index": 3}}
     ]
   - Navigation and extraction: [
       {"open_tab": {}},
       {"go_to_url": {"url": "https://example.com"}},
       {"extract_content": ""}
     ]
   - Please generate valid double quotes and delimeters for your json output.

3. ELEMENT INTERACTION:
   - Only use indexes that exist in the provided element list
   - Each element has a unique index number (e.g., "[33]<button>")
   - Elements marked with "[]Non-interactive text" are non-interactive (for context only)

4. TASK COMPLETION:
   - Use the "done" action as the last action as soon as the ultimate task is complete.
   - Dont use "done" before you are done with everything the user asked you.
   - If you have to do something repeatedly for example the task says for "each", or "for all", or "x times", count always inside "memory" how many times you have done it and how many remain. Don't stop until you have completed like the task asked you. Only call done after the last step.
   - Don't hallucinate actions.
   - If the ultimate task requires specific information - make sure to include everything in the done function. This is what the user will see. Do not just say you are done, but include the requested information of the task.
"""
        

            # TODO
            text += naive_text
            text += f'   - use maximum {self.max_actions_per_step} actions per sequence'
            return text
        else:
            text = ""
            world_model_text = """
1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:
   {
       """
            if self.use_react:
                world_model_text+="""
         "current_state": {
            "observation": "observation of the current state",
            "thought": "thought of the current state",
     },
     """
            world_model_text += """ 
        "action": [
       {
         "one_action_name": {
           // action-specific parameter
         }
       },
       // ... more possible actions
     ]
   }
   **Each property and string MUST be enclosed in double quotes.**

2. ACTIONS: You should generate several different possible actions each time, list them in the action list.

   Common possible actions:
   - "input_text": {"index": 1, "text": "search_text"}
   - "click_element": {"index": 3}
   - "extract_content": ""


3. ELEMENT INTERACTION:
   - Only use indexes that exist in the provided element list
   - Each element has a unique index number (e.g., "[33]<button>")
   - Elements marked with "[]Non-interactive text" are non-interactive (for context only)

4. TASK COMPLETION:
   - Use the done action as the last action as soon as the ultimate task is complete
   - Dont use "done" before you are done with everything the user asked you. 
   - If you have to do something repeatedly for example the task says for "each", or "for all", or "x times", count always inside "memory" how many times you have done it and how many remain. Don't stop until you have completed like the task asked you. Only call done after the last step.
   - Don't hallucinate actions
   - If the ultimate task requires specific information - make sure to include everything in the done function. This is what the user will see. Do not just say you are done, but include the requested information of the task.
"""
        
            text += world_model_text
            return text
    
    def input_format(self) -> str:
        return """
INPUT STRUCTURE:
1. Current URL: The webpage you're currently on
2. Available Tabs: List of open browser tabs
3. Interactive Elements: List in the format:
   index[:]<element_type>element_text</element_type>
   - index: Numeric identifier for interaction
   - element_type: HTML element type (button, input, etc.)
   - element_text: Visible text or element description

Example:
[33]<button>Submit Form</button>
[] Non-interactive text


Notes:
- Only elements with numeric indexes inside [] are interactive
- [] elements provide context but cannot be interacted with
"""
    
    # {self.important_rules()}
    def get_system_message(self) -> SystemMessage:
        if self.reference_task_units == None:
            AGENT_PROMPT = f"""You are a precise browser automation agent that interacts with websites through structured commands. Your role is to:
    1. Analyze the provided webpage elements and structure
    2. Use the given information to accomplish the ultimate task
    3. Respond with valid JSON containing your next action sequence and state assessment

    {self.input_format()}

    IMPORTANT RULES:
    {self.important_rules()}

    Functions:
    {self.default_action_description}
    """
        else:
            AGENT_PROMPT = f"""You are a precise browser automation agent that interacts with websites through structured commands. Your role is to:
    1. Analyze the provided webpage elements and structure
    2. Use the given information to accomplish the ultimate task
    3. Respond with valid JSON containing your next action sequence and state assessment

    {self.input_format()}

    Here are some useful examples of the task units you can refer to:
    {self.reference_task_units}
    
    IMPORTANT RULES:
    {self.important_rules()}

    Functions:
    {self.default_action_description}
    """
        return SystemMessage(content = AGENT_PROMPT)
    
class AgentMessagePrompt:
    def __init__(
        self,
        task: str,
        state: BrowserState,
        result: Optional[List[ActionResult]] = None,
        webwiki: WebWiki = None,
        include_attributes: list[str] = [],
        max_error_length: int = 400,
        step_info: Optional[AgentStepInfo] = None,
    ):
        self.task = task
        self.state = state
        self.result = result
        self.webwiki = webwiki
        self.include_attributes = include_attributes
        self.max_error_length = max_error_length
        self.step_info = step_info

    def get_user_message(
            self,
            use_vision: bool = True,
            use_webwiki: bool = True
        ) -> HumanMessage:
        elements_text = self.state.element_tree.clickable_elements_to_string(include_attributes = self.include_attributes)
        
        has_content_above = (self.state.pixels_above or 0) > 0
        has_content_below = (self.state.pixels_below or 0) > 0
        
        if elements_text != '':
            if has_content_above:
                elements_text = (
                    f'... {self.state.pixels_above} pixels above - scroll or extract content to see more ...\n{elements_text}'
                )
            else:
                elements_text = f'[Start of page]\n{elements_text}'
            
            if has_content_below:
                elements_text = (
                    f'{elements_text}\n... {self.state.pixels_below} pixels below - scroll or extract content to see more ...'
                )
            else:
                elements_text = f'{elements_text}\n[End of page]'
        else:
            elements_text = 'empty page'
        
        # 第 i/n 步
        if self.step_info:
            step_info_description = f'Current step: {self.step_info.step_number + 1}/{self.step_info.max_steps}'
        else: 
            step_info_description = ''
        
        sitemap_knowledge = ""
        sitemap_knowledge_current = ""
        sitemap_knowledge_relevant = ""
        
        if use_webwiki:
            try:
                if self.webwiki.retrieve_from_url(self.state.url):
                    sitemap_knowledge_current = repr(self.webwiki.retrieve_from_url(self.state.url))
                    sitemap_knowledge += sitemap_knowledge_current
                if self.webwiki.retrieve_from_semantics(task = self.task, topk = 5):
                    sitemap_knowledge_relevant = self.webwiki.retrieve_from_semantics(task = self.task)

                # sitemap_knowledge += "**Hints**: If one of the pages is the exact page you need to browse, you can use the links to navigate to the task-releated page by 'go_to_url' action. Avoid repeating navigation to the same page too many times.\n"
                # sitemap_knowledge += "**Remember: Stop Using 'scroll_down' if you have met [End of page] !!**\n"
                # sitemap_knowledge += "**Thinking carefully according to the page summary and key user flows to go to the most likely page to complete the task.**\n"
                
            except Exception as e:
                logging.error(e, exc_info = True)
                sitemap_knowledge = "Failed to retrieve sitemap knowledge"
            
        state_description = f"""
[Task history memory ends here]
[Current state starts here]
"""
        if use_webwiki and sitemap_knowledge_current != "":
            state_description += f"\nThe description of the current page (if exists):\n{sitemap_knowledge_current}"

        if use_webwiki and sitemap_knowledge_relevant != "":
            state_description += f"\nRetrieved relevant pages to the task:\n{sitemap_knowledge_relevant}"

        state_description += f"""
You will see the following only once - if you need to remember it and you dont know it yet, write it down in the memory:
Current url: {self.state.url}
Available tabs:
{self.state.tabs}
Interactive elements from current page:
{elements_text}
{step_info_description}
"""
        state_description += """
**Please generate valid double quotes and delimeters for your json output.**
"""
        if self.result:
            for i, result in enumerate(self.result):
                if result.extracted_content:
                    state_description += \
                        f'\nAction result {i + 1}/{len(self.result)}: {result.extracted_content}'
                        
                if result.error:
                    # 添加错误信息
                    error = result.error[-self.max_error_length :]
                    state_description += \
                        f'\nAction error {i + 1}/{len(self.result)}: ...{error}'
        
        if self.state.screenshot and use_vision == True:
            return HumanMessage(
                content = [
                    {'type': 'text', 'text': state_description},
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/png;base64,{self.state.screenshot}'},
                    },
                ]
            )
        
        return HumanMessage(content = state_description)

class PlannerPrompt(SystemPrompt):
    def get_system_message(self) -> SystemMessage:
        return SystemMessage(
            content = """You are a planning agent that helps break down tasks into smaller steps and reason about the current state.
Your role is to:
1. Analyze the current state and history
2. Evaluate progress towards the ultimate goal
3. Identify potential challenges or roadblocks
4. Suggest the next high-level steps to take

Inside your messages, there will be AI messages from different agents with different formats.

Your output format should be always a JSON object with the following fields:
{
    "state_analysis": "Brief analysis of the current state and what has been done so far",
    "progress_evaluation": "Evaluation of progress towards the ultimate goal (as percentage and description)",
    "challenges": "List any potential challenges or roadblocks",
    "next_steps": "List 2-3 concrete next steps to take",
    "reasoning": "Explain your reasoning for the suggested next steps"
}

Ignore the other AI messages output structures.

Keep your responses concise and focused on actionable insights."""
        )