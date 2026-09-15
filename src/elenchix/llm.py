from __future__ import annotations

import json
import re
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from openai import BadRequestError, OpenAI
from pydantic import BaseModel

from elenchix.config import LLMConfig, LLMRoleConfig
from elenchix.schemas import PlanningToolCall

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _load_react_agent():
    """Import the pinned LangGraph without its upstream Reviver default notice."""
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

    # The pinned checkpoint module constructs Reviver() at import time; our
    # planner does not configure or use that checkpoint deserializer. Limit the
    # filter to this exact upstream notice and restore the caller's filters.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The default value of `allowed_objects` will change in a future version\.",
            category=LangChainPendingDeprecationWarning,
            module=r"langgraph\.checkpoint\..*",
        )
        from langgraph.prebuilt import create_react_agent
    return create_react_agent


@dataclass(frozen=True)
class FunctionTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    def invoke(self, arguments: Mapping[str, Any]) -> Any:
        return self.handler(**dict(arguments))


class StructuredLLM(Protocol):
    def generate(
        self,
        *,
        role: str,
        system: str,
        payload: Mapping[str, Any],
        schema: type[SchemaT],
    ) -> SchemaT: ...


class ToolCallingLLM(StructuredLLM, Protocol):
    def generate_with_tools(
        self,
        *,
        role: str,
        system: str,
        payload: Mapping[str, Any],
        schema: type[SchemaT],
        tools: Sequence[FunctionTool],
        max_tool_steps: int,
    ) -> tuple[SchemaT, list[PlanningToolCall]]: ...


def _parse_json_content(content: str, schema: type[SchemaT]) -> SchemaT:
    cleaned = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", cleaned, re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    elif not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    return schema.model_validate_json(cleaned or "{}")


class OpenAICompatibleLLM:
    """Small OpenAI-compatible JSON client; credentials remain in the environment."""

    def __init__(self, config: LLMConfig):
        self.planning_max_retries = config.planning_max_retries
        self.teaching_max_empty_retries = config.teaching_max_empty_retries
        self.assessment_max_retries = config.assessment_max_retries
        self.teaching_history_limit = config.teaching_history_limit
        self.roles: dict[str, LLMRoleConfig] = {
            "planning": config.planning,
            "teaching": config.teaching,
            "assessment": config.assessment,
        }
        self.clients: dict[str, OpenAI] = {}
        self._credentials: dict[str, tuple[str, str | None]] = {}
        for role, role_config in self.roles.items():
            api_key, base_url = config.credentials(role_config)
            self._credentials[role] = (api_key, base_url)
            self.clients[role] = OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        *,
        role: str,
        system: str,
        payload: Mapping[str, Any],
        schema: type[SchemaT],
    ) -> SchemaT:
        role_config = self.roles[role]
        client = self.clients[role]
        prompt = {
            "task_input": payload,
            "required_json_schema": schema.model_json_schema(),
        }
        request = {
            "model": role_config.model,
            "temperature": role_config.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        }
        if role_config.extra_body:
            request["extra_body"] = role_config.extra_body
        try:
            response = client.chat.completions.create(
                **request, response_format={"type": "json_object"}
            )
        except BadRequestError:
            response = client.chat.completions.create(**request)
        content = response.choices[0].message.content or "{}"
        return _parse_json_content(content, schema)

    def generate_text(
        self,
        *,
        role: str,
        system: str,
        payload: Mapping[str, Any] | None = None,
        messages: Sequence[Mapping[str, str]] | None = None,
        tools: Sequence[FunctionTool] = (),
    ) -> str:
        """Generate teaching text from the deployed system+recent-history protocol."""

        role_config = self.roles[role]
        if messages is None:
            if payload is None:
                raise ValueError("generate_text requires messages or payload")
            conversation = [
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, default=str),
                }
            ]
        else:
            conversation = [dict(message) for message in messages]
            if not conversation:
                raise ValueError("teaching messages must not be empty")
            if any(message.get("role") not in {"user", "assistant"} for message in conversation):
                raise ValueError("teaching history accepts only user/assistant messages")
            conversation = conversation[-self.teaching_history_limit :]
        request: dict[str, Any] = {
            "model": role_config.model,
            "temperature": role_config.temperature,
            "messages": [
                {"role": "system", "content": system},
                *conversation,
            ],
        }
        if role_config.extra_body:
            request["extra_body"] = role_config.extra_body
        streaming = bool(role_config.extra_body.get("enable_thinking", False))
        for attempt in range(1, self.teaching_max_empty_retries + 1):
            content = self._complete_optional_tools(
                self.clients[role], request, tools=tools, streaming=streaming
            )
            if content and content.strip():
                return content.strip()
            if attempt < self.teaching_max_empty_retries:
                continue
        raise RuntimeError(
            f"{role} model returned an empty response after "
            f"{self.teaching_max_empty_retries} attempts"
        )

    @staticmethod
    def _streamed_content(chunks: Any) -> str:
        """Collect only visible answer tokens from a thinking-mode stream."""
        return OpenAICompatibleLLM._streamed_message(chunks)["content"]

    @staticmethod
    def _streamed_message(chunks: Any) -> dict:
        parts: list[str] = []
        calls: dict[int, dict] = {}
        for chunk in chunks:
            choices = getattr(chunk, "choices", ()) or ()
            if not choices:
                continue
            delta = choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                parts.append(str(content))
            for fragment in getattr(delta, "tool_calls", ()) or ():
                call = calls.setdefault(
                    fragment.index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                call["id"] += getattr(fragment, "id", None) or ""
                function = getattr(fragment, "function", None)
                for field in ("name", "arguments"):
                    call["function"][field] += getattr(function, field, None) or ""
        return {"content": "".join(parts), "tool_calls": [calls[i] for i in sorted(calls)]}

    def _complete_optional_tools(
        self,
        client: Any,
        request: dict,
        *,
        tools: Sequence[FunctionTool] = (),
        streaming: bool = False,
        json_mode: bool = False,
        max_tool_steps: int = 2,
    ) -> str:
        """Fetch graph context only when the model requests it; never preload it."""
        tool_map = {tool.name: tool for tool in tools}
        messages = list(request["messages"])
        for step in range(max_tool_steps + 1):
            current = {**request, "messages": messages}
            if tools:
                current["tools"] = [
                    {"type": "function", "function": {
                        "name": tool.name, "description": tool.description,
                        "parameters": tool.parameters,
                    }}
                    for tool in tools
                ]
                current["tool_choice"] = "auto" if step < max_tool_steps else "none"
            if streaming:
                current["stream"] = True
            if json_mode:
                current["response_format"] = {"type": "json_object"}
            try:
                response = client.chat.completions.create(**current)
            except BadRequestError:
                if not json_mode:
                    raise
                current.pop("response_format")
                response = client.chat.completions.create(**current)
            if streaming:
                message = self._streamed_message(response)
            else:
                raw = response.choices[0].message
                message = {
                    "content": raw.content or "",
                    "tool_calls": [
                        {"id": call.id, "type": "function", "function": {
                            "name": call.function.name, "arguments": call.function.arguments,
                        }}
                        for call in getattr(raw, "tool_calls", ()) or ()
                    ],
                }
            if not message["tool_calls"]:
                return message["content"]
            if step == max_tool_steps:
                raise RuntimeError("model exceeded the optional graph-tool call limit")
            messages = [*messages, {"role": "assistant", **message}]
            for call in message["tool_calls"]:
                function = call["function"]
                try:
                    arguments = json.loads(function["arguments"] or "{}")
                    result = tool_map[function["name"]].invoke(arguments)
                except (KeyError, TypeError, ValueError) as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                messages.append({
                    "role": "tool", "tool_call_id": call["id"],
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })
        raise RuntimeError("model did not produce a final response")

    def generate_assessment(
        self,
        *,
        prompt: str,
        schema: type[SchemaT],
    ) -> SchemaT:
        """Run the deployed full-conversation evaluation request with retries."""

        last_error: Exception | None = None
        for attempt in range(1, self.assessment_max_retries + 1):
            try:
                return self.generate_assessment_once(prompt=prompt, schema=schema)
            except Exception as exc:  # noqa: BLE001 - preserve the standalone client retry API
                last_error = exc
                if attempt < self.assessment_max_retries:
                    time.sleep(min(2, attempt))
        raise RuntimeError(
            f"assessment failed after {self.assessment_max_retries} attempts"
        ) from last_error

    def generate_assessment_once(
        self, *, prompt: str, schema: type[SchemaT], tools: Sequence[FunctionTool] = ()
    ) -> SchemaT:
        """One generation; AssessmentAgent owns the full parse-and-evidence retry budget."""

        role_config = self.roles["assessment"]
        request: dict[str, Any] = {
            "model": role_config.model,
            "temperature": role_config.temperature,
            "messages": [{
                "role": "user",
                "content": prompt + "\n\n【返回JSON结构约束】\n"
                + json.dumps(schema.model_json_schema(), ensure_ascii=False),
            }],
        }
        if role_config.extra_body:
            request["extra_body"] = role_config.extra_body
        client = self.clients["assessment"].with_options(max_retries=0)
        content = self._complete_optional_tools(client, request, tools=tools, json_mode=True)
        return _parse_json_content(content or "{}", schema)

    def generate_with_tools(
        self,
        *,
        role: str,
        system: str,
        payload: Mapping[str, Any],
        schema: type[SchemaT],
        tools: Sequence[FunctionTool],
        max_tool_steps: int,
    ) -> tuple[SchemaT, list[PlanningToolCall]]:
        if not tools:
            raise ValueError("planning requires at least one tool")
        role_config = self.roles[role]
        client = self.clients[role]
        tool_map = {tool.name: tool for tool in tools}
        tool_payload = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_input": payload,
                        "required_json_schema": schema.model_json_schema(),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        trace: list[PlanningToolCall] = []
        for step in range(max_tool_steps + 1):
            request = {
                "model": role_config.model,
                "temperature": role_config.temperature,
                "messages": messages,
                "tools": tool_payload,
                "tool_choice": "required" if step == 0 else "auto",
            }
            if role_config.extra_body:
                request["extra_body"] = role_config.extra_body
            try:
                response = client.chat.completions.create(**request)
            except BadRequestError:
                if step != 0:
                    raise
                request["tool_choice"] = "auto"
                response = client.chat.completions.create(**request)
            message = response.choices[0].message
            if not message.tool_calls:
                if not trace:
                    raise RuntimeError("planning model returned without using a tool")
                return _parse_json_content(message.content or "{}", schema), trace
            if step >= max_tool_steps:
                raise RuntimeError("planning model exceeded max_planning_tool_steps")
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in message.tool_calls
                    ],
                }
            )
            for call in message.tool_calls:
                arguments: dict[str, Any] = {}
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    tool = tool_map[call.function.name]
                    result = tool.invoke(arguments)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                trace.append(
                    PlanningToolCall(
                        tool_name=call.function.name,
                        arguments=arguments,
                        result=result,
                        source="react",
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.function.name,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )
        raise RuntimeError("planning tool loop did not produce a final plan")

    def run_react(
        self,
        *,
        system: str,
        payload: Mapping[str, Any],
        schema: type[SchemaT],
        tools: Sequence[FunctionTool],
        max_tool_steps: int,
    ) -> tuple[SchemaT, list[PlanningToolCall]]:
        """Run the deployed LangGraph prebuilt ReAct strategy for planning."""

        from langchain_core.messages import HumanMessage, ToolMessage
        from langchain_core.tools import StructuredTool
        from langchain_openai import ChatOpenAI

        create_react_agent = _load_react_agent()

        role_config = self.roles["planning"]
        api_key, base_url = self._credentials["planning"]
        chat_model = ChatOpenAI(
            model=role_config.model,
            api_key=api_key,
            base_url=base_url,
            temperature=role_config.temperature,
            max_retries=2,
        )
        langchain_tools = [
            StructuredTool.from_function(
                func=tool.handler,
                name=tool.name,
                description=tool.description,
            )
            for tool in tools
        ]
        agent = create_react_agent(chat_model, langchain_tools)
        prompt = system + "\n\n运行输入：\n" + json.dumps(payload, ensure_ascii=False)
        last_error: Exception | None = None
        for attempt in range(1, self.planning_max_retries + 1):
            try:
                result = agent.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config={"recursion_limit": 100},
                )
                trace: list[PlanningToolCall] = []
                calls: dict[str, tuple[str, dict[str, Any]]] = {}
                for message in result["messages"]:
                    for call in getattr(message, "tool_calls", ()) or ():
                        calls[str(call["id"])] = (
                            str(call["name"]),
                            dict(call.get("args") or {}),
                        )
                    if isinstance(message, ToolMessage):
                        name, arguments = calls.get(
                            str(message.tool_call_id),
                            (str(message.name or "unknown"), {}),
                        )
                        content = message.content
                        try:
                            parsed: Any = json.loads(str(content))
                        except json.JSONDecodeError:
                            parsed = content
                        trace.append(
                            PlanningToolCall(
                                tool_name=name,
                                arguments=arguments,
                                result=parsed,
                                source="react",
                            )
                        )
                if not trace:
                    raise RuntimeError("planning ReAct agent returned without using a tool")
                content = result["messages"][-1].content
                if isinstance(content, list):
                    content = "".join(
                        str(block.get("text", ""))
                        if isinstance(block, dict)
                        else str(block)
                        for block in content
                    )
                return _parse_json_content(str(content or "{}"), schema), trace
            except Exception as exc:  # noqa: BLE001 - deployed ReAct loop retries all failures
                last_error = exc
                if attempt < self.planning_max_retries:
                    time.sleep(min(2, attempt))
        raise RuntimeError(
            f"planning ReAct failed after {self.planning_max_retries} attempts"
        ) from last_error
