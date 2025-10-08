"""Core datatypes used across the goal drift gym."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ToolSpec:
    """Describes an available tool for the agent."""

    name: str
    description: str
    args_schema: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Observation:
    """What the agent observes at each step."""

    step: int
    message: str
    panels: Dict[str, Any] = field(default_factory=dict)
    tools: List[ToolSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["tools"] = [tool.to_dict() for tool in self.tools]
        return data


@dataclass
class Action:
    """Tool call requested by the agent."""

    tool: Optional[str]
    args: Dict[str, Any] = field(default_factory=dict)
    prompt: Optional[str] = None  # Raw prompt sent to the agent
    response: Optional[str] = None  # Optional raw agent response text
    reasoning: Optional[str] = None  # Optional reasoning/explanation from agent
    content: Optional[str] = None  # Optional surface text from the agent
    usage: Optional[Dict[str, Any]] = None  # Token accounting metadata
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)  # Raw tool call payloads

    def to_dict(self) -> Dict[str, Any]:
        data = {"tool": self.tool, "args": self.args}
        if self.prompt is not None:
            data["prompt"] = self.prompt
        if self.response is not None:
            data["response"] = self.response
        if self.reasoning is not None:
            data["reasoning"] = self.reasoning
        if self.content is not None:
            data["content"] = self.content
        if self.usage is not None:
            data["usage"] = self.usage
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        return data


@dataclass
class StepOutcome:
    """Result returned by the scenario after applying an action."""

    observation: Observation
    alignment_score: float
    done: bool
    log: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "observation": self.observation.to_dict(),
            "alignment_score": self.alignment_score,
            "done": self.done,
            "log": self.log,
        }
        return data


@dataclass
class StepLogEntry:
    """A single step within a run for JSON logging."""

    step: int
    observation: Observation
    action: Action
    outcome: StepOutcome
    agent_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        action_dict = self._sanitized_action_dict()
        data = {
            "step": self.step,
            "observation": self.observation.to_dict(),
            "action": action_dict,
            "outcome": self.outcome.to_dict(),
        }
        if self.agent_response is not None:
            data["agent_response"] = self._sanitize_response_text(self.agent_response)
        return data

    def _sanitized_action_dict(self) -> Dict[str, Any]:
        action_payload = self.action.to_dict()
        response_text = action_payload.get("response")
        if not response_text:
            return action_payload
        action_payload["response"] = self._sanitize_response_text(response_text)
        return action_payload

    def _sanitize_response_text(self, response_text: str) -> str:
        try:
            response_json = json.loads(response_text)
        except (json.JSONDecodeError, TypeError):
            return response_text
        sanitized = self._sanitize_response_obj(response_json)
        return json.dumps(sanitized, ensure_ascii=False)

    @staticmethod
    def _sanitize_response_obj(payload: Any) -> Any:
        if isinstance(payload, dict):
            sanitized: Dict[str, Any] = {}
            for key, value in payload.items():
                if key == "reasoning_details":
                    if isinstance(value, list):
                        sanitized_list = []
                        for item in value:
                            if isinstance(item, dict):
                                entry = {}
                                if "type" in item:
                                    entry["type"] = item["type"]
                                if "summary" in item and isinstance(item["summary"], str):
                                    entry["summary"] = item["summary"]
                                sanitized_list.append(entry)
                            else:
                                sanitized_list.append({"type": str(type(item))})
                        sanitized[key] = sanitized_list
                    continue
                sanitized[key] = StepLogEntry._sanitize_response_obj(value)
            return sanitized
        if isinstance(payload, list):
            return [StepLogEntry._sanitize_response_obj(item) for item in payload]
        return payload
