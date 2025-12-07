"""
Responses API utilities for AgentClinic environment.
Provides conversion between OpenAI's Responses API and Chat Completions formats.
"""
from typing import Any, Dict, List, Optional

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types import CompletionUsage


def adapt_responses_to_chat_completion(responses_response) -> ChatCompletion:
    """Convert Responses API response to ChatCompletion format."""
    output_text = getattr(responses_response, "output_text", "")

    # Extract tool calls
    tool_calls = None
    output = getattr(responses_response, "output", [])
    if output:
        extracted_calls = []
        for item in output:
            if getattr(item, "type", None) == "function_call":
                extracted_calls.append({
                    "id": getattr(item, "call_id", ""),
                    "type": "function",
                    "function": {
                        "name": getattr(item, "name", ""),
                        "arguments": str(getattr(item, "arguments", {})),
                    },
                })
        if extracted_calls:
            tool_calls = extracted_calls

    # Convert usage format
    usage_obj = None
    response_usage = getattr(responses_response, "usage", None)
    if response_usage:
        usage_obj = CompletionUsage(
            prompt_tokens=getattr(response_usage, "input_tokens", 0),
            completion_tokens=getattr(response_usage, "output_tokens", 0),
            total_tokens=getattr(response_usage, "total_tokens", 0)
        )

    return ChatCompletion(
        id=getattr(responses_response, "id", "responses-api-adapter"),
        object="chat.completion",
        created=getattr(responses_response, "created_at", 0),
        model=getattr(responses_response, "model", ""),
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=output_text,
                    tool_calls=tool_calls
                ),
                finish_reason="stop",
            )
        ],
        usage=usage_obj,
    )


def extract_system_message(messages: List[Dict[str, Any]]) -> tuple[Optional[str], List[Dict[str, Any]]]:
    """Extract system messages as instructions for Responses API."""
    system_messages = []
    other_messages = []

    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            system_messages.append(msg.get("content", ""))
        else:
            other_messages.append(msg)

    system_content = "\n\n".join(system_messages) if system_messages else None
    return system_content, other_messages


def adapt_tools_for_responses_api(oai_tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Convert tool format from Chat Completions (externally-tagged) to Responses API (internally-tagged)."""
    if not oai_tools:
        return None

    adapted_tools = []
    for tool in oai_tools:
        if isinstance(tool, dict) and tool.get("type") == "function" and "function" in tool:
            func = tool["function"]
            adapted_tools.append({
                "type": "function",
                "name": func.get("name"),
                "description": func.get("description"),
                "parameters": func.get("parameters"),
            })
        else:
            adapted_tools.append(tool)

    return adapted_tools if adapted_tools else None
