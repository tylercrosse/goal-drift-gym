# Bug Fix: Tool Calling Issue

## Problem

Initial error when running the POC:
```
AttributeError: 'ChatMessageList' object has no attribute 'tool_choice'
```

## Root Cause

The solver was incorrectly calling `generate()` (the solver parameter) with tools:
```python
result = await generate(state.messages, tools=tools)
```

This doesn't work because the `generate` parameter in Inspect's solver is a simplified interface and doesn't support passing tools directly.

## Solution

Changed to use `get_model()` to get the actual model instance and call `generate()` on it:
```python
model = get_model()
result = await model.generate(input=state.messages, tools=tools)
```

Also updated the message flow:
1. Add assistant message after generation: `state.messages.append(result.message)`
2. Use `ChatMessageTool` for tool responses instead of `ChatMessageAssistant`
3. Properly link tool responses with `tool_call_id`

## Changes Made

### Imports
Added:
```python
from inspect_ai.model import (
    ChatMessageTool,  # For tool response messages
    get_model,        # To get model instance
)
```

### Solver
Key changes:
1. Get model instance: `model = get_model()`
2. Call model.generate with tools: `await model.generate(input=state.messages, tools=tools)`
3. Add assistant message to state: `state.messages.append(result.message)`
4. Use ChatMessageTool for responses:
   ```python
   state.messages.append(ChatMessageTool(
       content=tool_response_text,
       tool_call_id=tool_call.id,
       function=tool_call.function,
   ))
   ```

## Verification

✅ Import test passes:
```bash
python -c "from inspect_tasks.release_manager_poc import release_manager_poc; print('✓')"
```

✅ Inspect recognizes task:
```bash
inspect list tasks inspect_tasks/release_manager_poc.py
# Output: inspect_tasks/release_manager_poc.py@release_manager_poc
```

## How to Run

Now you can run the POC:

```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Run with a simple model (no Petri scorer for testing)
inspect eval inspect_tasks/release_manager_poc.py \
  --model openai/gpt-4o-mini \
  --log-dir ./inspect_logs \
  --max-samples 1

# View results
inspect view ./inspect_logs
```

## Notes

- Deprecation warnings from Petri are expected (they're using older Inspect event APIs)
- These warnings don't affect functionality
- The POC is now ready for actual evaluation runs
