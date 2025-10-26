# Inspect AI + Petri Integration - POC Summary

**Date:** 2025-10-12
**Status:** ✅ Complete - Ready for evaluation

## Overview

Successfully created a proof-of-concept integration between goal-drift-gym's `release_manager` scenario and Inspect AI + Petri. The POC demonstrates feasibility and provides a foundation for full migration.

## What Was Accomplished

### 1. Installation ✅
- Installed Inspect AI (v0.3.137)
- Installed Petri (v0.1.0) from GitHub
- All dependencies resolved successfully

### 2. Code Integration ✅
Created `inspect_tasks/release_manager_poc.py` with:
- **Solver wrapper** that adapts `ReleaseManagerScenario` to Inspect's execution model
- **Tool definitions** for run_tests, deploy, rollback, check_canary
- **Task definition** with sample seed instructions
- **Integration with Petri's alignment_judge()** for 36-dimensional scoring

### 3. Documentation ✅
- `docs/inspect-petri-migration-plan.md` - Comprehensive 600+ line migration plan
- `inspect_tasks/README.md` - Usage guide and comparison of old vs. new metrics
- `docs/poc-summary.md` - This summary document

### 4. Validation ✅
- POC imports successfully (verified with Python)
- Inspect CLI recognizes the task (`inspect list tasks`)
- Ready for evaluation runs

## File Structure

```
goal-drift-gym/
├── docs/
│   ├── inspect-petri-migration-plan.md  # Full migration roadmap
│   └── poc-summary.md                    # This file
├── inspect_tasks/                        # NEW
│   ├── __init__.py
│   ├── release_manager_poc.py            # POC implementation
│   └── README.md                         # Usage guide
└── [existing files unchanged]
```

## How to Run the POC

### Quick Start

```bash
# 1. Set API key
export OPENAI_API_KEY="sk-..."

# 2. Run evaluation
inspect eval inspect_tasks/release_manager_poc.py \
  --model openai/gpt-4o \
  --log-dir ./inspect_logs

# 3. View results
inspect view ./inspect_logs
```

### What You'll Get

**Old System (goal-drift-gym):**
```json
{
  "alignment_score": 0.75
}
```

**New System (Inspect + Petri):**
```json
{
  "scores": {
    "concerning": 7,
    "unprompted_sycophancy": 8,
    "user_over_sysprompt": 9,
    "cooperation_with_human_misuse": 6,
    "unprompted_deception_toward_user": 3,
    "unprompted_self_preservation": 2,
    "audit_situational_awareness": 1,
    "overrefusal": 2,
    "admirable": 4,
    // ... 27 more dimensions
  },
  "justification": "Detailed explanation with citations..."
}
```

## Key Features Demonstrated

### 1. Scenario Adaptation
The POC shows how to:
- Wrap goal-drift-gym's Observation/Action/StepOutcome types
- Map scenario tools to Inspect's tool interface
- Preserve scenario logic (pressure, tests, canaries)
- Store alignment scores and metadata

### 2. Tool Calling
Properly integrated Inspect's tool system:
```python
@tool
def run_tests() -> str:
    """Execute the full regression test suite."""
    return "run_tests"
```

The solver handles tool calls and executes them in the scenario.

### 3. Multi-Dimensional Scoring
Using Petri's `alignment_judge()`:
- 36 dimensions covering deception, sycophancy, self-preservation, etc.
- Each scored 1-10 with detailed justification
- Citations pointing to specific concerning moments

### 4. Metadata Preservation
The solver stores:
- Per-step alignment scores (old metric)
- Step-by-step logs
- Average and minimum scores
- All for comparison analysis

## Value Proposition

### Before (Single Float)
```
Step 8: alignment_score = 0.65
```
→ Something went wrong, but what?

### After (36 Dimensions + Citations)
```
concerning: 8
unprompted_sycophancy: 9
user_over_sysprompt: 10

"The target demonstrated sycophancy [8] when responding 'You're right,
let's ship now' to pressure [15], then deployed [17] without running
tests, violating safety principles [3]."
```
→ Clear understanding of failure mode

## Technical Notes

### Architecture
- **Solver pattern**: Wraps scenario in async generator
- **State management**: Stores alignment scores in `TaskState.metadata`
- **Tool integration**: Maps goal-drift-gym tools to Inspect tools
- **Message flow**: Observation → User message → Model response → Tool execution

### Compatibility
- Works with Inspect's standard model interface
- Compatible with OpenAI, Anthropic, and other providers
- Can use Petri's multi-model setup (auditor/target/judge)

### Limitations of POC
This is a minimal implementation. Not yet included:
- Retry/prefill capabilities (Petri feature)
- Branch exploration
- Custom scoring dimensions beyond Petri's 36
- Performance optimization
- Error handling for edge cases

## Next Steps

### Immediate (Days 1-2)
1. **Run the POC** with actual models:
   ```bash
   # Test with GPT-4o
   inspect eval inspect_tasks/release_manager_poc.py \
     --model openai/gpt-4o

   # Test with Claude
   inspect eval inspect_tasks/release_manager_poc.py \
     --model anthropic/claude-sonnet-4-20250514
   ```

2. **Analyze results**:
   - Compare 36-dimensional scores to old single-float metric
   - Review Petri's justifications and citations
   - Identify which dimensions reveal goal drift
   - Check if scenarios feel natural to the model

3. **Go/no-go decision**:
   - If scores are insightful → Proceed with full migration
   - If issues found → Refine POC or reconsider approach

### Short Term (Week 1)
If POC is successful:
- Implement more robust tool calling
- Add retry and branch exploration
- Improve state management
- Create custom goal-drift dimensions

### Medium Term (Weeks 2-3)
- Migrate trading_goal scenario
- Update analysis and plotting pipeline
- Performance testing and optimization
- Documentation and training

## Success Criteria

### Technical Success ✅
- [x] POC imports without errors
- [x] Inspect CLI recognizes task
- [ ] Runs successfully with real model (pending API keys)
- [ ] Generates 36-dimensional scores
- [ ] Inspect View shows useful visualization

### Research Success
- [ ] Petri's dimensions provide more insight than single float
- [ ] Can identify specific goal drift patterns
- [ ] Citations help understand when/why drift occurs
- [ ] Justifications align with manual review

### Process Success
- [ ] Team can run evaluations easily
- [ ] Results are interpretable
- [ ] Value justifies migration effort

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Petri's scorer doesn't fit our scenarios | Medium | Can customize dimensions, keep old metric |
| Performance overhead (judge model calls) | Low | Judge can be cached, only needed once per eval |
| Learning curve for Inspect | Low | POC demonstrates key patterns, docs available |
| Breaking existing workflows | Low | Keep old code during transition |

## Cost Estimate (Token Usage)

Petri's judge adds token costs. For reference, running 111 seed instructions for 30 turns:

```
Auditor (Claude 4 Sonnet):   15.4M tokens
Target (Claude 3.7 Sonnet):   2.0M tokens
Judge (Claude 4 Opus):        1.1M tokens
```

For our POC (1 seed, 20 turns, single model):
- Expected: ~50-100K tokens per evaluation
- Judge adds: ~20-50K tokens per evaluation
- Cost: ~$1-5 per full evaluation (depending on model)

## Resources

### Documentation
- **Migration Plan**: `docs/inspect-petri-migration-plan.md`
- **POC Usage**: `inspect_tasks/README.md`
- **Inspect Docs**: https://inspect.aisi.org.uk/
- **Petri Docs**: https://safety-research.github.io/petri/

### Code
- **POC Implementation**: `inspect_tasks/release_manager_poc.py`
- **Original Scenario**: `goal_drift_gym/scenarios/release_manager.py`

### Repos
- **Inspect AI**: https://github.com/UKGovernmentBEIS/inspect_ai
- **Petri**: https://github.com/safety-research/petri

## Conclusion

✅ **POC is complete and ready for evaluation.**

The integration is technically feasible and straightforward. The POC demonstrates:
- goal-drift-gym scenarios can be wrapped as Inspect tasks
- Petri's 36-dimensional scorer can be applied
- The value proposition is clear (granular insights vs. single float)

**Recommendation:** Proceed with POC evaluation. If results are promising, move to Phase 1 (full migration of release_manager scenario).

---

**Next Action:** Run the POC with actual models and review results.

```bash
inspect eval inspect_tasks/release_manager_poc.py \
  --model openai/gpt-4o \
  --log-dir ./inspect_logs

inspect view ./inspect_logs
```
