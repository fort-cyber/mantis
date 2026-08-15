# MVP ADK Reference Implementation

This directory contains a minimalistic reference implementation of Mantis. It is
not production-ready and does not implement the full Mantis pipeline, but is a
working minimal harness that demonstrates how to build your own with the ADK.

## Core Principles

1. We use the ADK workflow graph orchestrator, defined in `workflow.json`.
2. We store results in sqlite.
3. We use minimalistic prompts with only a few agents, rather than a more
   complex agent architecture.
4. We demonstrate a plugin mechanism to hook up new sandboxes.
5. We **DO NOT** protect against prompt injection or malicious code. If you want
   such a thing you will require significantly more effort to handle that.

## Additional Documentation

The core is the documentation. There may be bugs and sharp edges to be ironed
out, but hopefully the demo at least works out of the box.

## Suggested Integration with Mantis Skills

If you'd like to use the Mantis skills directly in this harness, you could use
something like this:

```python
import pathlib

from google.adk.skills import load_skill_from_dir
from google.adk.tools import skill_toolset

researcher_skill = load_skill_from_dir(
    pathlib.path(__file__).parent / "mantis-researcher"
)

mantis_skill_toolset = skill_toolset.SkillToolset(
    skills = [researcher_skill]
)
```

And load the skill toolset as a tool for the relevant agents. This is likely to
also be a very useful pattern to customizing different stages of your pipeline
for your own codebases. This will give them better grounding and also more
efficient since they won't have to rediscover some esoteric properties of your
codebase or deployment that makes something a false positive.
