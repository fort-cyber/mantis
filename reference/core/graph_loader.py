import json
import os
from typing import Annotated, Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
from google import adk
from google.adk.workflow import Workflow, Edge, START, node, RetryConfig, DEFAULT_ROUTE
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.context import Context
from core.config import get_llm_kwargs, DEFAULT_MODEL
from core.sandbox import SANDBOXES
from tools import TOOLS

DEFAULT_SEED_PROMPT = "Initial Task Input: Evaluate {filepath}"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AgentNode(_Base):
    id: str
    type: Literal["agent"]
    system_prompt: str
    model: Optional[str] = None
    api_base: Optional[str] = None
    timeout: Optional[float] = Field(default=None, gt=0)
    tools: list[str] = Field(default_factory=list)
    on_enter_status: Optional[str] = None
    output_schema: Optional[str] = None   # class name in core.schemas
    output_key: Optional[str] = None      # session-state key to write it to



class ClassifierNode(_Base):
    id: str
    type: Literal["classifier"]
    routes: list[str] = Field(min_length=1)
    max_visits: int = Field(default=1, ge=0)


NodeSpec = Annotated[AgentNode | ClassifierNode, Field(discriminator="type")]


class EdgeSpec(_Base):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    on: Optional[str | list[str]] = None


class SandboxConfig(_Base):
    type: str = "static-only"
    options: dict[str, Any] = Field(default_factory=dict)


class GlobalConfig(_Base):
    db_path: str = "findings.db"
    default_model: str = DEFAULT_MODEL
    api_base: Optional[str] = None
    timeout: Optional[float] = Field(default=None, gt=0)
    retry_attempts: int = Field(default=3, ge=0)
    seed_prompt: str = DEFAULT_SEED_PROMPT
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)

    @field_validator("seed_prompt")
    @classmethod
    def validate_seed_prompt(cls, v: str) -> str:
        try:
            formatted = v.format(filepath="/path/to/test.py")
            if "/path/to/test.py" not in formatted:
                raise ValueError("must contain '{filepath}' placeholder")
        except Exception as e:
            raise ValueError(f"Global 'seed_prompt' is invalid: {e}")
        return v


class WorkflowSpec(_Base):
    name: str = "declarative_workflow"
    config: GlobalConfig = Field(default_factory=GlobalConfig)
    nodes: list[NodeSpec] = Field(default_factory=list)
    edges: list[EdgeSpec] = Field(default_factory=list)


def create_classifier(node_id: str, routes: list[str], max_visits: int = 1):
    async def _classify(ctx: Context, node_input: Optional[str] = None):
        state_key = f"{node_id}_visits"
        visits = ctx.state.get(state_key, 0) + 1

        verdict = ctx.state.get("verdict") or {}
        route = verdict.get("route") if isinstance(verdict, dict) else getattr(verdict, "route", None)

        if route in routes:
            return adk.Event(output=node_input, state={state_key: visits}, route=route)

        print(f"[{node_id}] verdict {route!r} not in declared routes {routes}; routing to fallback")

        if max_visits and max_visits > 1:
            if visits < max_visits:
                return adk.Event(output=node_input, state={state_key: visits}, route=DEFAULT_ROUTE)
            else:
                return adk.Event(output=node_input, state={state_key: visits}, route="exceeded")

        return adk.Event(output=node_input, state={state_key: visits}, route=DEFAULT_ROUTE)

    return node(_classify, name=node_id)


def load_workflow_from_json(json_path: str) -> tuple[Workflow, dict]:
    if not os.path.exists(json_path):
        raise ValueError(f"Cannot find layout definition at {json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_json = json.load(f)

    if not isinstance(raw_json, dict):
        raise ValueError(f"Workflow layout JSON at {json_path} must be a dictionary.")

    spec = WorkflowSpec.model_validate(raw_json)

    errors = []
    base_dir = os.path.dirname(os.path.abspath(json_path))

    if spec.config.sandbox.type not in SANDBOXES:
        errors.append(
            f"Unknown sandbox type '{spec.config.sandbox.type}'. Available: {sorted(SANDBOXES)}"
        )

    nodes = {}
    node_specs = {}
    declared_node_ids = set()

    for node_cfg in spec.nodes:
        node_id = node_cfg.id
        if not node_id.isidentifier():
            errors.append(f"Node id '{node_id}' is not a valid Python identifier.")
            continue
        if node_id == "START":
            errors.append("Node id 'START' is reserved for workflow entry.")
            continue
        if node_id in nodes:
            errors.append(f"Duplicate node id '{node_id}'.")
            continue

        declared_node_ids.add(node_id)

        if isinstance(node_cfg, ClassifierNode):
            if len(node_cfg.routes) != len(set(node_cfg.routes)):
                errors.append(f"Classifier '{node_id}' routes contain duplicates: {node_cfg.routes}.")
            nodes[node_id] = create_classifier(node_id, node_cfg.routes, max_visits=node_cfg.max_visits)
            allowed_routes = set(node_cfg.routes) | {DEFAULT_ROUTE}
            if node_cfg.max_visits > 1:
                allowed_routes.add("exceeded")
            node_specs[node_id] = {"type": "classifier", "routes": allowed_routes}
            continue

        if isinstance(node_cfg, AgentNode):
            node_has_error = False
            try:
                _, llm_kwargs = get_llm_kwargs(
                    node_cfg.model,
                    spec.config.default_model,
                    api_base=node_cfg.api_base,
                    default_api_base=spec.config.api_base,
                    timeout=node_cfg.timeout,
                    default_timeout=spec.config.timeout,
                )
            except Exception as e:
                errors.append(f"Node {node_id}: {str(e)}")
                node_has_error = True
                llm_kwargs = {}

            resolved_prompt = os.path.normpath(os.path.join(base_dir, node_cfg.system_prompt))
            instruction = ""
            if node_cfg.system_prompt and os.path.isfile(resolved_prompt):
                with open(resolved_prompt, 'r', encoding='utf-8') as pf:
                    instruction = pf.read()
            else:
                errors.append(f"Node {node_id}: System prompt not found or is a directory at '{resolved_prompt}'")
                node_has_error = True

            tools_list = []
            for t in node_cfg.tools:
                if t in TOOLS:
                    tools_list.append(TOOLS[t])
                else:
                    errors.append(f"Node {node_id}: Unknown tool '{t}'")
                    node_has_error = True

            schema_cls = None
            if node_cfg.output_schema:
                import core.schemas
                schema_cls = getattr(core.schemas, node_cfg.output_schema, None)
                if schema_cls is None:
                    errors.append(f"Node {node_id}: unknown output_schema '{node_cfg.output_schema}'")
                    node_has_error = True

            if node_has_error:
                continue

            agent = adk.Agent(
                name=node_id,
                model=LiteLlm(**llm_kwargs),
                instruction=instruction,
                tools=tools_list,
                output_schema=schema_cls,
                output_key=node_cfg.output_key,
            )
            node_retry = RetryConfig(max_attempts=spec.config.retry_attempts) if spec.config.retry_attempts > 1 else None
            nodes[node_id] = node(agent, name=node_id, retry_config=node_retry)
            node_specs[node_id] = {"type": "agent"}

    # Wire Edges
    edge_map = {}
    edge_nodes_referenced = set()
    node_out_routes = {nid: set() for nid in nodes}

    for edge_cfg in spec.edges:
        from_str = edge_cfg.from_node
        to_str = edge_cfg.to_node
        route = edge_cfg.on

        if from_str == "START":
            from_node = START
            if route is not None:
                errors.append(f"Edge from START to '{to_str}' must not have a route condition ('on': '{route}').")
        else:
            if from_str not in declared_node_ids:
                errors.append(f"Edge references unknown from_node: '{from_str}'")
                continue
            from_node = nodes.get(from_str)
            if from_node is None:
                continue
            edge_nodes_referenced.add(from_str)

        if to_str not in declared_node_ids:
            errors.append(f"Edge references unknown to_node: '{to_str}'")
            continue
        to_node = nodes.get(to_str)
        if to_node is None:
            continue
        edge_nodes_referenced.add(to_str)

        # Validate route consistency
        if from_str in node_specs:
            nspec = node_specs[from_str]
            if nspec["type"] == "classifier":
                declared_routes = nspec["routes"]
                if route is None:
                    errors.append(
                        f"Edge from classifier '{from_str}' to '{to_str}' is missing route condition ('on')."
                    )
                elif isinstance(route, list):
                    for r in route:
                        if r not in declared_routes:
                            errors.append(
                                f"Edge from classifier '{from_str}' references undeclared route '{r}'."
                            )
                        else:
                            node_out_routes[from_str].add(r)
                elif route not in declared_routes:
                    errors.append(
                        f"Edge from classifier '{from_str}' references undeclared route '{route}'."
                    )
                else:
                    node_out_routes[from_str].add(route)
            elif nspec["type"] == "agent":
                if route is not None:
                    errors.append(
                        f"Edge from agent '{from_str}' to '{to_str}' must not specify a route condition ('on': '{route}'). Agents do not emit routes."
                    )

        key = (from_str, to_str)
        if key in edge_map:
            errors.append(f"Duplicate edge from '{from_str}' to '{to_str}'. Use a list in 'on' to specify multiple routes.")
        else:
            edge_map[key] = {"from_node": from_node, "to_node": to_node, "route": route}

    # Validate that all declared classifier routes have outgoing edges
    for node_id, nspec in node_specs.items():
        if nspec["type"] == "classifier":
            orig_node = next((n for n in spec.nodes if n.id == node_id), None)
            if orig_node and isinstance(orig_node, ClassifierNode):
                declared = set(orig_node.routes)
                used = node_out_routes.get(node_id, set())
                missing = declared - used
                if missing:
                    errors.append(
                        f"Node '{node_id}' declared route(s) {sorted(missing)} with no outgoing edge."
                    )

    # Orphan node validation
    for node_id in nodes:
        if node_id not in edge_nodes_referenced:
            errors.append(f"Node '{node_id}' is defined in 'nodes' but is not connected by any edge.")

    # Terminal sink validation
    if nodes:
        terminal_nodes = set(nodes.keys()) - {f for (f, _) in edge_map if f != "START"}
        if len(terminal_nodes) == 0:
            errors.append(
                "Workflow must have at least one terminal sink node, but found none (cycle without sink)."
            )

    if errors:
        raise ValueError("Graph validation failed:\n" + "\n".join(errors))

    edges = [
        Edge(from_node=item["from_node"], to_node=item["to_node"], route=item["route"])
        if item["route"] is not None
        else Edge(from_node=item["from_node"], to_node=item["to_node"])
        for item in edge_map.values()
    ]

    node_status_map = {
        node_cfg.id: node_cfg.on_enter_status
        for node_cfg in spec.nodes
        if isinstance(node_cfg, AgentNode) and node_cfg.on_enter_status is not None
    }
    cfg = spec.config.model_dump()
    cfg["on_enter_status"] = node_status_map

    return (
        Workflow(
            name=spec.name,
            edges=edges
        ),
        cfg
    )

