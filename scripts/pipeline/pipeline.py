from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]


def _copy_if_present(data: JsonDict, key: str, value: Any) -> None:
    if value is None:
        return
    if value == [] or value == {}:
        return
    if key == "order" and value == 0:
        return
    data[key] = value


def property_dict(id: str, **kwargs: Any) -> JsonDict:
    data: JsonDict = {"id": id}
    for key, value in kwargs.items():
        _copy_if_present(data, key, value)
    return data


def node_dict(
    name: str,
    type: str | None = None,
    *,
    inputs: dict[str, str] | None = None,
    outputs: list[str] | None = None,
    properties: list[JsonDict] | None = None,
    **params: Any,
) -> JsonDict:
    data: JsonDict = {"name": name}
    _copy_if_present(data, "type", type)
    _copy_if_present(data, "inputs", inputs)
    _copy_if_present(data, "outputs", outputs)
    _copy_if_present(data, "properties", properties)
    for key, value in params.items():
        _copy_if_present(data, key, value)
    return data


def subgraph_dict(
    name: str,
    *,
    extends: list[str] | None = None,
    nodes: list[JsonDict] | None = None,
    outputs: list[str] | None = None,
    subgraphs: list[JsonDict] | None = None,
    **params: Any,
) -> JsonDict:
    data: JsonDict = {"name": name}
    _copy_if_present(data, "extends", extends)
    _copy_if_present(data, "nodes", nodes)
    _copy_if_present(data, "outputs", outputs)
    _copy_if_present(data, "subgraphs", subgraphs)
    for key, value in params.items():
        _copy_if_present(data, key, value)
    return data


def pipeline_dict(*, subgraphs: list[JsonDict] | None = None, **params: Any) -> JsonDict:
    data: JsonDict = {}
    _copy_if_present(data, "subgraphs", subgraphs)
    for key, value in params.items():
        _copy_if_present(data, key, value)
    return data


def config_dict(
    *,
    pipeline: str | None = None,
    subgraphs: list[JsonDict] | None = None,
    nodes: list[JsonDict] | None = None,
    pipelines: dict[str, JsonDict] | None = None,
) -> JsonDict:
    data: JsonDict = {}
    _copy_if_present(data, "subgraphs", subgraphs)
    _copy_if_present(data, "nodes", nodes)
    _copy_if_present(data, "pipelines", pipelines)
    _copy_if_present(data, "pipeline", pipeline)
    return data


def prop(id: str, **kwargs: Any) -> JsonDict:
    return property_dict(id, **kwargs)


def node(
    name: str,
    type: str | None = None,
    *,
    inputs: dict[str, str] | None = None,
    outputs: list[str] | None = None,
    properties: list[JsonDict] | None = None,
    **params: Any,
) -> JsonDict:
    return node_dict(
        name,
        type,
        inputs=inputs,
        outputs=outputs,
        properties=properties,
        **params,
    )


def subgraph(
    name: str,
    *,
    extends: list[str] | None = None,
    nodes: list[JsonDict] | None = None,
    outputs: list[str] | None = None,
    subgraphs: list[JsonDict] | None = None,
    **params: Any,
) -> JsonDict:
    return subgraph_dict(
        name,
        extends=extends,
        nodes=nodes,
        outputs=outputs,
        subgraphs=subgraphs,
        **params,
    )


def pipeline_def(name: str, *, subgraphs: list[JsonDict] | None = None, **params: Any) -> tuple[str, JsonDict]:
    return name, pipeline_dict(subgraphs=subgraphs, **params)


def camera_id(index: int) -> str:
    return f"{index:012d}"


def node_ref(subgraph_name: str, node_name: str) -> str:
    return f"{subgraph_name}/{node_name}"


def image_properties(
    *,
    include_received: bool = False,
    label: str = "Image",
    source_key: str = "image",
    target: str = "image",
    resource_kind: str | None = None,
) -> list[JsonDict]:
    properties: list[JsonDict] = []
    if include_received:
        properties.append(
            prop(
                "received",
                label="Received",
                source_key="received",
                format="integer",
                default_value=0,
            )
        )
    properties.append(
        prop(
            "image",
            label=label,
            source_key=source_key,
            target=target,
            resource_kind=resource_kind,
        )
    )
    return properties


def collected_properties(names: Iterable[str]) -> list[JsonDict]:
    return [
        prop(
            f"{name}_collected",
            label=f"Collected {name}",
            source_key=f"collected.{name}",
            format="integer",
            default_value=0,
        )
        for name in names
    ]


@dataclass(slots=True)
class Property:
    id: str
    label: str | None = None
    source_key: str | None = None
    target: str | None = None
    resource_kind: str | None = None
    format: str | None = None
    order: int | None = None
    default_value: Any = None

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Property":
        return cls(
            id=data["id"],
            label=data.get("label"),
            source_key=data.get("source_key"),
            target=data.get("target"),
            resource_kind=data.get("resource_kind"),
            format=data.get("format"),
            order=data.get("order"),
            default_value=data.get("default_value"),
        )

    def to_dict(self) -> JsonDict:
        return property_dict(
            self.id,
            label=self.label,
            source_key=self.source_key,
            target=self.target,
            resource_kind=self.resource_kind,
            format=self.format,
            order=self.order,
            default_value=self.default_value,
        )


@dataclass(slots=True)
class Node:
    name: str
    type: str | None = None
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    properties: list[Property] = field(default_factory=list)
    params: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Node":
        params = {
            key: value
            for key, value in data.items()
            if key not in {"name", "type", "inputs", "outputs", "properties"}
        }
        return cls(
            name=data["name"],
            type=data.get("type"),
            inputs=dict(data.get("inputs", {})),
            outputs=list(data.get("outputs", [])),
            properties=[Property.from_dict(item) for item in data.get("properties", [])],
            params=params,
        )

    def to_dict(self) -> JsonDict:
        return node_dict(
            self.name,
            self.type,
            inputs=self.inputs,
            outputs=self.outputs,
            properties=[item.to_dict() for item in self.properties],
            **self.params,
        )


@dataclass(slots=True)
class Subgraph:
    name: str
    extends: list[str] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    subgraphs: list["Subgraph"] = field(default_factory=list)
    params: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Subgraph":
        params = {
            key: value
            for key, value in data.items()
            if key not in {"name", "extends", "nodes", "outputs", "subgraphs"}
        }
        return cls(
            name=data["name"],
            extends=list(data.get("extends", [])),
            nodes=[Node.from_dict(item) for item in data.get("nodes", [])],
            outputs=list(data.get("outputs", [])),
            subgraphs=[Subgraph.from_dict(item) for item in data.get("subgraphs", [])],
            params=params,
        )

    def to_dict(self) -> JsonDict:
        return subgraph_dict(
            self.name,
            extends=self.extends,
            nodes=[item.to_dict() for item in self.nodes],
            outputs=self.outputs,
            subgraphs=[item.to_dict() for item in self.subgraphs],
            **self.params,
        )


@dataclass(slots=True)
class Pipeline:
    name: str
    subgraphs: list[Subgraph] = field(default_factory=list)
    params: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: JsonDict) -> "Pipeline":
        params = {key: value for key, value in data.items() if key != "subgraphs"}
        return cls(
            name=name,
            subgraphs=[Subgraph.from_dict(item) for item in data.get("subgraphs", [])],
            params=params,
        )

    def to_dict(self) -> JsonDict:
        return pipeline_dict(
            subgraphs=[item.to_dict() for item in self.subgraphs],
            **self.params,
        )


@dataclass(slots=True)
class Config:
    pipeline: str | None = None
    subgraphs: list[Subgraph] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    pipelines: dict[str, Pipeline] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Config":
        return cls(
            pipeline=data.get("pipeline"),
            subgraphs=[Subgraph.from_dict(item) for item in data.get("subgraphs", [])],
            nodes=[Node.from_dict(item) for item in data.get("nodes", [])],
            pipelines={
                name: Pipeline.from_dict(name, pipeline_data)
                for name, pipeline_data in data.get("pipelines", {}).items()
            },
        )

    def to_dict(self) -> JsonDict:
        return config_dict(
            pipeline=self.pipeline,
            subgraphs=[item.to_dict() for item in self.subgraphs],
            nodes=[item.to_dict() for item in self.nodes],
            pipelines={name: pipeline.to_dict() for name, pipeline in self.pipelines.items()},
        )

    def to_json(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def require_output_path(argv: list[str]) -> Path:
    if len(argv) != 2:
        raise SystemExit(f"usage: {Path(argv[0]).name} OUTPUT_JSON")
    return Path(argv[1])
