from __future__ import annotations

import statistics
from dataclasses import dataclass
from itertools import permutations
from collections import defaultdict
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np
from pm4py import convert, util as pm4py_util
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algorithm
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.discovery.dfg import algorithm as dfg_algorithm
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.statistics.end_activities.log import get as end_activities_get
from pm4py.statistics.rework.log import get as rework_get
from pm4py.statistics.start_activities.log import get as start_activities_get
from pm4py.statistics.variants.log import get as variants_get
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.petri_net.variants import token_decoration_frequency, token_decoration_performance
from pm4py.visualization.petri_net.util import performance_map
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer

from .log_loader import EventLogContainer


@dataclass
class ProcessModelArtifacts:
    net: Any
    initial_marking: Any
    final_marking: Any
    svg: bytes
    frequency_decorations: Optional[Dict[Any, Any]] = None
    performance_decorations: Optional[Dict[Any, Any]] = None


def _format_timedelta(delta: timedelta) -> str:
    total_seconds = int(delta.total_seconds())
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")
    return " ".join(parts)


def _compute_decorations(
    event_log: EventLog,
    net: Any,
    initial_marking: Any,
    final_marking: Any,
    measure: str = "frequency",
) -> Dict[Any, Any]:
    """Generic helper that runs token replay once to produce decorations."""

    activity_key = "concept:name"
    timestamp_key = "time:timestamp"

    variants_idx = variants_get.get_variants_from_log_trace_idx(event_log)
    variants = variants_get.convert_variants_trace_idx_to_trace_obj(event_log, variants_idx)

    token_parameters = {
        token_replay.Variants.TOKEN_REPLAY.value.Parameters.ACTIVITY_KEY: activity_key,
        token_replay.Variants.TOKEN_REPLAY.value.Parameters.VARIANTS: variants,
        token_replay.Variants.TOKEN_REPLAY.value.Parameters.SHOW_PROGRESS_BAR: False,
    }

    aligned_traces = token_replay.apply(
        event_log,
        net,
        initial_marking,
        final_marking,
        parameters=token_parameters,
    )

    element_statistics = performance_map.single_element_statistics(
        event_log,
        net,
        initial_marking,
        aligned_traces,
        variants_idx,
        activity_key=activity_key,
        timestamp_key=timestamp_key,
        parameters={"show_progress_bar": False},
    )

    aggregated_statistics = performance_map.aggregate_statistics(
        element_statistics,
        measure=measure,
    )
    return aggregated_statistics


def compute_overview(log_container: EventLogContainer) -> Dict[str, Any]:
    df = log_container.df
    cases = df["case_id"].nunique()
    events = len(df)
    activities = df["activity"].nunique()
    first_event = df["timestamp"].min()
    last_event = df["timestamp"].max()

    case_durations = (
        df.groupby("case_id")["timestamp"].agg(["min", "max"]).assign(duration=lambda g: g["max"] - g["min"])
    )["duration"]
    durations = case_durations.dt.total_seconds().tolist()
    events_per_case = df.groupby("case_id")["activity"].size().tolist()

    overview = {
        "cases": cases,
        "events": events,
        "activities": activities,
        "start": first_event,
        "end": last_event,
        "median_case_duration": _format_timedelta(timedelta(seconds=statistics.median(durations))) if durations else "n/a",
        "avg_case_duration": _format_timedelta(timedelta(seconds=statistics.mean(durations))) if durations else "n/a",
        "median_events_per_case": round(statistics.median(events_per_case), 2) if events_per_case else "n/a",
    }
    return overview


def discover_process_model(event_log: EventLog, variant: str = "im") -> ProcessModelArtifacts:
    """
    Discover a Petri net and return the net along with a rendered SVG.
    """
    variant_map = {
        "im": inductive_miner.Variants.IM,
        "imf": inductive_miner.Variants.IMf,
        "imd": inductive_miner.Variants.IMd,
    }
    if isinstance(variant, inductive_miner.Variants):
        miner_variant = variant
    else:
        miner_variant = variant_map.get(str(variant).lower())
    if miner_variant is None:
        raise ValueError(f"Unsupported discovery variant: {variant}")

    process_tree = inductive_miner.apply(event_log, variant=miner_variant)
    net, initial_marking, final_marking = convert.convert_to_petri_net(process_tree)

    frequency_decorations = _compute_decorations(event_log, net, initial_marking, final_marking, measure="frequency")
    performance_decorations = _compute_decorations(event_log, net, initial_marking, final_marking, measure="performance")

    svg_bytes = render_petri_net_svg(
        net,
        initial_marking,
        final_marking,
        event_log,
        mode="frequency",
        decorations=frequency_decorations,
    )

    return ProcessModelArtifacts(
        net=net,
        initial_marking=initial_marking,
        final_marking=final_marking,
        svg=svg_bytes,
        frequency_decorations=frequency_decorations,
        performance_decorations=performance_decorations,
    )


def render_petri_net_svg(
    net: Any,
    initial_marking: Any,
    final_marking: Any,
    event_log: EventLog,
    mode: str = "frequency",
    decorations: Optional[Dict[Any, Any]] = None,
) -> bytes:
    if mode == "frequency":
        params = {
            token_decoration_frequency.Parameters.FORMAT: "svg",
            token_decoration_frequency.Parameters.RANKDIR: "LR",
            token_decoration_frequency.Parameters.FONT_SIZE: 13,
            token_decoration_frequency.Parameters.ACTIVITY_KEY: "concept:name",
            token_decoration_frequency.Parameters.TIMESTAMP_KEY: "time:timestamp",
        }
        variant = pn_visualizer.Variants.FREQUENCY
        if decorations is None:
            decorations = _compute_decorations(event_log, net, initial_marking, final_marking, measure="frequency")
    elif mode == "performance":
        params = {
            token_decoration_performance.Parameters.FORMAT: "svg",
            token_decoration_performance.Parameters.RANKDIR: "LR",
            token_decoration_performance.Parameters.FONT_SIZE: 13,
            token_decoration_performance.Parameters.ACTIVITY_KEY: "concept:name",
            token_decoration_performance.Parameters.TIMESTAMP_KEY: "time:timestamp",
        }
        variant = pn_visualizer.Variants.PERFORMANCE
        if decorations is None:
            decorations = _compute_decorations(event_log, net, initial_marking, final_marking, measure="performance")
    else:
        params = {pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "svg"}
        variant = pn_visualizer.Variants.WO_DECORATION
        decorations = None

    gviz = pn_visualizer.apply(
        net,
        initial_marking,
        final_marking,
        log=event_log,
        parameters=params,
        variant=variant,
        aggregated_statistics=decorations,
    )
    return pn_visualizer.serialize(gviz)


def export_petri_net_pnml(artifacts: ProcessModelArtifacts) -> bytes:
    """
    Export the discovered Petri net to PNML for download.
    """
    pnml_string = pnml_exporter.serialize(artifacts.net, artifacts.initial_marking, artifacts.final_marking)
    if isinstance(pnml_string, bytes):
        return pnml_string
    return pnml_string.encode("utf-8")


def build_dfg_figures(log_container: EventLogContainer) -> Tuple[bytes, bytes]:
    """
    Generate SVGs for frequency and performance DFG visualisations.
    """
    parameters = {
        dfg_algorithm.Parameters.ACTIVITY_KEY: "concept:name",
        dfg_algorithm.Parameters.TIMESTAMP_KEY: "time:timestamp",
        dfg_algorithm.Parameters.CASE_ID_KEY: "case:concept:name",
    }

    dfg_frequency = dfg_algorithm.apply(
        log_container.event_log, variant=dfg_algorithm.Variants.FREQUENCY, parameters=parameters
    )
    dfg_performance = dfg_algorithm.apply(
        log_container.event_log, variant=dfg_algorithm.Variants.PERFORMANCE, parameters=parameters
    )
    start_activities = start_activities_get.get_start_activities(log_container.event_log)
    end_activities = end_activities_get.get_end_activities(log_container.event_log)

    frequency_variant = dfg_visualizer.Variants.FREQUENCY.value
    performance_variant = dfg_visualizer.Variants.PERFORMANCE.value
    freq_params = {
        frequency_variant.Parameters.FORMAT: "svg",
        frequency_variant.Parameters.START_ACTIVITIES: start_activities,
        frequency_variant.Parameters.END_ACTIVITIES: end_activities,
    }
    perf_params = {
        performance_variant.Parameters.FORMAT: "svg",
        performance_variant.Parameters.START_ACTIVITIES: start_activities,
        performance_variant.Parameters.END_ACTIVITIES: end_activities,
    }

    freq_gviz = dfg_visualizer.apply(
        dfg_frequency,
        log=log_container.event_log,
        parameters=freq_params,
        variant=dfg_visualizer.Variants.FREQUENCY,
    )
    perf_gviz = dfg_visualizer.apply(
        dfg_performance,
        log=log_container.event_log,
        parameters=perf_params,
        variant=dfg_visualizer.Variants.PERFORMANCE,
    )

    freq_svg = dfg_visualizer.serialize(freq_gviz)
    perf_svg = dfg_visualizer.serialize(perf_gviz)

    return freq_svg, perf_svg


def compute_variants_table(log_container: EventLogContainer, top_n: Optional[int] = None) -> pd.DataFrame:
    variants = variants_get.get_variants(log_container.event_log)
    rows = []
    sorted_variants = sorted(variants.items(), key=lambda item: len(item[1]), reverse=True)
    if top_n is not None:
        sorted_variants = sorted_variants[:top_n]
    for variant_key, traces in sorted_variants:
        if isinstance(variant_key, (list, tuple)):
            variant_label = " → ".join(str(step) for step in variant_key)
        else:
            variant_label = str(variant_key)
        rows.append(
            {
                "variant": variant_label,
                "cases": len(traces),
                "case_percentage": round(len(traces) / len(log_container.event_log) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def compute_rework_table(log_container: EventLogContainer) -> pd.DataFrame:
    rework_stats = rework_get.apply(log_container.event_log)
    total_cases = log_container.df["case_id"].nunique() or 1
    rows = []
    for activity, case_count in sorted(rework_stats.items(), key=lambda item: item[1], reverse=True):
        rows.append(
            {
                "activity": activity,
                "cases_with_rework": case_count,
                "case_percentage": round(case_count / total_cases * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def compute_resource_summary(log_container: EventLogContainer, top_n: int = 10) -> pd.DataFrame:
    if "resource" not in log_container.df.columns:
        return pd.DataFrame(columns=["resource", "events", "cases"])

    df = log_container.df
    events_series = df.groupby("resource").size().rename("events")
    cases_series = df.groupby("resource")["case_id"].nunique().rename("cases")
    summary = (
        pd.concat([events_series, cases_series], axis=1)
        .sort_values("events", ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={"index": "resource"})
    )
    return summary


def compute_overview_insights(log_container: EventLogContainer) -> Dict[str, Any]:
    overview = compute_overview(log_container)

    variants_df = compute_variants_table(log_container)
    if not variants_df.empty:
        top_variant_row = variants_df.iloc[0]
        top_variant = (top_variant_row["variant"], top_variant_row["case_percentage"])
    else:
        top_variant = None

    resource_df = compute_resource_summary(log_container, top_n=1)
    if not resource_df.empty:
        res_row = resource_df.iloc[0]
        top_resource = (res_row["resource"], int(res_row["events"]), int(res_row["cases"]))
    else:
        top_resource = None

    rework_df = compute_rework_table(log_container)
    if not rework_df.empty:
        rework_row = rework_df.iloc[0]
        top_rework = (rework_row["activity"], rework_row["case_percentage"])
    else:
        top_rework = None

    throughput_df = compute_throughput_distribution(log_container)
    avg_hours = throughput_df["duration_hours"].mean() if not throughput_df.empty else None
    median_hours = throughput_df["duration_hours"].median() if not throughput_df.empty else None

    time_span = None
    if overview.get("start") and overview.get("end"):
        start = overview["start"].strftime("%Y-%m-%d") if hasattr(overview["start"], "strftime") else str(overview["start"])
        end = overview["end"].strftime("%Y-%m-%d") if hasattr(overview["end"], "strftime") else str(overview["end"])
        time_span = f"{start} → {end}"

    return {
        "overview": overview,
        "top_variant": top_variant,
        "top_resource": top_resource,
        "top_rework": top_rework,
        "avg_duration_hours": avg_hours,
        "median_duration_hours": median_hours,
        "time_span": time_span,
    }


def compute_variant_detail(log_container: EventLogContainer, variant_label: str) -> Dict[str, Any]:
    sequence = [step.strip() for step in variant_label.split("→")]
    sequence = [step for step in sequence if step]

    log = log_container.event_log
    matching_cases = []
    for trace in log:
        activities = [event.get("concept:name") for event in trace]
        if activities == sequence:
            matching_cases.append(trace.attributes.get("concept:name"))

    df = log_container.df
    duration = None
    if matching_cases:
        subset = df[df["case_id"].isin(matching_cases)]
        if not subset.empty:
            grouped = subset.groupby("case_id")
            durations = (grouped["timestamp"].max() - grouped["timestamp"].min()).dt.total_seconds() / 3600
            duration = {
                "avg_hours": durations.mean(),
                "median_hours": durations.median(),
            }

    coverage_pct = 0.0
    if matching_cases:
        total_cases = len(log_container.event_log) or 1
        coverage_pct = len(matching_cases) / total_cases * 100

    return {
        "sequence": sequence,
        "cases": matching_cases,
        "duration": duration,
        "case_percentage": coverage_pct,
    }


def compute_dfg_frequency_data(
    log_container: EventLogContainer, max_activities: int = 12
) -> Dict[str, Any]:
    parameters = {
        dfg_algorithm.Parameters.ACTIVITY_KEY: "concept:name",
        dfg_algorithm.Parameters.TIMESTAMP_KEY: "time:timestamp",
        dfg_algorithm.Parameters.CASE_ID_KEY: "case:concept:name",
    }

    dfg_frequency = dfg_algorithm.apply(
        log_container.event_log, variant=dfg_algorithm.Variants.FREQUENCY, parameters=parameters
    )
    dfg_performance = dfg_algorithm.apply(
        log_container.event_log, variant=dfg_algorithm.Variants.PERFORMANCE, parameters=parameters
    )
    activity_frequency_series = log_container.df["activity"].value_counts()
    activity_frequency = activity_frequency_series.to_dict()
    start_activities = start_activities_get.get_start_activities(log_container.event_log)
    end_activities = end_activities_get.get_end_activities(log_container.event_log)

    # Limit to the most frequent activities to keep the graph legible.
    important_activities = sorted(activity_frequency.items(), key=lambda item: item[1], reverse=True)
    if max_activities and len(important_activities) > max_activities:
        keep = {act for act, _ in important_activities[:max_activities]}
    else:
        keep = set(activity_frequency.keys())

    filtered_edges = {edge: freq for edge, freq in dfg_frequency.items() if edge[0] in keep and edge[1] in keep}
    filtered_performance = {
        edge: dfg_performance.get(edge) for edge in filtered_edges.keys() if edge in dfg_performance
    }
    filtered_activity_freq = {act: freq for act, freq in activity_frequency.items() if act in keep}
    filtered_start = {act: freq for act, freq in start_activities.items() if act in keep}
    filtered_end = {act: freq for act, freq in end_activities.items() if act in keep}

    df = log_container.df.sort_values(["case_id", "timestamp"])
    edge_case_map: Dict[tuple[str, str], set[str]] = defaultdict(set)
    node_case_map: Dict[str, set[str]] = defaultdict(set)
    rework_counts: Dict[tuple[str, str], int] = defaultdict(int)

    for case_id, case_df in df.groupby("case_id"):
        events = [act for act in case_df["activity"].tolist() if act in keep]
        prev_act: Optional[str] = None
        for act in events:
            node_case_map[act].add(case_id)
            if prev_act is not None:
                edge = (prev_act, act)
                edge_case_map[edge].add(case_id)
                if prev_act == act:
                    rework_counts[edge] += 1
                    if edge not in filtered_edges:
                        filtered_edges[edge] = rework_counts[edge]
                elif edge not in filtered_edges:
                    filtered_edges[edge] = len(edge_case_map[edge])
            prev_act = act

    edge_case_counts = {edge: len(cases) for edge, cases in edge_case_map.items()}
    node_case_counts = {node: len(cases) for node, cases in node_case_map.items()}

    metadata = {
        "total_activities": len(activity_frequency),
        "kept_activities": len(filtered_activity_freq),
        "activity_limit": max_activities,
        "truncated": max(0, len(activity_frequency) - len(filtered_activity_freq)),
        "total_cases": len(log_container.event_log),
        "total_edges": len(filtered_edges),
        "max_edge_weight": max(filtered_edges.values(), default=0),
        "max_edge_duration": max(filtered_performance.values(), default=0) if filtered_performance else 0,
        "min_edge_duration": min(filtered_performance.values(), default=0) if filtered_performance else 0,
    }

    return {
        "edges": filtered_edges,
        "activities": filtered_activity_freq,
        "starts": filtered_start,
        "ends": filtered_end,
        "performance_edges": filtered_performance,
        "rework_edges": {edge: count for edge, count in rework_counts.items() if count > 0},
        "edge_cases": edge_case_counts,
        "node_cases": node_case_counts,
        "metadata": metadata,
    }


def render_bpmn_svg(artifacts: ProcessModelArtifacts) -> bytes:
    """
    Convert the discovered Petri net to BPMN and render it as SVG.
    """
    bpmn_graph = convert.convert_to_bpmn(artifacts.net, artifacts.initial_marking, artifacts.final_marking)
    parameters = {
        bpmn_visualizer.Variants.CLASSIC.value.Parameters.FORMAT: "svg",
        bpmn_visualizer.Variants.CLASSIC.value.Parameters.FONT_SIZE: 14,
    }
    gviz = bpmn_visualizer.apply(bpmn_graph, parameters=parameters, variant=bpmn_visualizer.Variants.CLASSIC)
    return bpmn_visualizer.serialize(gviz)


def compute_declare_constraints(
    log_container: EventLogContainer,
    *,
    top_activities: int = 10,
    max_constraints: int = 15,
) -> pd.DataFrame:
    """
    Derive lightweight Declare-style constraints (response, precedence, coexistence) for the most common activities.
    """
    df = log_container.df.copy()
    df = df.sort_values(["case_id", "timestamp"])
    total_cases = df["case_id"].nunique()
    if total_cases == 0:
        return pd.DataFrame(columns=["constraint", "type", "support", "confidence"])

    top_activity_names = (
        df["activity"].value_counts().head(top_activities).index.tolist()
    )
    traces: List[List[str]] = []
    for _, case_df in df.groupby("case_id"):
        traces.append(case_df["activity"].tolist())

    results: List[Dict[str, Any]] = []
    for activity_a, activity_b in permutations(top_activity_names, 2):
        response_counts = 0
        response_total = 0
        precedence_counts = 0
        precedence_total = 0
        coexistence_counts = 0
        coexistence_total = 0

        for events in traces:
            positions: Dict[str, List[int]] = {}
            for idx, act in enumerate(events):
                positions.setdefault(act, []).append(idx)

            pos_a = positions.get(activity_a)
            pos_b = positions.get(activity_b)

            # Response: every occurrence of A is eventually followed by B
            if pos_a:
                response_total += 1
                if pos_b:
                    fulfills = all(any(pb > pa for pb in pos_b) for pa in pos_a)
                    if fulfills:
                        response_counts += 1
            else:
                response_counts += 1  # vacuously satisfied

            # Precedence: every occurrence of B has at least one preceding A
            if pos_b:
                precedence_total += 1
                if pos_a:
                    fulfills = all(any(pa < pb for pa in pos_a) for pb in pos_b)
                    if fulfills:
                        precedence_counts += 1
            else:
                precedence_counts += 1

            # Coexistence: if one event appears, the other also appears
            if pos_a or pos_b:
                coexistence_total += 1
                if pos_a and pos_b:
                    coexistence_counts += 1

        if response_total > 0:
            confidence = response_counts / response_total
            support = response_counts / total_cases
            results.append(
                {
                    "constraint": f"{activity_a} leads to {activity_b}",
                    "type": "Response",
                    "support": round(support, 3),
                    "confidence": round(confidence, 3),
                }
            )

        if precedence_total > 0:
            confidence = precedence_counts / precedence_total
            support = precedence_counts / total_cases
            results.append(
                {
                    "constraint": f"{activity_a} precedes {activity_b}",
                    "type": "Precedence",
                    "support": round(support, 3),
                    "confidence": round(confidence, 3),
                }
            )

        if coexistence_total > 0:
            confidence = coexistence_counts / coexistence_total
            support = coexistence_counts / total_cases
            results.append(
                {
                    "constraint": f"{activity_a} co-exists with {activity_b}",
                    "type": "Co-existence",
                    "support": round(support, 3),
                    "confidence": round(confidence, 3),
                }
            )

    if not results:
        return pd.DataFrame(columns=["constraint", "type", "support", "confidence"])

    df_constraints = pd.DataFrame(results)
    df_constraints = df_constraints.sort_values(
        ["confidence", "support"], ascending=[False, False]
    ).drop_duplicates("constraint")
    if max_constraints:
        df_constraints = df_constraints.head(max_constraints)
    return df_constraints.reset_index(drop=True)


def compute_transition_matrix(
    log_container: EventLogContainer, top_activities: int = 12
) -> Dict[str, Any]:
    """
    Build a Markov-style transition matrix for the most frequent activities.
    """
    df = (
        log_container.df.sort_values(["case_id", "timestamp"])
        .copy()
    )
    activity_order = df["activity"].value_counts().head(top_activities).index.tolist()
    if not activity_order:
        return {"states": [], "matrix": [], "probabilities": []}

    states = ["⟨start⟩"] + activity_order + ["⟨end⟩"]
    index_map = {state: idx for idx, state in enumerate(states)}
    matrix = np.zeros((len(states), len(states)), dtype=float)

    for _, case_df in df.groupby("case_id"):
        events = ["⟨start⟩"]
        for act in case_df["activity"]:
            if act in activity_order:
                events.append(act)
        events.append("⟨end⟩")
        for src, dst in zip(events, events[1:]):
            src_idx = index_map[src]
            dst_idx = index_map[dst]
            matrix[src_idx, dst_idx] += 1

    totals = matrix.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        probabilities = np.divide(matrix, totals[:, None], where=totals[:, None] > 0)
    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "states": states,
        "matrix": matrix.tolist(),
        "probabilities": probabilities.tolist(),
        "totals": totals.tolist(),
    }


def compute_throughput_distribution(log_container: EventLogContainer) -> pd.DataFrame:
    durations = (
        log_container.df.groupby("case_id")["timestamp"].agg(["min", "max"]).assign(duration=lambda g: g["max"] - g["min"])
    )
    durations["duration_hours"] = durations["duration"].dt.total_seconds() / 3600
    durations = durations.reset_index().rename(columns={"case_id": "Case"})
    return durations[["Case", "duration_hours"]]


def compute_events_over_time(log_container: EventLogContainer) -> pd.DataFrame:
    df = log_container.df.copy()
    df["date"] = df["timestamp"].dt.date
    summary = df.groupby("date").size().reset_index(name="events")
    summary = summary.rename(columns={"date": "Date", "events": "Events"})
    return summary


def align_log_against_model(
    log_container: EventLogContainer, artifacts: ProcessModelArtifacts
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    alignments = alignments_algorithm.apply(
        log_container.event_log,
        artifacts.net,
        artifacts.initial_marking,
        artifacts.final_marking,
        parameters={pm4py_util.constants.PARAMETER_CONSTANT_ACTIVITY_KEY: "concept:name"},
    )

    alignment_rows: List[Dict[str, Any]] = []
    fitness_scores: List[float] = []

    for idx, (case, alignment) in enumerate(zip(log_container.event_log, alignments), start=1):
        fitness = alignment["fitness"]
        fitness_scores.append(fitness)
        case_name = case.attributes.get("concept:name") if isinstance(case.attributes, dict) else None
        case_name = case_name or f"Trace {idx}"
        alignment_rows.append(
            {
                "case_id": case_name,
                "fitness": round(fitness, 4),
                "cost": alignment["cost"],
                "log_moves": len(alignment["alignment"]) - sum(1 for move in alignment["alignment"] if move[0] == move[1]),
            }
        )

    summary = {
        "avg_fitness": round(statistics.mean(fitness_scores), 4) if fitness_scores else 0.0,
        "min_fitness": round(min(fitness_scores), 4) if fitness_scores else 0.0,
        "max_fitness": round(max(fitness_scores), 4) if fitness_scores else 0.0,
    }
    return pd.DataFrame(alignment_rows), summary
