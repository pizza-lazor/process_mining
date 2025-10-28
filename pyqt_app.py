from __future__ import annotations

import difflib
import logging
import math
import pathlib
import re
import sys
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import DateAxisItem
from PyQt6 import QtCore, QtGui, QtWidgets, QtSvgWidgets

from pm4py.objects.petri_net import obj as pn_obj

PetriPlace = pn_obj.PetriNet.Place
PetriTransition = pn_obj.PetriNet.Transition
PetriArc = pn_obj.PetriNet.Arc

from app import process_analysis
from app.log_filters import filter_by_activity, filter_by_attribute, filter_by_case_ids, filter_by_time_range
from app.log_loader import (
    EventLogContainer,
    LogFormatError,
    load_log_from_csv,
    load_log_from_xes,
    read_csv_summary,
    try_auto_detect_columns,
)

SAMPLE_LOG_PATH = pathlib.Path("data/sample_order_to_cash.csv")
LOG_NAME = "process_mining_app"
LOG_FILE_PATH = pathlib.Path.cwd() / "process_mining_app.log"


def configure_logging() -> logging.Logger:
    logger = logging.getLogger(LOG_NAME)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.propagate = False
        logger.info("Logging initialised. Writing to %s", LOG_FILE_PATH)
    return logger


BASE_LOGGER = configure_logging()


class PandasTableModel(QtCore.QAbstractTableModel):
    def __init__(self, dataframe: Optional[pd.DataFrame] = None):
        super().__init__()
        self._dataframe = dataframe if dataframe is not None else pd.DataFrame()

    def set_dataframe(self, dataframe: pd.DataFrame):
        self.beginResetModel()
        self._dataframe = dataframe.copy()
        self.endResetModel()

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:  # type: ignore[override]
        if parent.isValid():
            return 0
        return len(self._dataframe.index)

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:  # type: ignore[override]
        if parent.isValid():
            return 0
        return len(self._dataframe.columns)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
        if not index.isValid() or role not in (QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.ToolTipRole):
            return None
        value = self._dataframe.iat[index.row(), index.column()]
        if pd.isna(value):
            return ""
        return str(value)

    def headerData(  # type: ignore[override]
        self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.ItemDataRole.DisplayRole
    ):
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            try:
                return str(self._dataframe.columns[section])
            except IndexError:
                return None
        return str(section + 1)


class StatsCard(QtWidgets.QFrame):
    def __init__(self, title: str, *, accent: str = "#6C83FF"):
        super().__init__()
        self.setObjectName("StatsCard")
        self._accent = accent
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)

        self.title_label = QtWidgets.QLabel(title.upper())
        self.title_label.setObjectName("StatsCardTitle")
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)

        self.value_label = QtWidgets.QLabel("—")
        self.value_label.setObjectName("StatsCardValue")
        self.value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.setMinimumWidth(150)
        self.setMaximumHeight(100)

        layout.addWidget(self.title_label)
        layout.addStretch(1)
        layout.addWidget(self.value_label)

        self._apply_card_style()

    def set_value(self, value: str) -> None:
        self.value_label.setText(str(value))

    def _apply_card_style(self) -> None:
        self.setStyleSheet(
            f"""
            QFrame#StatsCard {{
                border-radius: 14px;
                background-color: rgba(18, 21, 32, 0.9);
                border: 1px solid rgba(120, 130, 180, 0.12);
                border-left: 4px solid {self._accent};
                padding: 8px 10px;
            }}
            QFrame#StatsCard::hover {{
                border: 1px solid rgba(108, 131, 255, 0.25);
            }}
            QLabel#StatsCardTitle {{
                color: #8a93c9;
                font-size: 10px;
                letter-spacing: 0.8px;
            }}
            QLabel#StatsCardValue {{
                color: #f4f6ff;
                font-size: 22px;
                font-weight: 600;
            }}
            """
        )


class PetriNetGraphWidget(pg.GraphicsLayoutWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setBackground("transparent")

        self._view = self.addViewBox()
        self._view.setMenuEnabled(False)
        self._view.setAspectLocked(False)
        self._view.enableAutoRange(pg.ViewBox.XYAxes, enable=True)

        self._graph_item = pg.GraphItem()
        self._view.addItem(self._graph_item)
        self._graph_item.scatter.sigClicked.connect(self._handle_node_click)

        self._node_info: Dict[int, str] = {}
        self._graph: Optional[nx.DiGraph] = None
        self._artifacts: Optional[process_analysis.ProcessModelArtifacts] = None
        self._ordered_nodes: list[Any] = []
        self._ordered_edges: list[tuple[Any, Any, Dict[str, Any]]] = []
        self._positions: Optional[np.ndarray] = None
        self._layout_mode = "force"
        self._metric_mode = "frequency"
        self._edge_scale = 1.0
        self._initial_marking = None
        self._final_marking = None
        self._edge_arrows: list[pg.ArrowItem] = []

        self._empty_label = QtWidgets.QGraphicsSimpleTextItem("Select a log to visualise its Petri net.")
        font = QtGui.QFont("Segoe UI", 11)
        self._empty_label.setFont(font)
        self._empty_label.setBrush(QtGui.QColor("#9aa5d9"))
        self._view.addItem(self._empty_label)
        self._empty_label.setPos(-180, -20)
        self._empty_label.setVisible(True)

    def clear(self) -> None:
        self._graph_item.setData(pos=np.empty((0, 2)), adj=None)
        self._node_info.clear()
        for arrow in self._edge_arrows:
            self._view.removeItem(arrow)
        self._edge_arrows = []
        self._graph = None
        self._artifacts = None
        self._ordered_nodes = []
        self._ordered_edges = []
        self._positions = None
        self._initial_marking = None
        self._final_marking = None
        self._empty_label.setVisible(True)
        self._view.autoRange()

    def update_graph(
        self,
        artifacts: process_analysis.ProcessModelArtifacts,
        *,
        metric_mode: str = "frequency",
        layout_mode: str = "force",
        edge_scale: float = 1.0,
    ) -> None:
        net = artifacts.net
        nodes = list(net.places) + list(net.transitions)
        if not nodes:
            self.clear()
            return

        graph = nx.DiGraph()
        for place in net.places:
            graph.add_node(place)
        for transition in net.transitions:
            graph.add_node(transition)
        for arc in net.arcs:
            graph.add_edge(arc.source, arc.target, arc=arc)

        if graph.number_of_nodes() == 0:
            self.clear()
            return

        self._graph = graph
        self._artifacts = artifacts
        self._ordered_nodes = list(graph.nodes())
        self._ordered_edges = list(graph.edges(data=True))
        self._layout_mode = layout_mode
        self._metric_mode = metric_mode
        self._edge_scale = edge_scale
        self._initial_marking = artifacts.initial_marking
        self._final_marking = artifacts.final_marking

        self._positions = self._compute_positions(layout_mode)
        self._render()

    def set_layout_mode(self, layout_mode: str) -> None:
        if self._graph is None:
            return
        self._layout_mode = layout_mode
        self._positions = self._compute_positions(layout_mode)
        self._render()

    def set_metric_mode(self, metric_mode: str) -> None:
        if self._graph is None:
            return
        self._metric_mode = metric_mode
        self._render()

    def set_edge_scale(self, edge_scale: float) -> None:
        self._edge_scale = edge_scale
        if self._graph is not None:
            self._render()

    def reset_view(self) -> None:
        self._view.autoRange()

    def _compute_positions(self, layout_mode: str) -> np.ndarray:
        assert self._graph is not None
        if layout_mode == "hierarchical":
            return self._hierarchical_layout()
        if layout_mode == "circular":
            layout = nx.circular_layout(self._graph)
        elif layout_mode == "kamada":
            layout = nx.kamada_kawai_layout(self._graph)
        else:
            layout = nx.spring_layout(self._graph, seed=42, k=1 / max(self._graph.number_of_nodes(), 1))
        return np.array([layout[node] for node in self._ordered_nodes], dtype=float)

    def _hierarchical_layout(self) -> np.ndarray:
        assert self._graph is not None
        levels: Dict[Any, int] = {}
        queue: deque[tuple[Any, int]] = deque()

        if self._initial_marking:
            for place in self._initial_marking:
                queue.append((place, 0))
        else:
            # fallback to arbitrary start
            queue.append((self._ordered_nodes[0], 0))

        while queue:
            node, level = queue.popleft()
            if node in levels:
                continue
            levels[node] = level
            for succ in self._graph.successors(node):
                queue.append((succ, level + 1))

        # assign remaining nodes
        max_level = max(levels.values(), default=0)
        for node in self._ordered_nodes:
            if node not in levels:
                max_level += 1
                levels[node] = max_level

        # group nodes by level and compute positions
        grouped: Dict[int, list[Any]] = {}
        for node, lvl in levels.items():
            grouped.setdefault(lvl, []).append(node)

        positions = np.zeros((len(self._ordered_nodes), 2), dtype=float)
        for lvl, nodes in grouped.items():
            count = len(nodes)
            for idx, node in enumerate(nodes):
                x = float(lvl)
                y = float(idx - (count - 1) / 2)
                positions[self._ordered_nodes.index(node)] = [x, y]

        # normalise to fit nicely
        if len(grouped) > 1:
            positions[:, 0] = positions[:, 0] / max(grouped.keys()) * len(grouped)
        positions /= max(abs(positions).max(), 1)
        return positions

    def _render(self) -> None:
        if self._graph is None or self._positions is None:
            self.clear()
            return

        index_map = {node: idx for idx, node in enumerate(self._ordered_nodes)}
        if self._ordered_edges:
            adj = np.array([[index_map[src], index_map[dst]] for src, dst, _ in self._ordered_edges], dtype=int)
        else:
            adj = None

        decorations = self._select_decorations(self._metric_mode)
        frequency_decorations = self._artifacts.frequency_decorations if self._artifacts else {}
        performance_decorations = self._artifacts.performance_decorations if self._artifacts else {}

        symbols: list[str] = []
        sizes: list[float] = []
        brushes: list[QtGui.QBrush] = []
        pens: list[QtGui.QPen] = []
        labels: list[str] = []
        label_colors: list[QtGui.QColor] = []
        node_info: Dict[int, str] = {}

        for node in self._ordered_nodes:
            idx = index_map[node]
            stats = frequency_decorations.get(node, {}) if isinstance(node, PetriTransition) else {}
            if isinstance(node, PetriPlace):
                symbols.append("o")
                sizes.append(18)
                if self._initial_marking and node in self._initial_marking:
                    brushes.append(pg.mkBrush("#60D394"))
                    pens.append(pg.mkPen("#0f1321", width=1.2))
                    role = "Start place"
                elif self._final_marking and node in self._final_marking:
                    brushes.append(pg.mkBrush("#F25F5C"))
                    pens.append(pg.mkPen("#0f1321", width=1.2))
                    role = "Completion place"
                else:
                    brushes.append(pg.mkBrush("#4D96FF"))
                    pens.append(pg.mkPen("#0f1321", width=1.0))
                    role = "Intermediate place"
                label = node.name or "place"
                labels.append(label)
                label_colors.append(QtGui.QColor("#e3e7ff"))
                node_info[idx] = f"<b>{role}</b><br>{label}"
            elif isinstance(node, PetriTransition):
                label = stats.get("label", node.label or node.name or "τ")
                color = stats.get("color", "#7F5AF0")
                count = _extract_count(label)
                size = 24 + (math.log1p(count) * 3 if count is not None else 0)
                symbols.append("s")
                sizes.append(size)
                brushes.append(pg.mkBrush(color))
                pens.append(pg.mkPen("#0f1321", width=1.4))
                labels.append(label)
                label_colors.append(QtGui.QColor("#ffffff"))
                info_lines = [f"<b>Activity:</b> {label}"]
                if count is not None:
                    info_lines.append(f"Cases: {int(count)}")
                perf_stats = performance_decorations.get(node, {})
                perf_label = perf_stats.get("label")
                if perf_label:
                    info_lines.append(f"Avg duration: {perf_label}")
                node_info[idx] = "<br>".join(info_lines)
            else:
                symbols.append("o")
                sizes.append(16)
                brushes.append(pg.mkBrush("#6c83ff"))
                pens.append(pg.mkPen("#0f1321", width=1.0))
                label = str(getattr(node, "name", "node"))
                labels.append(label)
                label_colors.append(QtGui.QColor("#e3e7ff"))
                node_info[idx] = label

        edge_pens: Optional[list[QtGui.QPen]] = None
        edge_values: list[Optional[float]] = []
        if adj is not None:
            edge_pens = []
            for _, _, data in self._ordered_edges:
                arc_obj = data.get("arc")
                stats = decorations.get(arc_obj, {}) if decorations else {}
                label = stats.get("label")
                if self._metric_mode == "performance":
                    edge_values.append(_duration_to_seconds(label))
                else:
                    edge_values.append(_extract_count(label))

            min_val = min((v for v in edge_values if v is not None), default=0.0)
            max_val = max((v for v in edge_values if v is not None), default=1.0)
            span = max(max_val - min_val, 1e-6)

            for raw_value, (_, _, data) in zip(edge_values, self._ordered_edges):
                arc_obj = data.get("arc")
                stats = decorations.get(arc_obj, {}) if decorations else {}
                width_raw = stats.get("penwidth", 1.6)
                try:
                    width = max(1.0, float(width_raw) * self._edge_scale)
                except (TypeError, ValueError):
                    width = 1.6 * self._edge_scale

                if self._metric_mode == "performance" and raw_value is not None:
                    ratio = (raw_value - min_val) / span
                    colour = _gradient_colour(ratio)
                else:
                    colour = stats.get("color", "#6f789f")
                edge_pens.append(pg.mkPen(colour, width=width))

        self._graph_item.setData(
            pos=self._positions,
            adj=adj,
            pen=None,
            symbol=symbols,
            size=sizes,
            symbolBrush=brushes,
            symbolPen=pens,
            text=labels,
            textColor=label_colors,
        )
        self._node_info = node_info
        self._empty_label.setVisible(False)
        self._update_arrows(adj, edge_pens, index_map, decorations, edge_values)
        self._view.autoRange()

    def _select_decorations(self, metric_mode: str) -> Optional[Dict[Any, Any]]:
        if self._artifacts is None:
            return None
        if metric_mode == "performance":
            return self._artifacts.performance_decorations or {}
        if metric_mode == "frequency":
            return self._artifacts.frequency_decorations or {}
        return {}

    def _update_arrows(
        self,
        adj: Optional[np.ndarray],
        pens_edges: Optional[list[QtGui.QPen]],
        index_map: Dict[Any, int],
        decorations: Optional[Dict[Any, Any]],
        values: Optional[list[Optional[float]]],
    ) -> None:
        for arrow in self._edge_arrows:
            self._view.removeItem(arrow)
        self._edge_arrows = []

        if adj is None:
            return

        for idx, (src, dst, data) in enumerate(self._ordered_edges):
            pen = pens_edges[idx] if pens_edges and idx < len(pens_edges) else None
            start = self._positions[index_map[src]]
            end = self._positions[index_map[dst]]
            vector = end - start
            length = np.linalg.norm(vector)
            if length < 1e-6:
                continue
            direction = vector / length
            tip_position = start + direction * (length * 0.85)
            angle = math.degrees(math.atan2(vector[1], vector[0]))
            colour = pen.color() if pen else QtGui.QColor("#6f789f")
            arrow = pg.ArrowItem(
                pos=tip_position,
                angle=angle,
                brush=colour,
                pen=pen or pg.mkPen(colour),
                headLen=10 + self._edge_scale * 2,
                tailLen=6,
            )
            arrow.setZValue(5)
            stats = decorations.get(data.get("arc"), {}) if decorations else {}
            label = stats.get("label")
            src_name = getattr(src, "label", getattr(src, "name", str(src)))
            dst_name = getattr(dst, "label", getattr(dst, "name", str(dst)))
            tooltip = f"{src_name} → {dst_name}"
            if label:
                tooltip += f"<br>{label}"
            elif values and idx < len(values) and values[idx] is not None:
                tooltip += f"<br>Value: {values[idx]:.2f}"
            arrow.setToolTip(tooltip)
            self._view.addItem(arrow)
            self._edge_arrows.append(arrow)

    def _handle_node_click(self, scatter, points):
        if not points:
            return
        idx = int(points[0].index())
        description = self._node_info.get(idx)
        if description:
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), description, self)


class VariantFlowGraphWidget(QtWidgets.QGraphicsView):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)

        self._placeholder = QtWidgets.QGraphicsTextItem("Load a log to see the most common paths.")
        font = QtGui.QFont("Segoe UI", 11)
        self._placeholder.setFont(font)
        self._placeholder.setDefaultTextColor(QtGui.QColor("#9aa5d9"))
        self._scene.addItem(self._placeholder)
        self._placeholder.setPos(20, 20)

        self._node_items: Dict[str, Dict[str, Any]] = {}
        self._edge_items: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._axes_items: List[QtWidgets.QGraphicsItem] = []
        self._legend_panel: Optional[QtWidgets.QGraphicsRectItem] = None
        self._header_item: Optional[QtWidgets.QGraphicsTextItem] = None
        self._total_cases: int = 0
        self._highlighted_sequence: Optional[List[str]] = None
        self._start_token = "__start__"
        self._end_token = "__end__"

    def clear(self) -> None:
        self._scene.clear()
        self._node_items.clear()
        self._edge_items.clear()
        self._axes_items = []
        self._legend_panel = None
        self._header_item = None
        self._total_cases = 0
        self._highlighted_sequence = None

        self._placeholder = QtWidgets.QGraphicsTextItem("Load a log to see the most common paths.")
        font = QtGui.QFont("Segoe UI", 11)
        self._placeholder.setFont(font)
        self._placeholder.setDefaultTextColor(QtGui.QColor("#9aa5d9"))
        self._scene.addItem(self._placeholder)
        self._placeholder.setPos(20, 20)

    def set_data(self, data: Dict[str, Any]) -> None:
        if not data or not data.get("edges"):
            self.clear()
            return

        self._scene.clear()
        self._node_items.clear()
        self._edge_items.clear()
        self._axes_items = []
        self._legend_panel = None
        self._header_item = None
        self._highlighted_sequence = None

        activity_freq: Dict[str, int] = data.get("activities", {})
        edges: Dict[Tuple[str, str], int] = data.get("edges", {})
        start_freq: Dict[str, int] = data.get("starts", {})
        end_freq: Dict[str, int] = data.get("ends", {})

        total_cases = sum(start_freq.values())
        if not total_cases:
            total_cases = max(activity_freq.values(), default=1)
        self._total_cases = max(total_cases, 1)

        graph = nx.DiGraph()
        graph.add_node(self._start_token)
        graph.add_node(self._end_token)

        for activity, freq in activity_freq.items():
            graph.add_node(activity, freq=freq)

        for (src, dst), weight in edges.items():
            graph.add_edge(src, dst, weight=weight)

        for act, weight in start_freq.items():
            graph.add_edge(self._start_token, act, weight=weight)

        for act, weight in end_freq.items():
            graph.add_edge(act, self._end_token, weight=weight)

        node_levels = self._assign_levels(graph, self._start_token)
        positions = self._compute_positions(node_levels)
        max_weight = max((d.get("weight", 1) for _, _, d in graph.edges(data=True)), default=1)
        max_freq = max(activity_freq.values(), default=1)

        for node, pos in positions.items():
            if node == self._start_token:
                label = "Start"
                freq = sum(start_freq.values())
                item = self._create_node_item(pos, label, freq, self._total_cases, node_type="start")
            elif node == self._end_token:
                label = "Complete"
                freq = sum(end_freq.values())
                item = self._create_node_item(pos, label, freq, self._total_cases, node_type="end")
            else:
                freq = activity_freq.get(node, 0)
                item = self._create_node_item(pos, str(node), freq, self._total_cases, node_type="activity", max_freq=max_freq)
            item.setZValue(4)
            self._scene.addItem(item)
            self._node_items[node] = {
                "item": item,
                "base_pen": QtGui.QPen(item.pen()),
                "base_opacity": 1.0,
            }

        for src, dst, attrs in graph.edges(data=True):
            if src not in self._node_items or dst not in self._node_items:
                continue
            weight = attrs.get("weight", 0)
            path_item, arrow_item, label_item = self._create_edge_items(
                self._node_items[src]["item"],
                self._node_items[dst]["item"],
                weight,
                max_weight,
                self._total_cases,
            )
            tooltip = f"{self._node_label(src)} → {self._node_label(dst)}"
            tooltip += f"<br>{weight} cases"
            if self._total_cases:
                tooltip += f" ({weight / self._total_cases:.1%})"
            path_item.setToolTip(tooltip)
            arrow_item.setToolTip(tooltip)
            path_item.setZValue(2)
            arrow_item.setZValue(3)
            if label_item:
                label_item.setZValue(5)
            self._scene.addItem(path_item)
            self._scene.addItem(arrow_item)
            if label_item:
                self._scene.addItem(label_item)
            self._edge_items[(src, dst)] = {
                "path": path_item,
                "arrow": arrow_item,
                "label": label_item,
                "base_pen": QtGui.QPen(path_item.pen()),
                "base_arrow_brush": QtGui.QBrush(arrow_item.brush()),
                "base_arrow_pen": QtGui.QPen(arrow_item.pen()),
                "base_label_color": QtGui.QColor(label_item.defaultTextColor()) if label_item else None,
            }

        xs = [pos.x() for pos in positions.values()]
        ys = [pos.y() for pos in positions.values()]
        min_x = min(xs) if xs else 0.0
        max_x = max(xs) if xs else 600.0
        min_y = min(ys) if ys else -200.0
        max_y = max(ys) if ys else 200.0
        width = max_x - min_x
        height = max_y - min_y
        scene_rect = QtCore.QRectF(
            min_x - 260,
            min_y - 200,
            width + 520 if width > 0 else 780,
            height + 420 if height > 0 else 520,
        )
        self._scene.setSceneRect(scene_rect)

        self._draw_axes(positions, node_levels, scene_rect)
        self._header_item = self._create_header(scene_rect, len(activity_freq), len(edges))
        self._legend_panel = self._create_legend(scene_rect, max_weight, self._total_cases)

        if scene_rect.width() > 0 and scene_rect.height() > 0:
            self.fitInView(scene_rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self._apply_highlight(None, None)

    def _node_label(self, node: str) -> str:
        if node == self._start_token:
            return "Start"
        if node == self._end_token:
            return "Complete"
        return str(node)

    def _assign_levels(self, graph: nx.DiGraph, start_node: str) -> Dict[str, int]:
        levels: Dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque([(start_node, 0)])
        while queue:
            node, level = queue.popleft()
            if node in levels:
                continue
            levels[node] = level
            for succ in graph.successors(node):
                queue.append((succ, level + 1))
        max_level = max(levels.values(), default=0)
        for node in graph.nodes():
            if node not in levels:
                max_level += 1
                levels[node] = max_level
        return levels

    def _compute_positions(self, levels: Dict[str, int]) -> Dict[str, QtCore.QPointF]:
        positions: Dict[str, QtCore.QPointF] = {}
        grouped: Dict[int, List[str]] = {}
        for node, level in levels.items():
            grouped.setdefault(level, []).append(node)

        x_spacing = 220
        y_spacing = 120

        for level, nodes in grouped.items():
            nodes_sorted = sorted(nodes)
            total_height = (len(nodes_sorted) - 1) * y_spacing
            for idx, node in enumerate(nodes_sorted):
                x = level * x_spacing
                y = idx * y_spacing - total_height / 2
                positions[node] = QtCore.QPointF(x, y)
        return positions

    def _create_node_item(
        self,
        pos: QtCore.QPointF,
        label: str,
        freq: int,
        total_cases: int,
        *,
        node_type: str = "activity",
        max_freq: int = 1,
    ) -> QtWidgets.QGraphicsPathItem:
        if node_type == "activity":
            height = 60 + (freq / max_freq) * 40 if max_freq else 60
            width = 190
            color = QtGui.QColor("#7F5AF0")
            pen_color = QtGui.QColor("#0f1321")
        elif node_type == "start":
            height = 64
            width = 150
            color = QtGui.QColor("#60D394")
            pen_color = QtGui.QColor("#0f1321")
        else:  # end
            height = 64
            width = 150
            color = QtGui.QColor("#F25F5C")
            pen_color = QtGui.QColor("#0f1321")

        rect = QtCore.QRectF(-width / 2, -height / 2, width, height)
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, 14, 14)
        item = QtWidgets.QGraphicsPathItem(path)
        item.setBrush(QtGui.QBrush(color))
        item.setPen(QtGui.QPen(pen_color, 1.5))
        item.setPos(pos)

        title = QtWidgets.QGraphicsTextItem(label, item)
        title_font = QtGui.QFont("Segoe UI", 11, QtGui.QFont.Weight.DemiBold)
        title.setFont(title_font)
        title.setDefaultTextColor(QtGui.QColor("#ffffff"))
        title.setPos(-width / 2 + 14, -height / 2 + 12)

        if freq and node_type == "activity":
            subtitle = QtWidgets.QGraphicsTextItem(f"{freq:,} cases", item)
            subtitle_font = QtGui.QFont("Segoe UI", 9)
            subtitle.setFont(subtitle_font)
            subtitle.setDefaultTextColor(QtGui.QColor("#d7dbff"))
            subtitle.setPos(-width / 2 + 14, -height / 2 + 34)

        pct_text = ""
        if total_cases:
            pct = freq / total_cases
            pct_text = f" ({pct:.1%})"
        item.setToolTip(f"{label}<br>{freq:,} cases{pct_text}")
        return item

    def _create_edge_items(
        self,
        src_item: QtWidgets.QGraphicsItem,
        dst_item: QtWidgets.QGraphicsItem,
        weight: int,
        max_weight: int,
        total_cases: int,
    ) -> tuple[QtWidgets.QGraphicsPathItem, QtWidgets.QGraphicsPolygonItem, Optional[QtWidgets.QGraphicsTextItem]]:
        src_rect = src_item.mapRectToScene(src_item.boundingRect())
        dst_rect = dst_item.mapRectToScene(dst_item.boundingRect())
        start = QtCore.QPointF(src_rect.right(), src_rect.center().y())
        end = QtCore.QPointF(dst_rect.left(), dst_rect.center().y())

        ctrl_offset = (end.x() - start.x()) * 0.45
        path = QtGui.QPainterPath(start)
        ctrl1 = QtCore.QPointF(start.x() + ctrl_offset, start.y())
        ctrl2 = QtCore.QPointF(end.x() - ctrl_offset, end.y())
        path.cubicTo(ctrl1, ctrl2, end)

        norm = weight / max_weight if max_weight else 0.0
        colour = _gradient_colour(norm)
        pen = QtGui.QPen(colour, 1.6 + norm * 5.4)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)

        path_item = QtWidgets.QGraphicsPathItem(path)
        path_item.setPen(pen)
        path_item.setOpacity(0.9)

        arrow_tip = QtCore.QPointF(end.x(), end.y())
        direction = QtCore.QPointF(end - ctrl2)
        length = math.hypot(direction.x(), direction.y())
        if length:
            direction /= length
        else:
            direction = QtCore.QPointF(1, 0)
        normal = QtCore.QPointF(-direction.y(), direction.x())
        arrow_size = 10 + norm * 6
        polygon = QtGui.QPolygonF(
            [
                arrow_tip,
                arrow_tip - direction * arrow_size + normal * (arrow_size / 2),
                arrow_tip - direction * arrow_size - normal * (arrow_size / 2),
            ]
        )
        arrow_item = QtWidgets.QGraphicsPolygonItem(polygon)
        arrow_item.setBrush(QtGui.QBrush(colour))
        arrow_item.setPen(QtGui.QPen(colour))
        arrow_item.setOpacity(0.95)

        label_item: Optional[QtWidgets.QGraphicsTextItem] = None
        if weight:
            midpoint = path.pointAtPercent(0.48)
            label_text = f"{weight:,}"
            if total_cases:
                label_text += f" ({weight / total_cases:.0%})"
            label_item = QtWidgets.QGraphicsTextItem(label_text)
            font = QtGui.QFont("Segoe UI", 9)
            label_item.setFont(font)
            label_item.setDefaultTextColor(QtGui.QColor("#d7dbff"))
            label_item.setPos(midpoint + QtCore.QPointF(6, -22))

        return path_item, arrow_item, label_item

    def _draw_axes(
        self,
        positions: Dict[str, QtCore.QPointF],
        levels: Dict[str, int],
        scene_rect: QtCore.QRectF,
    ) -> None:
        self._axes_items = []
        if not positions:
            return

        level_to_nodes: Dict[int, List[str]] = {}
        for node, level in levels.items():
            level_to_nodes.setdefault(level, []).append(node)

        min_y = min(pos.y() for pos in positions.values()) if positions else scene_rect.top()
        max_y = max(pos.y() for pos in positions.values()) if positions else scene_rect.bottom()
        line_top = min_y - 120
        line_bottom = max_y + 140

        guide_pen = QtGui.QPen(QtGui.QColor(70, 82, 126, 130), 1, QtCore.Qt.PenStyle.DashLine)
        label_font = QtGui.QFont("Segoe UI", 9)
        label_font.setCapitalization(QtGui.QFont.Capitalization.AllUppercase)

        for level in sorted(level_to_nodes.keys()):
            node_key = level_to_nodes[level][0]
            x = positions.get(node_key, QtCore.QPointF(level * 220, 0)).x()
            line = self._scene.addLine(x, line_top, x, line_bottom, guide_pen)
            line.setZValue(-10)
            self._axes_items.append(line)

            if any(node == self._start_token for node in level_to_nodes[level]):
                label = "Start"
            elif any(node == self._end_token for node in level_to_nodes[level]):
                label = "Complete"
            else:
                label = f"Step {level}"
            label_item = QtWidgets.QGraphicsTextItem(label)
            label_item.setDefaultTextColor(QtGui.QColor("#8a93c9"))
            label_item.setFont(label_font)
            label_item.setPos(x - 36, line_top - 28)
            label_item.setZValue(-9)
            self._scene.addItem(label_item)
            self._axes_items.append(label_item)

    def _create_header(self, scene_rect: QtCore.QRectF, activities: int, edges: int) -> Optional[QtWidgets.QGraphicsTextItem]:
        summary = f"{self._total_cases:,} cases · {activities} activities · {edges} transitions"
        header_item = QtWidgets.QGraphicsTextItem(summary)
        font = QtGui.QFont("Segoe UI", 10)
        font.setLetterSpacing(QtGui.QFont.SpacingType.PercentageSpacing, 96)
        header_item.setFont(font)
        header_item.setDefaultTextColor(QtGui.QColor("#8a93c9"))
        header_item.setPos(scene_rect.left() + 20, scene_rect.top() + 20)
        header_item.setZValue(8)
        self._scene.addItem(header_item)
        return header_item

    def _create_legend(
        self,
        scene_rect: QtCore.QRectF,
        max_weight: int,
        total_cases: int,
    ) -> Optional[QtWidgets.QGraphicsRectItem]:
        panel_width = 250
        panel_height = 150
        x = scene_rect.right() - panel_width - 40
        y = scene_rect.top() + 30

        panel = QtWidgets.QGraphicsRectItem(0, 0, panel_width, panel_height)
        panel.setBrush(QtGui.QBrush(QtGui.QColor(18, 21, 32, 235)))
        panel.setPen(QtGui.QPen(QtGui.QColor(108, 131, 255, 80)))
        panel.setZValue(12)
        panel.setPos(x, y)
        self._scene.addItem(panel)

        title = QtWidgets.QGraphicsTextItem("Legend", panel)
        title_font = QtGui.QFont("Segoe UI", 10, QtGui.QFont.Weight.DemiBold)
        title.setFont(title_font)
        title.setDefaultTextColor(QtGui.QColor("#f3f5ff"))
        title.setPos(16, 12)

        subtitle = QtWidgets.QGraphicsTextItem("Hover points or paths to see coverage (% of cases).", panel)
        subtitle_font = QtGui.QFont("Segoe UI", 8)
        subtitle.setFont(subtitle_font)
        subtitle.setDefaultTextColor(QtGui.QColor("#9aa5d9"))
        subtitle.setPos(16, 34)

        def add_chip(color: QtGui.QColor, text: str, y_offset: float) -> None:
            chip = QtWidgets.QGraphicsRectItem(0, 0, 18, 18, panel)
            chip.setBrush(QtGui.QBrush(color))
            chip.setPen(QtGui.QPen(QtGui.QColor("#0f1321")))
            chip.setPos(18, y_offset)
            label = QtWidgets.QGraphicsTextItem(text, panel)
            label.setDefaultTextColor(QtGui.QColor("#d7dbff"))
            label.setFont(QtGui.QFont("Segoe UI", 9))
            label.setPos(44, y_offset - 2)

        add_chip(QtGui.QColor("#60D394"), "Case entry", 62)
        add_chip(QtGui.QColor("#7F5AF0"), "Activity step (size tracks usage)", 86)
        add_chip(QtGui.QColor("#F25F5C"), "Case completion", 110)

        gradient = QtGui.QLinearGradient(0, 0, 120, 0)
        gradient.setColorAt(0, QtGui.QColor("#2CB1BC"))
        gradient.setColorAt(1, QtGui.QColor("#F25F5C"))
        edge_pen = QtGui.QPen(QtGui.QBrush(gradient), 5)
        edge_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        line = QtWidgets.QGraphicsLineItem(0, 0, 120, 0, panel)
        line.setPen(edge_pen)
        line.setPos(18, 136)

        edge_label = QtWidgets.QGraphicsTextItem("Edge tint ∝ slower transitions", panel)
        edge_label.setDefaultTextColor(QtGui.QColor("#d7dbff"))
        edge_label.setFont(QtGui.QFont("Segoe UI", 9))
        edge_label.setPos(148, 128)

        if total_cases:
            panel.setToolTip(f"{total_cases:,} total cases visualised (max edge weight {max_weight:,}).")
        return panel

    def _apply_highlight(
        self,
        active_nodes: Optional[Set[str]],
        active_edges: Optional[Set[Tuple[str, str]]],
    ) -> None:
        for node_key, record in self._node_items.items():
            item: QtWidgets.QGraphicsPathItem = record["item"]
            base_pen: QtGui.QPen = record["base_pen"]
            if active_nodes is None:
                item.setOpacity(record.get("base_opacity", 1.0))
                item.setPen(QtGui.QPen(base_pen))
                item.setZValue(4)
            elif node_key in active_nodes:
                highlight_pen = QtGui.QPen(base_pen)
                highlight_pen.setWidthF(base_pen.widthF() + 1.4)
                highlight_pen.setColor(QtGui.QColor("#f7f9ff"))
                item.setPen(highlight_pen)
                item.setOpacity(1.0)
                item.setZValue(7)
            else:
                item.setOpacity(0.18)
                item.setPen(QtGui.QPen(base_pen))
                item.setZValue(2)

        for edge_key, record in self._edge_items.items():
            path_item: QtWidgets.QGraphicsPathItem = record["path"]
            arrow_item: QtWidgets.QGraphicsPolygonItem = record["arrow"]
            label_item: Optional[QtWidgets.QGraphicsTextItem] = record["label"]
            base_pen: QtGui.QPen = record["base_pen"]
            base_arrow_brush: QtGui.QBrush = record["base_arrow_brush"]
            base_arrow_pen: QtGui.QPen = record["base_arrow_pen"]
            base_label_color: Optional[QtGui.QColor] = record["base_label_color"]

            if active_edges is None:
                path_item.setPen(QtGui.QPen(base_pen))
                path_item.setOpacity(0.9)
                path_item.setZValue(2)
                arrow_item.setBrush(QtGui.QBrush(base_arrow_brush))
                arrow_item.setPen(QtGui.QPen(base_arrow_pen))
                arrow_item.setOpacity(0.95)
                arrow_item.setZValue(3)
                if label_item and base_label_color:
                    label_item.setDefaultTextColor(QtGui.QColor(base_label_color))
                    label_item.setOpacity(1.0)
                    label_item.setZValue(5)
            elif edge_key in active_edges:
                highlight_color = QtGui.QColor("#FFD166")
                highlight_pen = QtGui.QPen(highlight_color)
                highlight_pen.setWidthF(max(base_pen.widthF() + 2.2, 4.0))
                highlight_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
                path_item.setPen(highlight_pen)
                path_item.setOpacity(1.0)
                path_item.setZValue(8)
                arrow_item.setBrush(QtGui.QBrush(highlight_color))
                arrow_item.setPen(QtGui.QPen(highlight_color))
                arrow_item.setOpacity(1.0)
                arrow_item.setZValue(9)
                if label_item:
                    label_item.setDefaultTextColor(QtGui.QColor("#FFE39A"))
                    label_item.setOpacity(1.0)
                    label_item.setZValue(10)
            else:
                path_item.setPen(QtGui.QPen(base_pen))
                path_item.setOpacity(0.12)
                path_item.setZValue(1)
                arrow_item.setBrush(QtGui.QBrush(base_arrow_brush))
                arrow_item.setPen(QtGui.QPen(base_arrow_pen))
                arrow_item.setOpacity(0.18)
                arrow_item.setZValue(1)
                if label_item and base_label_color:
                    label_item.setDefaultTextColor(QtGui.QColor(base_label_color))
                    label_item.setOpacity(0.2)
                    label_item.setZValue(1)

    def set_highlighted_sequence(self, sequence: Optional[List[str]]) -> bool:
        if not self._node_items or not self._edge_items:
            return False
        if not sequence:
            self._highlighted_sequence = None
            self._apply_highlight(None, None)
            return True

        normalised = [step.strip() for step in sequence if isinstance(step, str) and step.strip()]
        if not normalised:
            self._highlighted_sequence = None
            self._apply_highlight(None, None)
            return False

        active_nodes: Set[str] = {self._start_token, self._end_token}
        active_nodes.update(normalised)

        active_edges: Set[Tuple[str, str]] = set()
        previous = self._start_token
        for step in normalised:
            key = (previous, step)
            if key in self._edge_items:
                active_edges.add(key)
            previous = step
        end_key = (previous, self._end_token)
        if end_key in self._edge_items:
            active_edges.add(end_key)

        if not active_edges:
            self._highlighted_sequence = None
            self._apply_highlight(None, None)
            return False

        self._highlighted_sequence = normalised
        self._apply_highlight(active_nodes, active_edges)
        return True

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # type: ignore[override]
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
            self.scale(factor, factor)
            event.accept()
        else:
            super().wheelEvent(event)


class _FlowNodeItem(QtWidgets.QGraphicsPathItem):
    def __init__(self, widget: "InteractiveProcessFlowWidget", key: str, path: QtGui.QPainterPath):
        super().__init__(path)
        self._widget = widget
        self._key = key
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:  # type: ignore[override]
        self._widget._handle_node_hover(self._key)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:  # type: ignore[override]
        self._widget._handle_node_hover(None)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:  # type: ignore[override]
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._widget._handle_node_click(self._key)
        super().mousePressEvent(event)


class _FlowEdgePathItem(QtWidgets.QGraphicsPathItem):
    def __init__(self, widget: "InteractiveProcessFlowWidget", key: tuple[str, str], path: QtGui.QPainterPath):
        super().__init__(path)
        self._widget = widget
        self._key = key
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:  # type: ignore[override]
        self._widget._handle_edge_hover(self._key)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:  # type: ignore[override]
        self._widget._handle_edge_hover(None)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:  # type: ignore[override]
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._widget._handle_edge_click(self._key)
        super().mousePressEvent(event)


class _FlowEdgeArrowItem(QtWidgets.QGraphicsPolygonItem):
    def __init__(self, widget: "InteractiveProcessFlowWidget", key: tuple[str, str], polygon: QtGui.QPolygonF):
        super().__init__(polygon)
        self._widget = widget
        self._key = key
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:  # type: ignore[override]
        self._widget._handle_edge_hover(self._key)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:  # type: ignore[override]
        self._widget._handle_edge_hover(None)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:  # type: ignore[override]
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._widget._handle_edge_click(self._key)
        super().mousePressEvent(event)


class InteractiveProcessFlowWidget(QtWidgets.QGraphicsView):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        gradient = QtGui.QLinearGradient(0, 0, 0, 540)
        gradient.setColorAt(0.0, QtGui.QColor(14, 17, 28))
        gradient.setColorAt(0.5, QtGui.QColor(18, 22, 37))
        gradient.setColorAt(1.0, QtGui.QColor(10, 13, 21))
        self.setBackgroundBrush(QtGui.QBrush(gradient))

        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)

        self._placeholder = QtWidgets.QGraphicsTextItem("Load a log to explore the process pulse.")
        font = QtGui.QFont("Segoe UI", 12)
        self._placeholder.setFont(font)
        self._placeholder.setDefaultTextColor(QtGui.QColor("#9aa5d9"))
        self._scene.addItem(self._placeholder)
        self._placeholder.setPos(60, 40)

        self._data: Optional[Dict[str, Any]] = None
        self._mode: str = "frequency"
        self._total_cases: int = 0
        self._node_items: Dict[str, Dict[str, Any]] = {}
        self._edge_items: Dict[tuple[str, str], Dict[str, Any]] = {}
        self._edge_details: Dict[tuple[str, str], Dict[str, Any]] = {}
        self._node_case_counts: Dict[str, int] = {}
        self._adjacency_out: Dict[str, Set[tuple[str, str]]] = {}
        self._adjacency_in: Dict[str, Set[tuple[str, str]]] = {}
        self._highlight_active: bool = False
        self._current_hover: Optional[str] = None
        self._current_edge_hover: Optional[tuple[str, str]] = None
        self._locked_nodes: Optional[Set[str]] = None
        self._locked_edges: Optional[Set[tuple[str, str]]] = None
        self._metadata_banner: Optional[QtWidgets.QGraphicsRectItem] = None
        self._info_panel: Optional[QtWidgets.QGraphicsTextItem] = None

        self._pulse_timer = QtCore.QTimer(self)
        self._pulse_timer.timeout.connect(self._tick_pulse)
        self._pulse_phase = 0.0
        self._pulse_timer.start(90)

    def clear(self) -> None:
        self._scene.clear()
        self._placeholder = QtWidgets.QGraphicsTextItem("Load a log to explore the process pulse.")
        font = QtGui.QFont("Segoe UI", 12)
        self._placeholder.setFont(font)
        self._placeholder.setDefaultTextColor(QtGui.QColor("#9aa5d9"))
        self._scene.addItem(self._placeholder)
        self._placeholder.setPos(60, 40)

        self._data = None
        self._total_cases = 0
        self._node_items.clear()
        self._edge_items.clear()
        self._edge_details.clear()
        self._node_case_counts.clear()
        self._adjacency_out.clear()
        self._adjacency_in.clear()
        self._metadata_banner = None
        self._info_panel = None
        self._highlight_active = False
        self._current_hover = None
        self._current_edge_hover = None
        self._locked_nodes = None
        self._locked_edges = None

    def set_data(self, data: Dict[str, Any]) -> None:
        self._data = data
        self._rebuild_scene()

    def set_mode(self, mode: str) -> None:
        if mode not in {"frequency", "performance"}:
            return
        if self._mode == mode:
            return
        self._mode = mode
        if self._data:
            self._rebuild_scene()

    def _rebuild_scene(self) -> None:
        self._scene.clear()
        self._node_items.clear()
        self._edge_items.clear()
        self._adjacency_out.clear()
        self._adjacency_in.clear()
        self._metadata_banner = None
        self._info_panel = None
        self._highlight_active = False
        self._current_hover = None
        self._current_edge_hover = None

        if not self._data or not self._data.get("edges"):
            self.clear()
            return

        edges: Dict[tuple[str, str], int] = self._data.get("edges", {})
        activities: Dict[str, int] = self._data.get("activities", {})
        starts: Dict[str, int] = self._data.get("starts", {})
        ends: Dict[str, int] = self._data.get("ends", {})
        performance_edges: Dict[tuple[str, str], Optional[float]] = self._data.get("performance_edges", {}) or {}
        rework_edges: Dict[tuple[str, str], int] = self._data.get("rework_edges", {}) or {}
        edge_cases: Dict[tuple[str, str], int] = self._data.get("edge_cases", {}) or {}
        node_cases: Dict[str, int] = self._data.get("node_cases", {}) or {}
        metadata: Dict[str, Any] = self._data.get("metadata", {})

        self._total_cases = int(metadata.get("total_cases") or sum(starts.values()) or 0)
        if not self._total_cases:
            self._total_cases = max(sum(activities.values()), 1)
        self._edge_details.clear()
        self._node_case_counts = {node: node_cases.get(node, 0) for node in activities.keys()}
        self._locked_nodes = None
        self._locked_edges = None

        graph = nx.DiGraph()
        start_token = "__start__"
        end_token = "__end__"
        graph.add_node(start_token)
        graph.add_node(end_token)
        for act in activities:
            graph.add_node(act)
        for (src, dst), weight in edges.items():
            graph.add_edge(src, dst, weight=weight)
        for act, weight in starts.items():
            graph.add_edge(start_token, act, weight=weight)
        for act, weight in ends.items():
            graph.add_edge(act, end_token, weight=weight)

        node_levels = self._assign_levels(graph, start_token)
        positions = self._compute_positions(node_levels)
        if not positions:
            self.clear()
            return

        max_freq = max((activities.get(node, 0) for node in activities), default=1)
        if max_freq == 0:
            max_freq = 1
        max_edge_volume = metadata.get("max_edge_weight", max_freq) or 1
        perf_max = metadata.get("max_edge_duration") or 0
        perf_min = metadata.get("min_edge_duration") or 0
        if perf_max == perf_min:
            perf_min = 0

        # draw nodes
        node_z = 8
        for node, pos in positions.items():
            if node == start_token:
                freq = sum(starts.values())
                item = self._create_node_item(node, pos, freq, node_type="start")
            elif node == end_token:
                freq = sum(ends.values())
                item = self._create_node_item(node, pos, freq, node_type="end")
            else:
                freq = activities.get(node, 0)
                item = self._create_node_item(node, pos, freq, node_type="activity", max_freq=max_freq)
            item.setZValue(node_z)
            self._scene.addItem(item)
            self._node_items[node] = {
                "item": item,
                "frequency": freq,
                "default_pen": QtGui.QPen(item.pen()),
                "default_opacity": item.opacity(),
            }
            if node in self._node_case_counts:
                self._node_items[node]["cases"] = self._node_case_counts[node]
            self._adjacency_out.setdefault(node, set())
            self._adjacency_in.setdefault(node, set())

        # edges
        for src, dst, data in graph.edges(data=True):
            weight = data.get("weight", 0)
            if src not in self._node_items or dst not in self._node_items:
                continue
            key = (src, dst)
            perf_value = performance_edges.get(key)
            # ensure loops derived from rework data are tracked even if absent in edges dict
            is_rework = rework_edges.get(key, 0) > 0 or src == dst
            path_item, glow_item, arrow_item, label_item = self._create_edge_items(
                src,
                dst,
                positions[src],
                positions[dst],
                weight,
                max_edge_volume,
                perf_value,
                perf_min,
                perf_max,
                is_rework=is_rework,
            )
            tooltip = self._format_edge_tooltip(src, dst, weight, perf_value, is_rework=is_rework)
            for gfx in (path_item, glow_item, arrow_item):
                gfx.setToolTip(tooltip)
            if label_item:
                label_item.setToolTip(tooltip)

            self._scene.addItem(glow_item)
            self._scene.addItem(path_item)
            self._scene.addItem(arrow_item)
            if label_item:
                self._scene.addItem(label_item)

            self._edge_items[key] = {
                "path": path_item,
                "glow": glow_item,
                "arrow": arrow_item,
                "label": label_item,
                "frequency": weight,
                "performance": perf_value,
                "is_rework": is_rework,
                "base_pen": QtGui.QPen(path_item.pen()),
                "base_glow_pen": QtGui.QPen(glow_item.pen()),
                "base_arrow_brush": QtGui.QBrush(arrow_item.brush()),
                "base_arrow_pen": QtGui.QPen(arrow_item.pen()),
                "base_label_color": QtGui.QColor(label_item.defaultTextColor()) if label_item else None,
            }
            self._adjacency_out.setdefault(src, set()).add(key)
            self._adjacency_in.setdefault(dst, set()).add(key)
            self._edge_details[key] = {
                "cases": edge_cases.get(key, 0),
                "rework": rework_edges.get(key, 0),
            }

        self._build_metadata_banner(metadata)
        self._build_info_panel(metadata)

        xs = [point.x() for point in positions.values()]
        ys = [point.y() for point in positions.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        padding_x = 260
        padding_y = 240
        self._scene.setSceneRect(
            QtCore.QRectF(min_x - padding_x, min_y - padding_y, (max_x - min_x) + 2 * padding_x, (max_y - min_y) + 2 * padding_y)
        )
        self.fitInView(self._scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def _assign_levels(self, graph: nx.DiGraph, start_node: str) -> Dict[str, int]:
        levels: Dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque([(start_node, 0)])
        while queue:
            node, level = queue.popleft()
            if node in levels:
                continue
            levels[node] = level
            for succ in graph.successors(node):
                queue.append((succ, level + 1))
        max_level = max(levels.values(), default=0)
        for node in graph.nodes():
            if node not in levels:
                max_level += 1
                levels[node] = max_level
        return levels

    def _compute_positions(self, levels: Dict[str, int]) -> Dict[str, QtCore.QPointF]:
        positions: Dict[str, QtCore.QPointF] = {}
        grouped: Dict[int, List[str]] = {}
        for node, level in levels.items():
            grouped.setdefault(level, []).append(node)

        x_spacing = 240
        y_spacing = 140
        for level, nodes in grouped.items():
            nodes_sorted = sorted(nodes)
            total_height = (len(nodes_sorted) - 1) * y_spacing
            for idx, node in enumerate(nodes_sorted):
                x = level * x_spacing
                y = idx * y_spacing - total_height / 2
                positions[node] = QtCore.QPointF(x, y)
        return positions

    def _create_node_item(
        self,
        key: str,
        pos: QtCore.QPointF,
        freq: int,
        *,
        node_type: str,
        max_freq: int = 1,
    ) -> _FlowNodeItem:
        if node_type == "start":
            base_color = QtGui.QColor("#00E5FF")
            border_color = QtGui.QColor("#0f1321")
            width, height = 160, 70
            label = "Launch"
        elif node_type == "end":
            base_color = QtGui.QColor("#FF3F8E")
            border_color = QtGui.QColor("#0f1321")
            width, height = 160, 70
            label = "Complete"
        else:
            base_color = QtGui.QColor("#7F5AF0")
            border_color = QtGui.QColor("#0f1321")
            width = 210
            gain = freq / max_freq if max_freq else 0
            height = 68 + gain * 40
            label = key

        rect = QtCore.QRectF(-width / 2, -height / 2, width, height)
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, 18, 18)
        item = _FlowNodeItem(self, key, path)
        gradient = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, base_color.lighter(120))
        gradient.setColorAt(1.0, base_color.darker(150))
        item.setBrush(QtGui.QBrush(gradient))
        pen = QtGui.QPen(border_color, 1.6)
        item.setPen(pen)
        item.setPos(pos)
        item.setOpacity(0.95)

        title = QtWidgets.QGraphicsTextItem(label, item)
        title_font = QtGui.QFont("Segoe UI", 11, QtGui.QFont.Weight.DemiBold)
        title.setFont(title_font)
        title.setDefaultTextColor(QtGui.QColor("#FFFFFF"))
        title.setPos(-width / 2 + 16, -height / 2 + 10)

        if node_type == "activity":
            subtitle = QtWidgets.QGraphicsTextItem(f"{freq:,} cases", item)
            subtitle_font = QtGui.QFont("Segoe UI", 9)
            subtitle.setFont(subtitle_font)
            subtitle.setDefaultTextColor(QtGui.QColor("#cad2ff"))
            subtitle.setPos(-width / 2 + 16, -height / 2 + 34)

        pct_text = ""
        if self._total_cases:
            pct_text = f" ({freq / self._total_cases:.1%})"
        item.setToolTip(f"{label}<br>{freq:,} cases{pct_text}")
        return item

    def _create_edge_items(
        self,
        src: str,
        dst: str,
        src_pos: QtCore.QPointF,
        dst_pos: QtCore.QPointF,
        freq: int,
        max_freq: int,
        perf_value: Optional[float],
        perf_min: float,
        perf_max: float,
        *,
        is_rework: bool = False,
    ) -> tuple[_FlowEdgePathItem, QtWidgets.QGraphicsPathItem, _FlowEdgeArrowItem, Optional[QtWidgets.QGraphicsTextItem]]:
        start = QtCore.QPointF(src_pos.x() + 110, src_pos.y())
        end = QtCore.QPointF(dst_pos.x() - 110, dst_pos.y())
        ctrl_offset = (end.x() - start.x()) * 0.45
        path = QtGui.QPainterPath(start)
        ctrl1 = QtCore.QPointF(start.x() + ctrl_offset, start.y() - 40)
        ctrl2 = QtCore.QPointF(end.x() - ctrl_offset, end.y() + 40)
        path.cubicTo(ctrl1, ctrl2, end)

        freq_norm = freq / max_freq if max_freq else 0
        width = 1.8 + freq_norm * 6.5
        if is_rework:
            color = QtGui.QColor("#8f96ad")
        elif self._mode == "performance" and perf_value is not None and perf_max:
            ratio = (perf_value - perf_min) / max(perf_max - perf_min, 1)
            color = _heat_colour(1 - ratio)
        else:
            color = _heat_colour(freq_norm)

        path_item = _FlowEdgePathItem(self, (src, dst), path)
        pen = QtGui.QPen(color, width)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        if is_rework:
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        path_item.setPen(pen)
        path_item.setOpacity(0.88)

        glow = QtWidgets.QGraphicsPathItem(path)
        glow_pen = QtGui.QPen(QtGui.QColor(color.red(), color.green(), color.blue(), 120), width + 6)
        glow_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        if is_rework:
            glow_pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        glow.setPen(glow_pen)
        glow.setOpacity(0.25 if is_rework else 0.4)

        direction = QtCore.QPointF(end - ctrl2)
        length = math.hypot(direction.x(), direction.y())
        if length:
            direction /= length
        else:
            direction = QtCore.QPointF(1, 0)
        normal = QtCore.QPointF(-direction.y(), direction.x())
        arrow_size = 14 + freq_norm * 6
        arrow_tip = end
        polygon = QtGui.QPolygonF(
            [
                arrow_tip,
                arrow_tip - direction * arrow_size + normal * (arrow_size / 2),
                arrow_tip - direction * arrow_size - normal * (arrow_size / 2),
            ]
        )
        arrow_item = _FlowEdgeArrowItem(self, (src, dst), polygon)
        arrow_item.setBrush(QtGui.QBrush(color))
        arrow_item.setPen(QtGui.QPen(color))
        arrow_item.setOpacity(0.92)
        if is_rework:
            arrow_item.setOpacity(0.75)

        label_item: Optional[QtWidgets.QGraphicsTextItem] = None
        label_text = self._format_edge_label(freq, perf_value, is_rework=is_rework)
        if label_text:
            midpoint = path.pointAtPercent(0.45)
            label_item = QtWidgets.QGraphicsTextItem(label_text)
            label_item.setDefaultTextColor(QtGui.QColor("#e8ecff"))
            label_item.setFont(QtGui.QFont("Segoe UI", 9))
            label_item.setPos(midpoint + QtCore.QPointF(6, -26))

        return path_item, glow, arrow_item, label_item

    def _format_edge_label(self, freq: int, perf_value: Optional[float], *, is_rework: bool = False) -> str:
        if self._mode == "performance":
            if perf_value is None:
                return ""
            pretty = _format_seconds_compact(perf_value)
            if self._total_cases:
                pct = freq / self._total_cases * 100
                return f"{pretty} avg · {freq:,} cases ({pct:.1f}%)"
            return f"{pretty} avg · {freq:,} cases"
        prefix = "Rework" if is_rework else f"{freq:,} cases"
        if self._total_cases:
            pct = freq / self._total_cases * 100
            return f"{prefix} · {pct:.1f}%"
        return prefix

    def _format_edge_tooltip(
        self,
        src: str,
        dst: str,
        freq: int,
        perf_value: Optional[float],
        *,
        is_rework: bool = False,
    ) -> str:
        src_label = "Start" if src == "__start__" else ("Complete" if src == "__end__" else src)
        dst_label = "Start" if dst == "__start__" else ("Complete" if dst == "__end__" else dst)
        tooltip = f"{src_label} → {dst_label}<br>{freq:,} cases"
        if self._total_cases:
            tooltip += f" ({freq / self._total_cases:.1%})"
        if perf_value is not None:
            tooltip += f"<br>Avg duration: {_format_seconds_compact(perf_value)}"
        if is_rework:
            tooltip += "<br><span style='color:#FFAD17;'>Rework loop detected</span>"
        details = self._edge_details.get((src, dst))
        if details:
            cases = details.get("cases")
            if cases:
                tooltip += f"<br>Cases touching edge: {cases:,}"
        return tooltip

    def _build_metadata_banner(self, metadata: Dict[str, Any]) -> None:
        panel = QtWidgets.QGraphicsRectItem(0, 0, 320, 110)
        panel.setBrush(QtGui.QBrush(QtGui.QColor(18, 21, 32, 230)))
        panel.setPen(QtGui.QPen(QtGui.QColor(108, 131, 255, 90)))
        panel.setPos(self._scene.sceneRect().left() + 30, self._scene.sceneRect().top() + 30)
        panel.setZValue(20)

        title = QtWidgets.QGraphicsTextItem("Process Flow", panel)
        title.setDefaultTextColor(QtGui.QColor("#f3f5ff"))
        title.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Weight.DemiBold))
        title.setPos(16, 12)

        subtitle = QtWidgets.QGraphicsTextItem(
            f"{self._total_cases:,} cases · {metadata.get('kept_activities', 0)} activities", panel
        )
        subtitle.setDefaultTextColor(QtGui.QColor("#9aa5d9"))
        subtitle.setFont(QtGui.QFont("Segoe UI", 9))
        subtitle.setPos(16, 36)

        mode_text = "Volume (cases)" if self._mode == "frequency" else "Velocity (avg duration)"
        mode_item = QtWidgets.QGraphicsTextItem(mode_text, panel)
        mode_item.setDefaultTextColor(QtGui.QColor("#FFD166"))
        mode_item.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Weight.Medium))
        mode_item.setPos(16, 58)

        truncated = int(metadata.get("truncated") or 0)
        if truncated:
            warning = QtWidgets.QGraphicsTextItem(
                f"{truncated} low-volume steps hidden for clarity", panel
            )
            warning.setDefaultTextColor(QtGui.QColor("#FFAD17"))
            warning.setFont(QtGui.QFont("Segoe UI", 8))
            warning.setPos(16, 78)

        self._scene.addItem(panel)
        self._metadata_banner = panel

    def _build_info_panel(self, metadata: Dict[str, Any]) -> None:
        panel = QtWidgets.QGraphicsTextItem(
            "Hover nodes or paths to spotlight transitions.\nClick to lock focus. Double-click background to reset.\nCtrl + scroll to zoom, drag to pan."
        )
        panel.setDefaultTextColor(QtGui.QColor("#9aa5d9"))
        panel.setFont(QtGui.QFont("Segoe UI", 9))
        panel.setPos(self._scene.sceneRect().left() + 38, self._scene.sceneRect().top() + 150)
        panel.setOpacity(0.9)
        panel.setZValue(18)
        self._scene.addItem(panel)
        self._info_panel = panel

    def _update_info_panel(self, text: Optional[str]) -> None:
        if not self._info_panel:
            return
        if text:
            self._info_panel.setPlainText(text)
            self._info_panel.setDefaultTextColor(QtGui.QColor("#f4f6ff"))
        else:
            self._info_panel.setPlainText(
                "Hover nodes or paths to spotlight transitions.\nClick to lock focus. Double-click background to reset.\nCtrl + scroll to zoom, drag to pan."
            )
            self._info_panel.setDefaultTextColor(QtGui.QColor("#9aa5d9"))

    def _handle_node_hover(self, key: Optional[str]) -> None:
        self._current_hover = key
        if key is None:
            if self._locked_nodes:
                self._apply_highlight(None, None)
            else:
                self._update_info_panel(None)
                self._apply_highlight(None, None)
            return
        node_info = self._node_items.get(key)
        if not node_info:
            self._apply_highlight(None, None)
            return

        freq = node_info.get("frequency", 0)
        pct = freq / self._total_cases * 100 if self._total_cases else 0
        label = "Start" if key == "__start__" else ("Complete" if key == "__end__" else key)
        case_count = node_info.get("cases", self._node_case_counts.get(key, 0))
        details = f"{label}\n{freq:,} events · {pct:.1f}% of flow"
        if case_count:
            details += f"\nTouches {case_count:,} cases"
        self._update_info_panel(details)

        active_nodes = {key}
        active_edges_out = self._adjacency_out.get(key, set())
        active_edges_in = self._adjacency_in.get(key, set())
        active_edges = set(active_edges_out) | set(active_edges_in)
        for edge_key in active_edges:
            src, dst = edge_key
            active_nodes.add(src)
            active_nodes.add(dst)
        self._apply_highlight(active_nodes, active_edges)

    def _handle_node_click(self, key: str) -> None:
        node_info = self._node_items.get(key)
        if not node_info:
            return
        freq = node_info.get("frequency", 0)
        pct = freq / self._total_cases * 100 if self._total_cases else 0
        label = "Start" if key == "__start__" else ("Complete" if key == "__end__" else key)
        case_count = node_info.get("cases", self._node_case_counts.get(key, 0))
        active_nodes = {key}
        active_edges_out = self._adjacency_out.get(key, set())
        active_edges_in = self._adjacency_in.get(key, set())
        active_edges = set(active_edges_out) | set(active_edges_in)
        for edge_key in active_edges:
            active_nodes.update(edge_key)
        self._apply_highlight(active_nodes, active_edges, persist=True)
        summary = f"{label}\n{freq:,} events · {pct:.1f}% of flow"
        if case_count:
            summary += f"\nTouches {case_count:,} cases"
        summary += "\nSelection locked. Double-click background to reset."
        self._update_info_panel(summary)

    def _handle_edge_hover(self, key: Optional[tuple[str, str]]) -> None:
        self._current_edge_hover = key
        if key is None:
            if self._locked_edges:
                self._apply_highlight(None, None)
            else:
                self._update_info_panel(None)
                self._apply_highlight(None, None)
            return

        record = self._edge_items.get(key)
        if not record:
            self._apply_highlight(None, None)
            return
        freq = record.get("frequency", 0)
        perf_value = record.get("performance")
        is_rework = record.get("is_rework", False)
        tooltip = self._format_edge_tooltip(key[0], key[1], freq, perf_value, is_rework=is_rework)
        tooltip = tooltip.replace("<br>", "\n")
        self._update_info_panel(tooltip)
        active_nodes = {key[0], key[1]}
        self._apply_highlight(active_nodes, {key})

    def _handle_edge_click(self, key: tuple[str, str]) -> None:
        record = self._edge_items.get(key)
        if not record:
            return
        active_nodes = {key[0], key[1]}
        self._apply_highlight(active_nodes, {key}, persist=True)
        freq = record.get("frequency", 0)
        perf_value = record.get("performance")
        is_rework = record.get("is_rework", False)
        tooltip = self._format_edge_tooltip(key[0], key[1], freq, perf_value, is_rework=is_rework).replace("<br>", "\n")
        tooltip += "\nSelection locked. Double-click background to reset."
        self._update_info_panel(tooltip)

    def _apply_highlight(
        self,
        active_nodes: Optional[Set[str]],
        active_edges: Optional[Set[tuple[str, str]]],
        *,
        persist: bool = False,
    ) -> None:
        if persist:
            self._locked_nodes = set(active_nodes or set())
            self._locked_edges = set(active_edges or set())
        elif active_nodes is None and active_edges is None and self._locked_nodes is not None:
            active_nodes = set(self._locked_nodes)
            active_edges = set(self._locked_edges or set())

        highlight_nodes = set(active_nodes) if active_nodes else set()
        highlight_edges = set(active_edges) if active_edges else set()

        if not persist and not highlight_nodes and not highlight_edges and self._locked_nodes:
            highlight_nodes = set(self._locked_nodes)
            highlight_edges = set(self._locked_edges or set())

        if not highlight_nodes:
            highlight_nodes = None
        if not highlight_edges:
            highlight_edges = None

        self._highlight_active = bool(highlight_nodes or highlight_edges)

        for key, record in self._node_items.items():
            item: QtWidgets.QGraphicsPathItem = record["item"]
            base_pen: QtGui.QPen = record["default_pen"]
            if not highlight_nodes or key in highlight_nodes:
                highlight_pen = QtGui.QPen(base_pen)
                highlight_pen.setWidthF(base_pen.widthF() + 1.2)
                highlight_pen.setColor(QtGui.QColor("#f7f9ff"))
                item.setPen(highlight_pen)
                item.setOpacity(1.0)
                item.setZValue(12)
            else:
                item.setPen(QtGui.QPen(base_pen))
                item.setOpacity(0.25)
                item.setZValue(5)

        for key, record in self._edge_items.items():
            path_item: QtWidgets.QGraphicsPathItem = record["path"]
            glow_item: QtWidgets.QGraphicsPathItem = record["glow"]
            arrow_item: QtWidgets.QGraphicsPolygonItem = record["arrow"]
            label_item: Optional[QtWidgets.QGraphicsTextItem] = record["label"]
            base_pen: QtGui.QPen = record["base_pen"]
            base_glow_pen: QtGui.QPen = record["base_glow_pen"]
            base_arrow_brush: QtGui.QBrush = record["base_arrow_brush"]
            base_arrow_pen: QtGui.QPen = record["base_arrow_pen"]
            base_label_color: Optional[QtGui.QColor] = record["base_label_color"]

            if not highlight_edges or key in highlight_edges:
                highlight_pen = QtGui.QPen(base_pen)
                highlight_pen.setWidthF(max(base_pen.widthF() + 2.6, 3.6))
                highlight_pen.setColor(QtGui.QColor("#FFE39A"))
                path_item.setPen(highlight_pen)
                path_item.setOpacity(1.0)
                path_item.setZValue(14)

                glow_pen = QtGui.QPen(QtGui.QColor("#FFE39A"), highlight_pen.widthF() + 5)
                glow_item.setPen(glow_pen)
                glow_item.setOpacity(0.6)
                glow_item.setZValue(13)

                arrow_item.setBrush(QtGui.QBrush(QtGui.QColor("#FFE39A")))
                arrow_item.setPen(QtGui.QPen(QtGui.QColor("#FFE39A")))
                arrow_item.setOpacity(1.0)
                arrow_item.setZValue(15)

                if label_item:
                    label_item.setDefaultTextColor(QtGui.QColor("#FFE39A"))
                    label_item.setOpacity(1.0)
                    label_item.setZValue(16)
            else:
                path_item.setPen(QtGui.QPen(base_pen))
                path_item.setOpacity(0.15)
                path_item.setZValue(6)

                glow_item.setPen(QtGui.QPen(base_glow_pen))
                glow_item.setOpacity(0.12)
                glow_item.setZValue(5)

                arrow_item.setBrush(QtGui.QBrush(base_arrow_brush))
                arrow_item.setPen(QtGui.QPen(base_arrow_pen))
                arrow_item.setOpacity(0.2)
                arrow_item.setZValue(5)

                if label_item and base_label_color:
                    label_item.setDefaultTextColor(QtGui.QColor(base_label_color))
                    label_item.setOpacity(0.25)
                    label_item.setZValue(5)

    def _tick_pulse(self) -> None:
        if not self._node_items or self._highlight_active:
            return
        self._pulse_phase += 0.08
        for idx, record in enumerate(self._node_items.values()):
            item: QtWidgets.QGraphicsPathItem = record["item"]
            base_scale = 1.0
            offset = math.sin(self._pulse_phase + idx * 0.6) * 0.03
            item.setScale(base_scale + offset)

    def _clear_lock(self) -> None:
        if not (self._locked_nodes or self._locked_edges):
            return
        self._locked_nodes = None
        self._locked_edges = None
        self._update_info_panel(None)
        self._apply_highlight(None, None)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # type: ignore[override]
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
            event.accept()
        else:
            super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        self._clear_lock()
        super().mouseDoubleClickEvent(event)


def _extract_count(label: Optional[str]) -> Optional[float]:
    if not label or "(" not in label or ")" not in label:
        return None
    try:
        number = label[label.rfind("(") + 1 : label.rfind(")")]
        return float(number)
    except (ValueError, TypeError):
        return None


def _duration_to_seconds(label: Optional[str]) -> Optional[float]:
    if not label:
        return None
    cleaned = label.strip()
    if not cleaned:
        return None
    # try to parse plain float
    try:
        return float(cleaned)
    except ValueError:
        pass

    match = re.match(r"([0-9]+\.?[0-9]*)\s*([a-zA-Z]+)", cleaned)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    unit_to_seconds = {
        "ns": 1e-9,
        "us": 1e-6,
        "ms": 1e-3,
        "s": 1,
        "sec": 1,
        "m": 60,
        "min": 60,
        "h": 3600,
        "hr": 3600,
        "d": 86400,
        "day": 86400,
        "w": 604800,
    }
    factor = unit_to_seconds.get(unit)
    if factor is None:
        return None
    return value * factor


def _gradient_colour(ratio: float) -> QtGui.QColor:
    ratio = max(0.0, min(1.0, ratio))
    start = QtGui.QColor("#2CB1BC")
    end = QtGui.QColor("#F25F5C")
    r = start.red() + ratio * (end.red() - start.red())
    g = start.green() + ratio * (end.green() - start.green())
    b = start.blue() + ratio * (end.blue() - start.blue())
    return QtGui.QColor(int(r), int(g), int(b))


def _heat_colour(ratio: float) -> QtGui.QColor:
    ratio = max(0.0, min(1.0, ratio))
    start = QtGui.QColor("#2CB1BC")
    mid = QtGui.QColor("#7F5AF0")
    end = QtGui.QColor("#F25F5C")
    if ratio < 0.5:
        t = ratio / 0.5 if ratio else 0.0
        r = start.red() + t * (mid.red() - start.red())
        g = start.green() + t * (mid.green() - start.green())
        b = start.blue() + t * (mid.blue() - start.blue())
    else:
        t = (ratio - 0.5) / 0.5 if ratio < 1.0 else 1.0
        r = mid.red() + t * (end.red() - mid.red())
        g = mid.green() + t * (end.green() - mid.green())
        b = mid.blue() + t * (end.blue() - mid.blue())
    return QtGui.QColor(int(r), int(g), int(b))


def _format_seconds_compact(seconds: float) -> str:
    if seconds is None:
        return "n/a"
    total = max(0.0, float(seconds))
    if total < 1:
        return f"{total * 1000:.0f}ms"
    if total < 60:
        return f"{total:.1f}s"
    minutes_total = total / 60
    if minutes_total < 60:
        whole_minutes = int(minutes_total)
        remaining_seconds = int(total % 60)
        if remaining_seconds:
            return f"{whole_minutes}m {remaining_seconds}s"
        return f"{whole_minutes}m"
    hours_total = minutes_total / 60
    if hours_total < 24:
        whole_hours = int(hours_total)
        remaining_minutes = int(minutes_total % 60)
        if remaining_minutes:
            return f"{whole_hours}h {remaining_minutes}m"
        return f"{whole_hours}h"
    days = int(hours_total // 24)
    remaining_hours = int(hours_total % 24)
    if remaining_hours:
        return f"{days}d {remaining_hours}h"
    return f"{days}d"


class ColumnMappingDialog(QtWidgets.QDialog):
    def __init__(self, columns: list[str], preview: pd.DataFrame, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Map CSV Columns")
        self.setModal(True)
        self.resize(400, 300)

        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()

        self.case_combo = QtWidgets.QComboBox()
        self.activity_combo = QtWidgets.QComboBox()
        self.timestamp_combo = QtWidgets.QComboBox()
        self.resource_combo = QtWidgets.QComboBox()
        self.lifecycle_combo = QtWidgets.QComboBox()

        for combo in (self.case_combo, self.activity_combo, self.timestamp_combo):
            combo.addItems(columns)

        self.resource_combo.addItem("None")
        self.lifecycle_combo.addItem("None")
        for combo in (self.resource_combo, self.lifecycle_combo):
            combo.addItems(columns)

        try:
            default_case, default_act, default_ts = try_auto_detect_columns(preview)
        except LogFormatError:
            default_case = default_act = default_ts = None

        if default_case and default_case in columns:
            self.case_combo.setCurrentText(default_case)
        if default_act and default_act in columns:
            self.activity_combo.setCurrentText(default_act)
        if default_ts and default_ts in columns:
            self.timestamp_combo.setCurrentText(default_ts)

        form_layout.addRow("Case identifier", self.case_combo)
        form_layout.addRow("Activity", self.activity_combo)
        form_layout.addRow("Timestamp", self.timestamp_combo)
        form_layout.addRow("Resource (optional)", self.resource_combo)
        form_layout.addRow("Lifecycle (optional)", self.lifecycle_combo)
        layout.addLayout(form_layout)

        preview_table = QtWidgets.QTableWidget()
        preview_rows = preview.head(10)
        preview_table.setRowCount(len(preview_rows.index))
        preview_table.setColumnCount(len(preview.columns))
        preview_table.setHorizontalHeaderLabels([str(col) for col in preview.columns])
        preview_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        for row_idx, (_, row) in enumerate(preview_rows.iterrows()):
            for col_idx, value in enumerate(row):
                preview_table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(str(value)))
        preview_table.resizeColumnsToContents()
        preview_table.setMaximumHeight(150)
        layout.addWidget(preview_table)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def mapping(self) -> dict[str, Optional[str]]:
        resource = self.resource_combo.currentText()
        lifecycle = self.lifecycle_combo.currentText()
        return {
            "case": self.case_combo.currentText(),
            "activity": self.activity_combo.currentText(),
            "timestamp": self.timestamp_combo.currentText(),
            "resource": None if resource == "None" else resource,
            "lifecycle": None if lifecycle == "None" else lifecycle,
        }


class ProcessMiningApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Process Mining Explorer (PyQt)")
        self.resize(1480, 940)

        self.logger = BASE_LOGGER.getChild("ui")
        self.logger.info("ProcessMiningApp initialising.")

        self._color_palette = ["#6C83FF", "#7F5AF0", "#2CB1BC", "#F25F5C", "#FFAD17", "#60D394", "#4D96FF"]

        self._model_metric_mode = "frequency"
        self._layout_mode = "force"
        self._edge_scale = 1.0
        self._variant_selection_connected = False
        self._happy_path_sequence: List[str] = []
        self._happy_path_label: str = ""
        self._dfg_data_cache: Optional[Dict[str, Any]] = None
        self._part_flow_data: Optional[Dict[str, Any]] = None
        self._markov_image_item: Optional[pg.ImageItem] = None
        self._markov_states: List[str] = []

        self._setup_palette()
        self._apply_theme()

        pg.setConfigOption("background", "transparent")
        pg.setConfigOption("foreground", "#E7EBFF")
        pg.setConfigOption("antialias", True)

        self.log_container: Optional[EventLogContainer] = None
        self.filtered_log: Optional[EventLogContainer] = None
        self._artifacts: Optional[process_analysis.ProcessModelArtifacts] = None

        self._build_ui()
        self.logger.info("User interface initialised. Awaiting log selection.")

    # UI construction -----------------------------------------------------
    def _setup_palette(self) -> None:
        palette = QtGui.QPalette()
        base = QtGui.QColor("#0f111a")
        alternate = QtGui.QColor("#161a28")
        text = QtGui.QColor("#f4f6ff")
        disabled_text = QtGui.QColor("#6e7392")
        highlight = QtGui.QColor("#6C83FF")

        palette.setColor(QtGui.QPalette.ColorRole.Window, base)
        palette.setColor(QtGui.QPalette.ColorRole.Base, base)
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, alternate)
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#1c2032"))
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, text)
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, text)
        palette.setColor(QtGui.QPalette.ColorRole.Text, text)
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#1a1e2f"))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, text)
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, highlight)
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff"))
        palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text, disabled_text)
        palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.WindowText, disabled_text)

        self.setPalette(palette)

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background-color: #0f111a;
                color: #f4f6ff;
                font-family: "Segoe UI", "Helvetica Neue", Arial;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid rgba(108, 131, 255, 0.25);
                border-radius: 12px;
                margin-top: 16px;
                padding: 18px;
                background-color: rgba(24, 27, 42, 0.75);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                color: #9aa5d9;
                font-weight: 600;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #5a6ef5, stop:1 #7f5af0);
                border: none;
                border-radius: 8px;
                padding: 8px 14px;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #6d7cff, stop:1 #956bff);
            }
            QPushButton:pressed {
                background-color: #4a57c9;
            }
            QListWidget, QTableView, QComboBox, QDateEdit {
                background-color: rgba(18, 21, 32, 0.85);
                border: 1px solid rgba(108, 131, 255, 0.2);
                border-radius: 8px;
                padding: 4px 6px;
            }
            QListWidget::item:selected {
                background-color: rgba(108, 131, 255, 0.35);
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid rgba(108, 131, 255, 0.2);
                border-radius: 14px;
                background-color: rgba(17, 19, 29, 0.82);
                padding: 12px;
            }
            QTabBar::tab {
                background-color: rgba(26, 30, 45, 0.65);
                color: #9aa5d9;
                padding: 10px 20px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                margin-right: 4px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background-color: rgba(40, 45, 70, 0.95);
                color: #f4f6ff;
            }
            QTabBar::tab:hover {
                color: #e3e7ff;
            }
            QScrollArea {
                border: none;
            }
            QHeaderView::section {
                background-color: rgba(24, 27, 40, 0.9);
                color: #9aa5d9;
                border: none;
                padding: 6px;
            }
            QTableView {
                gridline-color: rgba(108, 131, 255, 0.12);
                selection-background-color: rgba(108, 131, 255, 0.32);
                selection-color: #ffffff;
            }
            QLabel#HeaderTitle {
                font-size: 30px;
                font-weight: 700;
                color: #f4f6ff;
            }
            QLabel#HeaderSubtitle {
                color: #8a93c9;
                font-size: 14px;
            }
            QSplitter::handle {
                background-color: rgba(108, 131, 255, 0.25);
                width: 4px;
            }
            """
        )

    def _build_ui(self) -> None:
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QtWidgets.QVBoxLayout(central_widget)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(18)

        header = self._build_header()
        root_layout.addWidget(header)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_controls_panel())
        splitter.addWidget(self._build_main_content())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root_layout.addWidget(splitter, stretch=1)

        self.statusBar().showMessage(f"Load a log to begin. Logging to {LOG_FILE_PATH.name}.")

    def _build_header(self) -> QtWidgets.QWidget:
        frame = QtWidgets.QFrame()
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("Process Mining Explorer")
        title.setObjectName("HeaderTitle")
        subtitle = QtWidgets.QLabel("Celonis-inspired insights powered by pm4py and PyQt")
        subtitle.setObjectName("HeaderSubtitle")

        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(subtitle)

        return frame

    def _build_main_content(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)

        self._build_overview_tab()
        self._build_process_model_tab()
        self._build_variant_tab()
        self._build_behaviour_tab()
        self._build_conformance_tab()

        layout.addWidget(self.tabs)
        return container

    def _build_controls_panel(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(360)

        container = QtWidgets.QWidget()
        scroll.setWidget(container)

        layout = QtWidgets.QVBoxLayout(container)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        # Data loading controls
        load_group = QtWidgets.QGroupBox("1. Load Event Log")
        load_layout = QtWidgets.QVBoxLayout(load_group)
        self.sample_button = QtWidgets.QPushButton("Load Sample Order-to-Cash Log")
        self.sample_button.clicked.connect(self.load_sample_log)
        load_layout.addWidget(self.sample_button)

        self.csv_button = QtWidgets.QPushButton("Open CSV…")
        self.csv_button.clicked.connect(self.open_csv)
        load_layout.addWidget(self.csv_button)

        self.xes_button = QtWidgets.QPushButton("Open XES…")
        self.xes_button.clicked.connect(self.open_xes)
        load_layout.addWidget(self.xes_button)

        layout.addWidget(load_group)

        # Filter controls
        filter_group = QtWidgets.QGroupBox("2. Filters")
        filter_layout = QtWidgets.QVBoxLayout(filter_group)

        filter_layout.addWidget(QtWidgets.QLabel("Cases"))
        self.case_list = QtWidgets.QListWidget()
        self.case_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        filter_layout.addWidget(self.case_list)

        filter_layout.addWidget(QtWidgets.QLabel("Activities"))
        self.activity_list = QtWidgets.QListWidget()
        self.activity_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        filter_layout.addWidget(self.activity_list)

        date_group = QtWidgets.QGroupBox("Time range")
        date_layout = QtWidgets.QFormLayout(date_group)
        self.start_date = QtWidgets.QDateEdit(calendarPopup=True)
        self.end_date = QtWidgets.QDateEdit(calendarPopup=True)
        self.start_date.setDisplayFormat("yyyy-MM-dd")
        self.end_date.setDisplayFormat("yyyy-MM-dd")
        date_layout.addRow("Start", self.start_date)
        date_layout.addRow("End", self.end_date)
        filter_layout.addWidget(date_group)

        attribute_group = QtWidgets.QGroupBox("Attribute filter")
        attribute_layout = QtWidgets.QVBoxLayout(attribute_group)
        self.attribute_combo = QtWidgets.QComboBox()
        self.attribute_combo.currentIndexChanged.connect(self._on_attribute_changed)
        attribute_layout.addWidget(self.attribute_combo)
        self.attribute_values = QtWidgets.QListWidget()
        self.attribute_values.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        attribute_layout.addWidget(self.attribute_values)
        filter_layout.addWidget(attribute_group)

        self.apply_filters_button = QtWidgets.QPushButton("Apply Filters")
        self.apply_filters_button.clicked.connect(self.apply_filters)
        filter_layout.addWidget(self.apply_filters_button)

        layout.addWidget(filter_group)
        layout.addStretch(1)

        return scroll

    def _build_overview_tab(self) -> None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        metrics = [
            ("Cases", "cases", self._color_palette[0]),
            ("Activities", "activities", self._color_palette[2]),
            ("Avg case duration", "avg_case_duration", self._color_palette[3]),
            ("Median events / case", "median_events_per_case", self._color_palette[4]),
        ]
        self.overview_cards: dict[str, StatsCard] = {}

        cards_layout = QtWidgets.QHBoxLayout()
        cards_layout.setSpacing(12)
        for label_text, key, accent in metrics:
            card = StatsCard(label_text, accent=accent)
            self.overview_cards[key] = card
            cards_layout.addWidget(card)
        cards_layout.addStretch(1)
        layout.addLayout(cards_layout)

        insights_group = QtWidgets.QGroupBox("Quick insights")
        insights_layout = QtWidgets.QVBoxLayout(insights_group)
        self.overview_insights_label = QtWidgets.QLabel()
        self.overview_insights_label.setWordWrap(True)
        self.overview_insights_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.overview_insights_label.setStyleSheet("color: #d7dbff;")
        self.overview_insights_label.setText("<i>Load a log to surface quick insights.</i>")
        insights_layout.addWidget(self.overview_insights_label)
        layout.addWidget(insights_group)

        self.tabs.addTab(widget, "Overview")

    def _build_process_model_tab(self) -> None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        # Part / item flow explorer
        self.part_flow_widget = InteractiveProcessFlowWidget()
        self.part_flow_widget.setMinimumHeight(420)

        part_container = QtWidgets.QWidget()
        part_layout = QtWidgets.QVBoxLayout(part_container)
        part_layout.setContentsMargins(0, 0, 0, 0)
        part_layout.setSpacing(10)

        part_controls = QtWidgets.QHBoxLayout()
        part_controls.setSpacing(8)
        part_controls.addWidget(QtWidgets.QLabel("Highlight"))

        self.part_flow_mode_buttons = QtWidgets.QButtonGroup(self)
        volume_btn = QtWidgets.QPushButton("Volume")
        volume_btn.setCheckable(True)
        volume_btn.setChecked(True)
        performance_btn = QtWidgets.QPushButton("Velocity")
        performance_btn.setCheckable(True)

        for btn in (volume_btn, performance_btn):
            btn.setStyleSheet(
                """
                QPushButton {
                    padding: 6px 14px;
                    border-radius: 12px;
                    background-color: rgba(36, 40, 58, 0.8);
                    color: #d7dbff;
                }
                QPushButton:checked {
                    background-color: #715AFF;
                    color: #ffffff;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background-color: rgba(113, 90, 255, 0.65);
                }
                """
            )

        self.part_flow_mode_buttons.addButton(volume_btn, 0)
        self.part_flow_mode_buttons.addButton(performance_btn, 1)
        part_controls.addWidget(volume_btn)
        part_controls.addWidget(performance_btn)
        part_controls.addStretch(1)
        self.part_flow_mode_buttons.buttonClicked.connect(self._on_part_flow_mode_changed)
        part_layout.addLayout(part_controls)

        part_layout.addWidget(self.part_flow_widget, stretch=1)
        self.part_flow_caption = QtWidgets.QLabel(
            "Select a node or path to lock focus. Rework loops appear in a muted grey."
        )
        self.part_flow_caption.setStyleSheet("color: #9aa5d9; font-size: 11px;")
        self.part_flow_caption.setWordWrap(True)
        part_layout.addWidget(self.part_flow_caption)

        part_card = self._create_chart_card(
            "Part / Item Flow Explorer",
            part_container,
            "Interactive drill-down view that surfaces the dominant part flows. Hover to explore coverage, click to lock, and watch grey arcs for rework loops.",
        )
        layout.addWidget(part_card)

        # Interactive Petri + SVG side-by-side
        self.process_svg = QtSvgWidgets.QSvgWidget()
        self.process_svg.setMinimumHeight(380)
        self.process_svg.setMinimumWidth(520)
        self.process_graph_widget = PetriNetGraphWidget()
        self.process_graph_widget.setMinimumHeight(380)
        self.process_graph_widget.setMinimumWidth(520)

        model_container = QtWidgets.QWidget()
        model_layout = QtWidgets.QVBoxLayout(model_container)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(12)

        visual_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        visual_splitter.setChildrenCollapsible(False)
        visual_splitter.addWidget(self.process_svg)
        visual_splitter.addWidget(self.process_graph_widget)
        visual_splitter.setStretchFactor(0, 1)
        visual_splitter.setStretchFactor(1, 1)
        model_layout.addWidget(visual_splitter)

        controls_row = QtWidgets.QHBoxLayout()
        controls_row.setSpacing(10)

        self.export_pnml_button = QtWidgets.QPushButton("Export Petri Net (PNML)")
        self.export_pnml_button.clicked.connect(self.export_pnml)
        controls_row.addWidget(self.export_pnml_button)

        self.export_xes_button = QtWidgets.QPushButton("Export Filtered Log (XES)")
        self.export_xes_button.clicked.connect(self.export_xes)
        controls_row.addWidget(self.export_xes_button)

        self.decoration_mode_combo = QtWidgets.QComboBox()
        self.decoration_mode_combo.addItems(["Frequency", "Performance", "Minimal"])
        self.decoration_mode_combo.setCurrentText("Frequency")
        self.decoration_mode_combo.currentTextChanged.connect(self._on_model_mode_changed)
        controls_row.addWidget(self.decoration_mode_combo)

        self.layout_mode_combo = QtWidgets.QComboBox()
        self.layout_mode_combo.addItems(["Force-directed", "Circular", "Kamada-Kawai", "Hierarchical"])
        self.layout_mode_combo.setCurrentText("Force-directed")
        self.layout_mode_combo.currentTextChanged.connect(self._on_layout_mode_changed)
        controls_row.addWidget(self.layout_mode_combo)

        controls_row.addWidget(QtWidgets.QLabel("Edge scale"))
        self.edge_scale_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.edge_scale_slider.setRange(5, 30)
        self.edge_scale_slider.setValue(10)
        self.edge_scale_slider.setToolTip("Adjust edge thickness")
        self.edge_scale_slider.valueChanged.connect(self._on_edge_scale_changed)
        controls_row.addWidget(self.edge_scale_slider)

        self.reset_graph_button = QtWidgets.QPushButton("Reset View")
        self.reset_graph_button.clicked.connect(self.process_graph_widget.reset_view)
        controls_row.addWidget(self.reset_graph_button)
        controls_row.addStretch(1)
        model_layout.addLayout(controls_row)

        petri_card = self._create_chart_card(
            "Interactive Petri Net",
            model_container,
            "Switch overlays, layout styles, and export the underlying Petri net or filtered log.",
        )
        layout.addWidget(petri_card)

        # BPMN representation
        self.bpmn_widget = QtSvgWidgets.QSvgWidget()
        self.bpmn_widget.setMinimumHeight(360)
        bpmn_card = self._create_chart_card(
            "BPMN Blueprint",
            self.bpmn_widget,
            "Block-structured BPMN view converted from the discovered model for business-friendly communication.",
        )
        layout.addWidget(bpmn_card)

        # Markov heatmap
        self.markov_plot = pg.PlotWidget()
        self._configure_plot_widget(self.markov_plot)
        self.markov_plot.setMinimumHeight(340)
        self.markov_plot.setMenuEnabled(False)
        self.markov_plot.setMouseEnabled(x=False, y=False)
        self.markov_plot.getPlotItem().showGrid(x=False, y=False)
        self._markov_image_item: Optional[pg.ImageItem] = None
        self.markov_plot.getPlotItem().setTitle(
            "<span style='color:#e3e7ff;font-size:13pt;'>Transition Probabilities</span>"
        )
        markov_container = QtWidgets.QWidget()
        markov_layout = QtWidgets.QVBoxLayout(markov_container)
        markov_layout.setContentsMargins(0, 0, 0, 0)
        markov_layout.setSpacing(6)
        markov_layout.addWidget(self.markov_plot, stretch=1)
        self.markov_caption = QtWidgets.QLabel("Load a log to project transition probabilities.")
        self.markov_caption.setWordWrap(True)
        self.markov_caption.setStyleSheet("color: #9aa5d9; font-size: 11px;")
        markov_layout.addWidget(self.markov_caption)
        markov_card = self._create_chart_card(
            "Markov Flow Heatmap",
            markov_container,
            "Row-normalised transition probabilities for the busiest steps, exposing dominant loops and exits.",
        )
        layout.addWidget(markov_card)
        self._clear_markov_heatmap()

        # Declare constraints table
        self.declare_table = QtWidgets.QTableView()
        self.declare_model = PandasTableModel()
        self.declare_table.setModel(self.declare_model)
        self.declare_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.declare_table.verticalHeader().setVisible(False)
        self.declare_table.setAlternatingRowColors(True)
        declare_container = QtWidgets.QWidget()
        declare_layout = QtWidgets.QVBoxLayout(declare_container)
        declare_layout.setContentsMargins(0, 0, 0, 0)
        declare_layout.setSpacing(6)
        declare_layout.addWidget(self.declare_table, stretch=1)
        self.declare_caption = QtWidgets.QLabel("Deriving constraint catalogue…")
        self.declare_caption.setStyleSheet("color: #9aa5d9; font-size: 11px;")
        self.declare_caption.setWordWrap(True)
        declare_layout.addWidget(self.declare_caption)
        declare_card = self._create_chart_card(
            "Declare Constraints",
            declare_container,
            "Auto-derived obligations (response, precedence, coexistence) to explain behavioural expectations.",
        )
        layout.addWidget(declare_card)

        legend_group = QtWidgets.QGroupBox("How to read this model")
        legend_layout = QtWidgets.QVBoxLayout(legend_group)

        self.process_legend_label = QtWidgets.QLabel(
            """
            <ul>
                <li>The <b>part / item flow explorer</b> surfaces dominant transitions and grey rework loops with interactive drill-down.</li>
                <li>The <b>interactive Petri net</b> shows tokens and branching logic. Adjust overlays to focus on throughput or counts.</li>
                <li>The <b>BPMN blueprint</b> restates the model in a business-friendly diagram ready for stakeholders.</li>
                <li>The <b>Markov heatmap</b> highlights which steps typically follow each other, revealing loops and exits.</li>
                <li><b>Declare constraints</b> summarise obligations such as “Approve precedes Ship” with confidence levels.</li>
            </ul>
            """
        )
        self.process_legend_label.setWordWrap(True)
        self.process_legend_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        legend_layout.addWidget(self.process_legend_label)

        self.process_summary_label = QtWidgets.QLabel("Load a log to see process highlights.")
        self.process_summary_label.setWordWrap(True)
        self.process_summary_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        legend_layout.addWidget(self.process_summary_label)

        layout.addWidget(legend_group)
        layout.addStretch(1)

        self.tabs.addTab(widget, "Process Flow")

    def _build_variant_tab(self) -> None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        controls = QtWidgets.QHBoxLayout()
        self.variant_search_input = QtWidgets.QLineEdit()
        self.variant_search_input.setPlaceholderText("Search variants…")
        controls.addWidget(QtWidgets.QLabel("Search"))
        controls.addWidget(self.variant_search_input)

        self.variant_top_spin = QtWidgets.QSpinBox()
        self.variant_top_spin.setRange(1, 100)
        self.variant_top_spin.setValue(10)
        controls.addWidget(QtWidgets.QLabel("Top N"))
        controls.addWidget(self.variant_top_spin)

        controls.addStretch(1)

        layout.addLayout(controls)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.variant_table = QtWidgets.QTableView()
        self.variant_model = PandasTableModel()
        self.variant_table.setModel(self.variant_model)
        self.variant_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.variant_table.verticalHeader().setVisible(False)
        self.variant_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.variant_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        left_layout.addWidget(self.variant_table)

        splitter.addWidget(left_container)

        right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        self.variant_summary_label = QtWidgets.QLabel("Select a variant to see details.")
        self.variant_summary_label.setWordWrap(True)
        self.variant_summary_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        right_layout.addWidget(self.variant_summary_label)

        self.variant_flow_meta = QtWidgets.QLabel("")
        self.variant_flow_meta.setWordWrap(True)
        self.variant_flow_meta.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.variant_flow_meta.setStyleSheet("color: #9aa5d9; font-size: 11px;")
        right_layout.addWidget(self.variant_flow_meta)

        self.variant_flow_widget = VariantFlowGraphWidget()
        self.variant_flow_widget.setMinimumHeight(360)
        right_layout.addWidget(
            self._create_chart_card(
                "Path Flow",
                self.variant_flow_widget,
                "Visualises the most common transitions in the filtered log.",
            )
        )

        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        self.tabs.addTab(widget, "Variants")
        self.variant_search_input.textChanged.connect(self._on_variant_controls_changed)
        self.variant_top_spin.valueChanged.connect(self._on_variant_controls_changed)
        if self.variant_table.selectionModel():
            self.variant_table.selectionModel().selectionChanged.connect(self._on_variant_selection_changed)

    def _build_behaviour_tab(self) -> None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        self.dfg_flow_widget = InteractiveProcessFlowWidget()
        self.dfg_flow_widget.setMinimumSize(460, 380)

        dfg_container = QtWidgets.QWidget()
        dfg_container_layout = QtWidgets.QVBoxLayout(dfg_container)
        dfg_container_layout.setContentsMargins(0, 0, 0, 0)
        dfg_container_layout.setSpacing(10)

        mode_bar = QtWidgets.QHBoxLayout()
        mode_bar.setSpacing(8)
        mode_label = QtWidgets.QLabel("Highlight")
        mode_label.setStyleSheet("color: #9aa5d9; font-weight: 500;")
        mode_bar.addWidget(mode_label)

        self.dfg_mode_buttons = QtWidgets.QButtonGroup(self)
        self.dfg_mode_buttons.setExclusive(True)

        def build_mode_button(text: str, mode: str) -> QtWidgets.QPushButton:
            btn = QtWidgets.QPushButton(text)
            btn.setCheckable(True)
            btn.setProperty("mode", mode)
            btn.setStyleSheet(
                """
                QPushButton {
                    padding: 6px 14px;
                    border-radius: 12px;
                    background-color: rgba(36, 40, 58, 0.8);
                    color: #d7dbff;
                }
                QPushButton:checked {
                    background-color: #715AFF;
                    color: #ffffff;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background-color: rgba(113, 90, 255, 0.65);
                }
                """
            )
            idx = 0 if mode == "frequency" else 1
            self.dfg_mode_buttons.addButton(btn, idx)
            return btn

        self.dfg_freq_button = build_mode_button("Volume", "frequency")
        self.dfg_freq_button.setChecked(True)
        self.dfg_perf_button = build_mode_button("Velocity", "performance")

        mode_bar.addWidget(self.dfg_freq_button)
        mode_bar.addWidget(self.dfg_perf_button)
        mode_bar.addStretch(1)

        self.dfg_meta_label = QtWidgets.QLabel("")
        self.dfg_meta_label.setStyleSheet("color: #9aa5d9; font-size: 11px;")
        self.dfg_meta_label.setWordWrap(True)

        dfg_container_layout.addLayout(mode_bar)
        dfg_container_layout.addWidget(self.dfg_flow_widget, stretch=1)
        dfg_container_layout.addWidget(self.dfg_meta_label)

        dfg_card = self._create_chart_card(
            "Process Pulse Explorer",
            dfg_container,
            "Arcade-inspired directly-follows explorer. Zoom, pan, and hover to spotlight busy or slow transitions.",
        )
        layout.addWidget(dfg_card)
        self.dfg_mode_buttons.buttonClicked.connect(self._on_dfg_mode_changed)

        charts_row = QtWidgets.QHBoxLayout()
        charts_row.setSpacing(16)

        self.throughput_plot = pg.PlotWidget()
        self._configure_plot_widget(self.throughput_plot)
        throughput_card = self._create_chart_card(
            "Case Duration Distribution",
            self.throughput_plot,
            "Interactive histogram of case throughput times (hours).",
        )
        charts_row.addWidget(throughput_card, stretch=1)

        self.events_plot = pg.PlotWidget(axisItems={"bottom": DateAxisItem()})
        self._configure_plot_widget(self.events_plot)
        events_card = self._create_chart_card(
            "Events Over Time",
            self.events_plot,
            "Trend of recorded events by calendar date.",
        )
        charts_row.addWidget(events_card, stretch=1)

        layout.addLayout(charts_row)

        self.variants_plot = pg.PlotWidget()
        self._configure_plot_widget(self.variants_plot)
        layout.addWidget(
            self._create_chart_card(
                "Variant Coverage",
                self.variants_plot,
                "Top variants ranked by percentage of completed cases.",
            )
        )

        insights_row = QtWidgets.QHBoxLayout()
        insights_row.setSpacing(16)

        self.resource_plot = pg.PlotWidget()
        self._configure_plot_widget(self.resource_plot)
        insights_row.addWidget(
            self._create_chart_card(
                "Top Resources",
                self.resource_plot,
                "Most active resources by event volume; bubble size reflects unique cases handled.",
            ),
            stretch=1,
        )

        self.rework_table = QtWidgets.QTableView()
        self.rework_model = PandasTableModel()
        self.rework_table.setModel(self.rework_model)
        self.rework_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.rework_table.verticalHeader().setVisible(False)
        self.rework_table.setAlternatingRowColors(True)
        insights_row.addWidget(
            self._create_chart_card(
                "Activities With Rework",
                self.rework_table,
                "Activities executed multiple times within the same case.",
            ),
            stretch=1,
        )

        layout.addLayout(insights_row)

        self.variants_table = QtWidgets.QTableView()
        self.variants_model = PandasTableModel()
        self.variants_table.setModel(self.variants_model)
        self.variants_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.variants_table.verticalHeader().setVisible(False)
        self.variants_table.setAlternatingRowColors(True)
        self.variants_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(
            self._create_chart_card(
                "Variant Details",
                self.variants_table,
                "Use the table to inspect case counts and coverage by variant.",
            )
        )

        self.tabs.addTab(widget, "Behaviour")

    def _build_conformance_tab(self) -> None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        cards_layout = QtWidgets.QHBoxLayout()
        cards_layout.setSpacing(16)
        self.fitness_cards = {
            "avg_fitness": StatsCard("Average fitness", accent=self._color_palette[0]),
            "min_fitness": StatsCard("Min fitness", accent=self._color_palette[3]),
            "max_fitness": StatsCard("Max fitness", accent=self._color_palette[2]),
        }
        for card in self.fitness_cards.values():
            cards_layout.addWidget(card, stretch=1)
        layout.addLayout(cards_layout)

        self.conformance_table = QtWidgets.QTableView()
        self.conformance_model = PandasTableModel()
        self.conformance_table.setModel(self.conformance_model)
        self.conformance_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.conformance_table.verticalHeader().setVisible(False)
        self.conformance_table.setAlternatingRowColors(True)
        self.conformance_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(
            self._create_chart_card(
                "Alignment Details",
                self.conformance_table,
                "Drill into per-case fitness, alignment cost, and detected log moves.",
            )
        )

        self.tabs.addTab(widget, "Conformance")

    # Loading --------------------------------------------------------------
    def load_sample_log(self) -> None:
        if not SAMPLE_LOG_PATH.exists():
            self.logger.error("Sample log missing at %s", SAMPLE_LOG_PATH)
            self._show_error("Sample log not found. Ensure 'data/sample_order_to_cash.csv' exists.")
            return
        self.logger.info("Loading bundled sample log from %s", SAMPLE_LOG_PATH)
        try:
            container = load_log_from_csv(
                SAMPLE_LOG_PATH.read_bytes(),
                case_id_col="case_id",
                activity_col="activity",
                timestamp_col="timestamp",
                resource_col="resource",
            )
            self._set_log_container(container)
            self.statusBar().showMessage("Loaded sample log.", 5000)
            self.logger.info(
                "Sample log loaded successfully (%d events, %d cases).",
                len(container.df),
                container.df['case_id'].nunique(),
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Failed to load sample log.")
            self._show_error(f"Failed to load sample log: {exc}")

    def open_csv(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open CSV Log", "", "CSV files (*.csv)")
        if not file_path:
            return
        self.logger.info("Selected CSV file: %s", file_path)

        try:
            bytes_data = pathlib.Path(file_path).read_bytes()
            preview = read_csv_summary(bytes_data)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Unable to read CSV preview for %s", file_path)
            self._show_error(f"Unable to read CSV preview: {exc}")
            return

        dialog = ColumnMappingDialog(list(preview.columns), preview, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        mapping = dialog.mapping()
        try:
            container = load_log_from_csv(
                bytes_data,
                case_id_col=mapping["case"],
                activity_col=mapping["activity"],
                timestamp_col=mapping["timestamp"],
                resource_col=mapping["resource"],
                lifecycle_col=mapping["lifecycle"],
            )
            self._set_log_container(container)
            self.statusBar().showMessage(f"Loaded CSV log: {file_path}", 5000)
            self.logger.info(
                "CSV log loaded from %s (%d events, %d cases).",
                file_path,
                len(container.df),
                container.df['case_id'].nunique(),
            )
        except LogFormatError as exc:
            self.logger.warning("CSV column mapping error for %s: %s", file_path, exc)
            self._show_error(str(exc))
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Failed to load CSV file %s", file_path)
            self._show_error(f"Failed to load CSV: {exc}")

    def open_xes(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open XES Log", "", "XES files (*.xes)")
        if not file_path:
            return
        self.logger.info("Selected XES file: %s", file_path)
        try:
            bytes_data = pathlib.Path(file_path).read_bytes()
            container = load_log_from_xes(bytes_data)
            self._set_log_container(container)
            self.statusBar().showMessage(f"Loaded XES log: {file_path}", 5000)
            self.logger.info(
                "XES log loaded from %s (%d events, %d cases).",
                file_path,
                len(container.df),
                container.df['case_id'].nunique(),
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Failed to load XES file %s", file_path)
            self._show_error(f"Failed to load XES: {exc}")

    def _set_log_container(self, container: EventLogContainer) -> None:
        self.log_container = container
        self.filtered_log = container
        self._artifacts = None
        self.logger.info(
            "Log context updated (%d events, %d cases, %d activities).",
            len(container.df),
            container.df["case_id"].nunique(),
            container.df["activity"].nunique(),
        )
        self._populate_filters()
        self.refresh_views()

    # Filters --------------------------------------------------------------
    def _populate_filters(self) -> None:
        if not self.log_container:
            return
        self.case_list.blockSignals(True)
        self.activity_list.blockSignals(True)
        self.attribute_combo.blockSignals(True)

        self.case_list.clear()
        for case_id in sorted(self.log_container.case_ids):
            item = QtWidgets.QListWidgetItem(str(case_id))
            item.setSelected(True)
            self.case_list.addItem(item)

        self.activity_list.clear()
        for activity in sorted(self.log_container.activities):
            item = QtWidgets.QListWidgetItem(activity)
            item.setSelected(True)
            self.activity_list.addItem(item)

        df = self.log_container.df
        if not df.empty:
            min_ts = df["timestamp"].min().to_pydatetime()
            max_ts = df["timestamp"].max().to_pydatetime()
            start_qdate = QtCore.QDate(min_ts.year, min_ts.month, min_ts.day)
            end_qdate = QtCore.QDate(max_ts.year, max_ts.month, max_ts.day)
            self.start_date.setDate(start_qdate)
            self.end_date.setDate(end_qdate)
        else:
            today = QtCore.QDate.currentDate()
            self.start_date.setDate(today)
            self.end_date.setDate(today)

        attribute_columns = [col for col in df.columns if col not in {"case_id", "activity", "timestamp"}]
        self.attribute_combo.clear()
        self.attribute_combo.addItem("None")
        for column in attribute_columns:
            self.attribute_combo.addItem(column)
        self.attribute_values.clear()

        self.case_list.blockSignals(False)
        self.activity_list.blockSignals(False)
        self.attribute_combo.blockSignals(False)

    def _on_attribute_changed(self, index: int) -> None:
        if not self.log_container:
            return
        self.attribute_values.clear()
        attribute = self.attribute_combo.currentText()
        if index <= 0 or attribute == "None":
            return
        values = sorted(self.log_container.df[attribute].dropna().astype(str).unique())
        for value in values:
            item = QtWidgets.QListWidgetItem(value)
            item.setSelected(True)
            self.attribute_values.addItem(item)

    def apply_filters(self) -> None:
        if not self.log_container:
            return
        subset = self.log_container

        selected_cases = [item.text() for item in self.case_list.selectedItems()]
        if selected_cases and len(selected_cases) < self.case_list.count():
            subset = filter_by_case_ids(subset, selected_cases)

        selected_activities = [item.text() for item in self.activity_list.selectedItems()]
        if selected_activities and len(selected_activities) < self.activity_list.count():
            subset = filter_by_activity(subset, selected_activities)

        start_qdate = self.start_date.date()
        end_qdate = self.end_date.date()
        start_dt = datetime.combine(start_qdate.toPyDate(), datetime.min.time())
        end_dt = datetime.combine(end_qdate.toPyDate(), datetime.max.time())
        subset = filter_by_time_range(subset, start=start_dt, end=end_dt)

        attribute = self.attribute_combo.currentText()
        if attribute and attribute != "None" and self.attribute_values.count() > 0:
            selected_values = [item.text() for item in self.attribute_values.selectedItems()]
            if selected_values and len(selected_values) < self.attribute_values.count():
                subset = filter_by_attribute(subset, attribute, selected_values)

        if subset.df.empty:
            self.logger.warning("Filters result in empty dataset. Cases=%s Activities=%s Attribute=%s",
                                selected_cases, selected_activities, attribute)
            self._show_warning("Filtered log is empty. Adjust filters to continue.")
            return

        self.filtered_log = subset
        self._artifacts = None
        self.refresh_views()
        self.statusBar().showMessage("Filters applied.", 3000)
        self.logger.info(
            "Filters applied. Remaining %d events across %d cases.",
            len(subset.df),
            subset.df["case_id"].nunique(),
        )

    # Visual updates ------------------------------------------------------
    def refresh_views(self) -> None:
        if not self.filtered_log:
            return
        self.update_overview()
        self.update_process_model()
        self.update_variant_explorer()
        self.update_behaviour_views()
        self.update_conformance_view()

    def update_overview(self) -> None:
        if not self.filtered_log:
            return
        overview = process_analysis.compute_overview(self.filtered_log)
        for key, card in self.overview_cards.items():
            value = overview.get(key, "—")
            if isinstance(value, (int, float)):
                value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
            card.set_value(value)

        insights = process_analysis.compute_overview_insights(self.filtered_log)
        lines = []
        if insights.get("time_span"):
            lines.append(f"<b>Log span:</b> {insights['time_span']}")
        top_variant = insights.get("top_variant")
        if top_variant:
            lines.append(
                f"<b>Most common path:</b> {top_variant[0]} ({top_variant[1]:.1f}% of cases)"
            )
        top_resource = insights.get("top_resource")
        if top_resource:
            lines.append(
                f"<b>Most active resource:</b> {top_resource[0]} ({top_resource[1]} events across {top_resource[2]} cases)"
            )
        top_rework = insights.get("top_rework")
        if top_rework:
            lines.append(
                f"<b>Rework hot-spot:</b> {top_rework[0]} ({top_rework[1]:.1f}% of cases repeat it)"
            )
        avg_hours = insights.get("avg_duration_hours")
        if avg_hours is not None:
            lines.append(f"<b>Average case duration:</b> {avg_hours:.1f}h")
        self.overview_insights_label.setText(
            "<ul>" + "".join(f"<li>{line}</li>" for line in lines) + "</ul>"
            if lines
            else "<i>No additional insights for the current slice.</i>"
        )

    def update_process_model(self) -> None:
        if not self.filtered_log:
            self.process_graph_widget.clear()
            self.process_svg.load(QtCore.QByteArray())
            if hasattr(self, "part_flow_widget"):
                self.part_flow_widget.clear()
            if hasattr(self, "part_flow_caption"):
                self.part_flow_caption.setText("Load a log to explore part flow.")
            if hasattr(self, "bpmn_widget"):
                self.bpmn_widget.load(QtCore.QByteArray())
            if hasattr(self, "declare_model"):
                self.declare_model.set_dataframe(pd.DataFrame())
            if hasattr(self, "declare_caption"):
                self.declare_caption.setText("Load a log to derive constraints.")
            if hasattr(self, "markov_plot"):
                self._clear_markov_heatmap()
            return

        if hasattr(self, "part_flow_widget"):
            try:
                self._part_flow_data = process_analysis.compute_dfg_frequency_data(self.filtered_log, max_activities=16)
                self._refresh_part_flow_widget()
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.exception("Failed to compute part/item flow explorer data.")
                self._part_flow_data = None
                self.part_flow_widget.clear()
                self.part_flow_caption.setText("Part flow view unavailable – see logs for details.")
        try:
            self._artifacts = process_analysis.discover_process_model(self.filtered_log.event_log)
            self.logger.info("Process model discovered for current selection.")
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Process discovery failed.")
            self._show_error(f"Failed to discover process model: {exc}")
            self.process_graph_widget.clear()
            return

        self._update_process_visuals()

        try:
            bpmn_svg = process_analysis.render_bpmn_svg(self._artifacts)
            self.bpmn_widget.load(QtCore.QByteArray(bpmn_svg))
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Failed to generate BPMN rendering.")
            self.bpmn_widget.load(QtCore.QByteArray())
            self._show_warning(f"BPMN view unavailable: {exc}")

        try:
            declare_df = process_analysis.compute_declare_constraints(self.filtered_log)
            self.declare_model.set_dataframe(declare_df)
            if declare_df.empty:
                self.declare_caption.setText("No high-confidence obligations detected for the current filter slice.")
            else:
                self.declare_caption.setText(
                    "Confidence reflects the share of cases satisfying the rule; support is portion of the entire log."
                )
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Failed to derive declare constraints.")
            self.declare_model.set_dataframe(pd.DataFrame())
            self.declare_caption.setText("Constraints unavailable.")
            self._show_warning(f"Could not compute declare constraints: {exc}")

        try:
            markov_data = process_analysis.compute_transition_matrix(self.filtered_log)
            self._plot_markov_heatmap(markov_data)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Failed to compute transition probabilities.")
            self._clear_markov_heatmap()
            self._show_warning(f"Could not compute transition probabilities: {exc}")

    def _on_model_mode_changed(self, text: str) -> None:
        mode_map = {
            "frequency": "frequency",
            "performance": "performance",
            "minimal": "plain",
        }
        mode = mode_map.get(text.lower(), "frequency")
        if mode == self._model_metric_mode:
            return
        self._model_metric_mode = mode
        self._update_process_visuals()

    def _on_layout_mode_changed(self, text: str) -> None:
        layout_map = {
            "force-directed": "force",
            "circular": "circular",
            "kamada-kawai": "kamada",
            "hierarchical": "hierarchical",
        }
        layout_mode = layout_map.get(text.lower(), "force")
        self._layout_mode = layout_mode
        if self._artifacts:
            self.process_graph_widget.set_layout_mode(layout_mode)

    def _on_edge_scale_changed(self, value: int) -> None:
        self._edge_scale = max(0.5, value / 10.0)
        if self._artifacts:
            self.process_graph_widget.set_edge_scale(self._edge_scale)

    def _update_process_visuals(self) -> None:
        if not self._artifacts or not self.filtered_log:
            return

        mode = self._model_metric_mode
        if mode == "frequency":
            decorations = self._artifacts.frequency_decorations
        elif mode == "performance":
            decorations = self._artifacts.performance_decorations
        else:
            decorations = None

        svg_bytes = process_analysis.render_petri_net_svg(
            self._artifacts.net,
            self._artifacts.initial_marking,
            self._artifacts.final_marking,
            self.filtered_log.event_log,
            mode=mode,
            decorations=decorations,
        )
        self.process_svg.load(QtCore.QByteArray(svg_bytes))
        self.logger.info("Process model visual updated (%s mode).", mode)

        self.process_graph_widget.update_graph(
            self._artifacts,
            metric_mode=mode,
            layout_mode=self._layout_mode,
            edge_scale=self._edge_scale,
        )
        edge_count = len(getattr(self._artifacts.net, "arcs", []))
        self.logger.info(
            "Interactive Petri net refreshed (mode=%s, edges=%d).",
            mode,
            edge_count,
        )

        insights = process_analysis.compute_overview_insights(self.filtered_log)
        summary_lines = [
            f"<b>Events analysed:</b> {len(self.filtered_log.df):,} across {self.filtered_log.df['case_id'].nunique():,} cases."
        ]
        top_variant = insights.get("top_variant")
        if top_variant:
            summary_lines.append(
                f"<b>Typical journey:</b> {top_variant[0]} ({top_variant[1]:.1f}% of cases)."
            )
        top_rework = insights.get("top_rework")
        if top_rework:
            summary_lines.append(
                f"<b>Watch for rework:</b> {top_rework[0]} repeats in {top_rework[1]:.1f}% of cases."
            )
        if mode == "performance":
            summary_lines.append("<b>Colour legend:</b> teal edges are fast, coral edges mark slower hops.")
        elif mode == "frequency":
            summary_lines.append("<b>Colour legend:</b> thicker purple edges show busier paths.")
        else:
            summary_lines.append("<b>Minimal view:</b> switch to Frequency or Performance to see usage overlays.")
        self.process_summary_label.setText("<br>".join(summary_lines))

    def update_behaviour_views(self) -> None:
        if not self.filtered_log:
            return

        try:
            dfg_data = process_analysis.compute_dfg_frequency_data(self.filtered_log, max_activities=18)
            self._dfg_data_cache = dfg_data
            current_mode = "performance" if self.dfg_mode_buttons.checkedId() == 1 else "frequency"
            self.dfg_flow_widget.set_mode(current_mode)
            self.dfg_flow_widget.set_data(dfg_data)
            self._update_dfg_meta_label(dfg_data.get("metadata"), current_mode)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Failed to generate DFG visualisations.")
            self._show_warning(f"Unable to generate the process pulse view: {exc}")
            self.dfg_flow_widget.clear()
            self._dfg_data_cache = None
            self._update_dfg_meta_label(None, "frequency")
        else:
            self.logger.info("Interactive DFG visualisation refreshed.")

        try:
            throughput_df = process_analysis.compute_throughput_distribution(self.filtered_log)
            self._plot_throughput(throughput_df)

            events_df = process_analysis.compute_events_over_time(self.filtered_log)
            self._plot_events_over_time(events_df)

            variants_df = process_analysis.compute_variants_table(self.filtered_log)
            self.variants_model.set_dataframe(variants_df)
            self._plot_variants(variants_df)

            resources_df = process_analysis.compute_resource_summary(self.filtered_log)
            self._plot_resource_utilisation(resources_df)

            rework_df = process_analysis.compute_rework_table(self.filtered_log)
            self.rework_model.set_dataframe(rework_df)

            self.logger.info("Behaviour analytics refreshed (throughput, events, variants, resources, rework).")
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Failed to compute behavioural analytics.")
            self._show_warning(f"Could not compute behavioural analytics: {exc}")

    def update_conformance_view(self) -> None:
        if not self.filtered_log:
            return
        if not self._artifacts:
            try:
                self._artifacts = process_analysis.discover_process_model(self.filtered_log.event_log)
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.exception("Unable to discover model required for conformance analysis.")
                self._show_warning(f"Conformance analysis skipped: {exc}")
                return
        try:
            alignments_df, summary = process_analysis.align_log_against_model(self.filtered_log, self._artifacts)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Conformance analysis failed.")
            self._show_warning(f"Conformance analysis skipped: {exc}")
            return

        for key, card in self.fitness_cards.items():
            value = summary.get(key, "—")
            if isinstance(value, (int, float)):
                value = f"{value:.3f}"
            card.set_value(value)
        self.conformance_model.set_dataframe(alignments_df)
        self.logger.info(
            "Conformance metrics refreshed (avg=%s, min=%s, max=%s).",
            summary.get("avg_fitness"),
            summary.get("min_fitness"),
            summary.get("max_fitness"),
        )

    def update_variant_explorer(self) -> None:
        if not self.filtered_log:
            self.variant_model.set_dataframe(pd.DataFrame())
            self.variant_summary_label.setText("Load a log to see variants.")
            self.variant_flow_widget.clear()
            if hasattr(self, "variant_flow_meta"):
                self.variant_flow_meta.setText("")
            return

        try:
            dfg_data = process_analysis.compute_dfg_frequency_data(self.filtered_log)
            self.variant_flow_widget.set_data(dfg_data)
            self._update_flow_metadata(dfg_data.get("metadata"))
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Failed to compute flow map data.")
            self._show_warning(f"Could not build flow map: {exc}")
            self.variant_flow_widget.clear()
            if hasattr(self, "variant_flow_meta"):
                self.variant_flow_meta.setText(
                    "<small style='color:#FF7B7B;'>Flow view unavailable – see logs for details.</small>"
                )
        self._refresh_variant_table()

    def _on_variant_controls_changed(self, *_args) -> None:
        self._refresh_variant_table()

    def _refresh_variant_table(self) -> None:
        if not self.filtered_log:
            self.variant_model.set_dataframe(pd.DataFrame())
            return

        top_n = self.variant_top_spin.value()
        try:
            raw_variants_df = process_analysis.compute_variants_table(self.filtered_log, top_n=top_n)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Failed to compute variants table.")
            self._show_warning(f"Could not compute variants: {exc}")
            self.variant_model.set_dataframe(pd.DataFrame())
            self.variant_flow_widget.set_highlighted_sequence(None)
            self._happy_path_sequence = []
            self._happy_path_label = ""
            return

        if raw_variants_df.empty:
            self.variant_model.set_dataframe(pd.DataFrame())
            self.variant_summary_label.setText("<i>No variants match the current filters.</i>")
            self.variant_flow_widget.set_highlighted_sequence(None)
            self._happy_path_sequence = []
            self._happy_path_label = ""
            return

        self._happy_path_label = str(raw_variants_df.iloc[0]["variant"])
        self._happy_path_sequence = self._split_variant_label(self._happy_path_label)

        variants_df = raw_variants_df.copy()

        search = self.variant_search_input.text().strip()
        if search:
            mask = variants_df["variant"].str.contains(search, case=False, na=False)
            variants_df = variants_df[mask]

        self.variant_model.set_dataframe(variants_df)
        if variants_df.empty:
            self.variant_summary_label.setText("<i>No variants match the current filters.</i>")
            if self._happy_path_sequence:
                self.variant_flow_widget.set_highlighted_sequence(self._happy_path_sequence)
            return

        # Reset highlight before selecting to avoid flicker.
        self.variant_flow_widget.set_highlighted_sequence(None)

        selection_model = self.variant_table.selectionModel()
        if selection_model:
            selection_model.blockSignals(True)
            self.variant_table.selectRow(0)
            selection_model.blockSignals(False)
            if not self._variant_selection_connected:
                selection_model.selectionChanged.connect(self._on_variant_selection_changed)
                self._variant_selection_connected = True
        self._update_variant_detail(variants_df.iloc[0]["variant"])

    def _on_variant_selection_changed(self, selected: QtCore.QItemSelection, _deselected: QtCore.QItemSelection) -> None:
        if not selected.indexes():
            return
        row = selected.indexes()[0].row()
        if row < 0 or row >= len(self.variant_model._dataframe.index):
            return
        variant_label = self.variant_model._dataframe.iloc[row]["variant"]
        self._update_variant_detail(variant_label)

    def _update_variant_detail(self, variant_label: str) -> None:
        detail = process_analysis.compute_variant_detail(self.filtered_log, variant_label)
        cases = detail.get("cases", [])
        duration = detail.get("duration")

        lines = [f"<b>Variant:</b> {variant_label}"]
        coverage = detail.get("case_percentage")
        if coverage:
            lines.append(f"<b>Coverage:</b> {coverage:.1f}% of cases")
        if cases:
            lines.append(f"<b>Matching cases:</b> {', '.join(cases[:10])}{'…' if len(cases) > 10 else ''}")
        else:
            lines.append("<b>Matching cases:</b> none found in the filtered slice.")
        if duration:
            avg = duration.get("avg_hours")
            med = duration.get("median_hours")
            if avg is not None and med is not None:
                lines.append(
                    f"<b>Average duration:</b> {avg:.1f}h &nbsp; | &nbsp; <b>Median:</b> {med:.1f}h"
                )
        sequence = detail.get("sequence", [])
        if sequence:
            lines.append(f"<b>Path:</b> {' → '.join(sequence)}")
        else:
            sequence = []

        highlight_ok = self.variant_flow_widget.set_highlighted_sequence(sequence)

        if self._happy_path_sequence:
            comparison = self._describe_variant_difference(self._happy_path_sequence, sequence)
            if comparison:
                lines.append(f"<b>Vs. happy path:</b> {comparison}")
            elif sequence == self._happy_path_sequence:
                lines.append("<b>Vs. happy path:</b> matches the dominant flow.")

        if sequence and not highlight_ok:
            lines.append(
                "<b>Flow view:</b> low-frequency steps are hidden from the compact map, "
                "but still counted in the variant statistics."
            )

        self.variant_summary_label.setText("<br>".join(lines))

    @staticmethod
    def _split_variant_label(label: str) -> List[str]:
        return [step.strip() for step in label.split("→") if step and step.strip()]

    def _describe_variant_difference(self, baseline: List[str], candidate: List[str]) -> Optional[str]:
        if not baseline or not candidate:
            return None
        matcher = difflib.SequenceMatcher(None, baseline, candidate)
        narratives: list[str] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            baseline_segment = " → ".join(baseline[i1:i2]) or "∅"
            candidate_segment = " → ".join(candidate[j1:j2]) or "∅"
            if tag == "replace":
                narratives.append(f"replaces <i>{baseline_segment}</i> with <i>{candidate_segment}</i>")
            elif tag == "delete":
                narratives.append(f"skips <i>{baseline_segment}</i>")
            elif tag == "insert":
                narratives.append(f"adds <i>{candidate_segment}</i>")
            if len(narratives) >= 2:
                break
        if not narratives:
            return None
        return "; ".join(narratives)

    def _update_flow_metadata(self, metadata: Optional[Dict[str, Any]]) -> None:
        if not hasattr(self, "variant_flow_meta"):
            return
        if not metadata:
            self.variant_flow_meta.setText("")
            return

        parts: list[str] = []
        total_cases = metadata.get("total_cases")
        kept = metadata.get("kept_activities")
        total_activities = metadata.get("total_activities", kept)
        truncated = metadata.get("truncated", 0)
        max_edge = metadata.get("max_edge_weight")

        if total_cases:
            parts.append(f"{int(total_cases):,} cases visualised")
        if kept:
            if total_activities and total_activities != kept:
                parts.append(f"top {kept} of {total_activities} activities")
            else:
                parts.append(f"{kept} activities")
        if max_edge:
            parts.append(f"busiest hand-off {int(max_edge):,} cases")

        text = " • ".join(parts)
        if truncated:
            text += (
                " &nbsp; <span style='color:#FFAD17;'>"
                f"{int(truncated)} low-volume steps hidden for clarity</span>"
            )

        self.variant_flow_meta.setText(f"<small>{text}</small>")

    # UI helpers -----------------------------------------------------------

    def _refresh_part_flow_widget(self) -> None:
        if not hasattr(self, "part_flow_widget"):
            return
        if not self._part_flow_data:
            self.part_flow_widget.clear()
            if hasattr(self, "part_flow_caption"):
                self.part_flow_caption.setText("Load a log to explore part flow.")
            return
        mode_id = self.part_flow_mode_buttons.checkedId() if hasattr(self, "part_flow_mode_buttons") else 0
        mode = "performance" if mode_id == 1 else "frequency"
        self.part_flow_widget.set_mode(mode)
        self.part_flow_widget.set_data(self._part_flow_data)
        self._update_part_flow_caption(self._part_flow_data.get("metadata"), mode)

    def _update_part_flow_caption(self, metadata: Optional[Dict[str, Any]], mode: str) -> None:
        if not hasattr(self, "part_flow_caption"):
            return
        if not metadata:
            self.part_flow_caption.setText("Load a log to explore part flow.")
            return
        total_cases = metadata.get("total_cases")
        truncated = metadata.get("truncated", 0)
        if mode == "performance":
            base = (
                "Velocity mode: teal arcs are quicker, coral arcs spotlight slower transitions; grey arcs highlight rework loops."
            )
        else:
            base = "Volume mode: thicker links show busier transitions; grey arcs highlight repeat loops."
        parts = [base]
        if total_cases:
            parts.append(f"{int(total_cases):,} cases summarised")
        if truncated:
            parts.append(f"{int(truncated)} low-volume steps hidden for clarity")
        self.part_flow_caption.setText(" • ".join(parts))

    def _on_part_flow_mode_changed(self, button: QtWidgets.QAbstractButton) -> None:
        if not hasattr(self, "part_flow_widget"):
            return
        if button is None:
            return
        mode = "performance" if self.part_flow_mode_buttons.id(button) == 1 else "frequency"
        if not self._part_flow_data:
            self.part_flow_widget.set_mode(mode)
            return
        self.part_flow_widget.set_mode(mode)
        self.part_flow_widget.set_data(self._part_flow_data)
        self._update_part_flow_caption(self._part_flow_data.get("metadata"), mode)

    def _on_dfg_mode_changed(self, button: QtWidgets.QAbstractButton) -> None:
        if not button:
            return
        mode = "performance" if self.dfg_mode_buttons.id(button) == 1 else "frequency"
        self.dfg_flow_widget.set_mode(mode)
        if self._dfg_data_cache:
            self._update_dfg_meta_label(self._dfg_data_cache.get("metadata"), mode)

    def _update_dfg_meta_label(self, metadata: Optional[Dict[str, Any]], mode: str) -> None:
        if not hasattr(self, "dfg_meta_label"):
            return
        if not metadata:
            self.dfg_meta_label.setText("")
            return

        truncated = int(metadata.get("truncated") or 0)
        total_edges = metadata.get("total_edges")
        kept = metadata.get("kept_activities")

        if mode == "performance":
            base = (
                "Velocity view: teal edges are quick hand-offs, coral glows flag slower transitions."
            )
        else:
            base = (
                "Volume view: edge thickness and glow scale with case traffic; brighter paths show the main flow."
            )

        parts = [base]
        if total_edges:
            parts.append(f"{int(total_edges):,} transitions visualised across {kept} activities.")
        if truncated:
            parts.append(
                f"<span style='color:#FFAD17;'>Hiding {truncated} low-volume steps to keep the arcade view readable.</span>"
            )
        self.dfg_meta_label.setText("<br>".join(parts))

    def _plot_markov_heatmap(self, data: Dict[str, Any]) -> None:
        states = data.get("states") or []
        probabilities = data.get("probabilities") or []
        matrix = np.array(probabilities, dtype=float)
        if not states or matrix.size == 0:
            self._clear_markov_heatmap()
            return

        if self._markov_image_item:
            self.markov_plot.removeItem(self._markov_image_item)
            self._markov_image_item = None

        image_item = pg.ImageItem(matrix)
        cmap = pg.colormap.get("viridis")
        image_item.setLookupTable(cmap.getLookupTable(nPts=256))
        max_prob = float(matrix.max())
        image_item.setLevels((0.0, max(0.01, max_prob)))
        image_item.setPos(-0.5, -0.5)
        self.markov_plot.addItem(image_item)
        self._markov_image_item = image_item

        plot_item = self.markov_plot.getPlotItem()
        plot_item.invertY(True)
        plot_item.setRange(
            xRange=(-0.5, matrix.shape[1] - 0.5),
            yRange=(-0.5, matrix.shape[0] - 0.5),
            padding=0.05,
        )
        x_ticks = list(enumerate(states))
        y_ticks = list(enumerate(states))
        plot_item.getAxis("bottom").setTicks([x_ticks])
        plot_item.getAxis("left").setTicks([y_ticks])
        plot_item.getAxis("bottom").setStyle(autoExpandTextSpace=True, tickFont=QtGui.QFont("Segoe UI", 8))
        plot_item.getAxis("left").setStyle(autoExpandTextSpace=True, tickFont=QtGui.QFont("Segoe UI", 8))

        if max_prob > 0:
            self.markov_caption.setText(
                f"Peak transition probability: {max_prob:.2f}. Rows denote the current step, columns the next step."
            )
        else:
            self.markov_caption.setText("No significant transitions detected for the selected activities.")
        self._markov_states = states

    def _clear_markov_heatmap(self) -> None:
        plot_item = self.markov_plot.getPlotItem()
        if self._markov_image_item:
            plot_item.removeItem(self._markov_image_item)
            self._markov_image_item = None
        plot_item.clear()
        plot_item.setTitle(
            "<span style='color:#e3e7ff;font-size:13pt;'>Transition Probabilities</span>"
        )
        plot_item.getAxis("bottom").setTicks([])  # type: ignore[arg-type]
        plot_item.getAxis("left").setTicks([])  # type: ignore[arg-type]
        plot_item.invertY(True)
        self._markov_states = []
        self.markov_caption.setText("Load a log to project transition probabilities.")

    def _create_chart_card(self, title: str, content: QtWidgets.QWidget, subtitle: str = "") -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("ChartCard")
        frame_layout = QtWidgets.QVBoxLayout(frame)
        frame_layout.setContentsMargins(20, 20, 20, 20)
        frame_layout.setSpacing(10)

        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("ChartCardTitle")
        title_label.setStyleSheet("font-size: 16px; font-weight: 600; color: #f3f5ff;")
        frame_layout.addWidget(title_label)

        if subtitle:
            subtitle_label = QtWidgets.QLabel(subtitle)
            subtitle_label.setObjectName("ChartCardSubtitle")
            subtitle_label.setWordWrap(True)
            subtitle_label.setStyleSheet("color: #9aa5d9; font-size: 12px;")
            frame_layout.addWidget(subtitle_label)

        frame_layout.addWidget(content, stretch=1)

        frame.setStyleSheet(
            """
            QFrame#ChartCard {
                border-radius: 16px;
                border: 1px solid rgba(108, 131, 255, 0.18);
                background-color: rgba(20, 23, 35, 0.88);
            }
            """
        )
        return frame

    def _configure_plot_widget(self, plot: pg.PlotWidget) -> None:
        plot.setBackground("transparent")
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(x=True, y=False)
        item = plot.getPlotItem()
        item.showGrid(x=True, y=True, alpha=0.12)
        for axis_name in ("left", "bottom"):
            axis = item.getAxis(axis_name)
            axis.setPen(pg.mkPen(color="#282c40"))
            axis.setTextPen(pg.mkPen("#d1d7ff"))
        item.getViewBox().setBackgroundColor(QtGui.QColor(0, 0, 0, 0))
        item.getViewBox().setBorder(pg.mkPen(None))
        item.setDownsampling(mode="peak")

    def _color_for_index(self, index: int) -> str:
        return self._color_palette[index % len(self._color_palette)]

    def _reset_plot(self, plot: pg.PlotWidget, *, title: str, bottom: str, left: str) -> None:
        plot.clear()
        item = plot.getPlotItem()
        item.setTitle(f"<span style='color:#e3e7ff;font-size:13pt;'>{title}</span>")
        item.setLabel("bottom", bottom)
        item.setLabel("left", left)

    # Plot helpers --------------------------------------------------------
    def _plot_throughput(self, df: pd.DataFrame) -> None:
        title = "Case Duration Distribution"
        self._reset_plot(self.throughput_plot, title=title, bottom="Duration (hours)", left="Cases")

        if df.empty or df["duration_hours"].dropna().empty:
            self.throughput_plot.getPlotItem().setTitle(
                f"<span style='color:#e3e7ff;font-size:13pt;'>{title}</span>"
                "<br><span style='color:#8a93c9;font-size:10pt;'>No throughput data available.</span>"
            )
            return

        durations = df["duration_hours"].dropna().astype(float).values
        bins = min(24, max(8, int(np.sqrt(len(durations)))))
        hist, edges = np.histogram(durations, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        width = float(edges[1] - edges[0]) * 0.9 if len(edges) > 1 else 0.5
        color = self._color_for_index(0)

        bar = pg.BarGraphItem(x=centers, height=hist, width=width, brush=pg.mkBrush(color), pen=pg.mkPen("#1a1f33", width=1))
        self.throughput_plot.addItem(bar)
        curve = pg.PlotCurveItem(centers, hist, pen=pg.mkPen("#ffffff", width=2, style=QtCore.Qt.PenStyle.SolidLine))
        self.throughput_plot.addItem(curve)
        self.throughput_plot.setYRange(0, hist.max() * 1.2 if hist.max() > 0 else 1)

    def _plot_events_over_time(self, df: pd.DataFrame) -> None:
        title = "Events Over Time"
        self._reset_plot(self.events_plot, title=title, bottom="Date", left="Events")

        if df.empty:
            self.events_plot.getPlotItem().setTitle(
                f"<span style='color:#e3e7ff;font-size:13pt;'>{title}</span>"
                "<br><span style='color:#8a93c9;font-size:10pt;'>No events recorded for the selected slice.</span>"
            )
            return

        dates = pd.to_datetime(df["Date"])
        timestamps = dates.astype("int64") / 1e9
        counts = df["Events"].astype(float).values
        color = self._color_for_index(1)
        pen = pg.mkPen(color, width=3)
        self.events_plot.plot(
            timestamps,
            counts,
            pen=pen,
            symbol="o",
            symbolPen=pg.mkPen(color, width=1.5),
            symbolBrush=pg.mkBrush("#f9fafc"),
            symbolSize=9,
        )
        self.events_plot.getViewBox().setAutoVisible(y=True)

    def _plot_variants(self, df: pd.DataFrame) -> None:
        title = "Variant Coverage"
        self._reset_plot(self.variants_plot, title=title, bottom="Variant (top 8)", left="Cases (%)")

        if df.empty:
            self.variants_plot.getPlotItem().setTitle(
                f"<span style='color:#e3e7ff;font-size:13pt;'>{title}</span>"
                "<br><span style='color:#8a93c9;font-size:10pt;'>No variants computed for this slice.</span>"
            )
            return

        top_variants = df.head(8).reset_index(drop=True)
        for idx, row in top_variants.iterrows():
            height = float(row["case_percentage"])
            bar = pg.BarGraphItem(
                x=[float(idx)],
                height=[height],
                width=0.6,
                brush=pg.mkBrush(self._color_for_index(idx)),
                pen=pg.mkPen("#1a1f33", width=1),
            )
            self.variants_plot.addItem(bar)
        heights = top_variants["case_percentage"].astype(float).values
        ticks = [(idx, top_variants.loc[idx, "variant"]) for idx in range(len(top_variants))]
        axis = self.variants_plot.getPlotItem().getAxis("bottom")
        axis.setTicks([ticks])
        axis.setStyle(autoExpandTextSpace=True)
        self.variants_plot.setYRange(0, max(heights) * 1.2 if len(heights) else 1)
        self.variants_plot.getViewBox().setLimits(xMin=-1, xMax=len(top_variants))

    def _plot_resource_utilisation(self, df: pd.DataFrame) -> None:
        self._reset_plot(self.resource_plot, title="Resource Workload", bottom="Resource", left="Events")

        if df.empty:
            self.resource_plot.getPlotItem().setTitle(
                "<span style='color:#e3e7ff;font-size:13pt;'>Resource Workload</span>"
                "<br><span style='color:#8a93c9;font-size:10pt;'>No resource data available.</span>"
            )
            return

        x = np.arange(len(df))
        events = df["events"].astype(float).values
        brushes = [pg.mkBrush(self._color_for_index(idx)) for idx in range(len(df))]
        bars = pg.BarGraphItem(x=x, height=events, width=0.6, brushes=brushes, pen=pg.mkPen("#1a1f33", width=1))
        self.resource_plot.addItem(bars)

        scatter = pg.ScatterPlotItem(x=x, y=events, size=10 + df["cases"].astype(float).values, brush=pg.mkBrush("#ffffff"))
        self.resource_plot.addItem(scatter)

        axis = self.resource_plot.getPlotItem().getAxis("bottom")
        axis.setTicks([[(idx, df.loc[idx, "resource"]) for idx in range(len(df))]])
        self.resource_plot.setYRange(0, max(events) * 1.2 if len(events) else 1)
        self.resource_plot.getViewBox().setLimits(xMin=-1, xMax=len(df))

    # Exports --------------------------------------------------------------
    def export_pnml(self) -> None:
        if not self.filtered_log or not self._artifacts:
            self.logger.warning("PNML export requested without a discovered model.")
            self._show_warning("Load a process model before exporting PNML.")
            return
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Petri Net", "", "PNML files (*.pnml)")
        if not file_path:
            return
        try:
            pnml_bytes = process_analysis.export_petri_net_pnml(self._artifacts)
            pathlib.Path(file_path).write_bytes(pnml_bytes)
            self.statusBar().showMessage(f"Saved PNML to {file_path}", 5000)
            self.logger.info("PNML exported to %s", file_path)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Failed to export PNML to %s", file_path)
            self._show_error(f"Failed to export PNML: {exc}")

    def export_xes(self) -> None:
        if not self.filtered_log:
            self.logger.warning("XES export requested with no filtered log.")
            self._show_warning("No filtered log available to export.")
            return
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save XES Log", "", "XES files (*.xes)")
        if not file_path:
            return
        try:
            xes_bytes = self.filtered_log.to_xes_bytes()
            pathlib.Path(file_path).write_bytes(xes_bytes)
            self.statusBar().showMessage(f"Saved XES to {file_path}", 5000)
            self.logger.info("Filtered log exported to %s", file_path)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Failed to export XES to %s", file_path)
            self._show_error(f"Failed to export XES: {exc}")

    # Messaging ------------------------------------------------------------
    def _show_error(self, message: str) -> None:
        self.logger.error(message)
        QtWidgets.QMessageBox.critical(self, "Error", message)

    def _show_warning(self, message: str) -> None:
        self.logger.warning(message)
        QtWidgets.QMessageBox.warning(self, "Warning", message)


def main() -> None:
    logger = BASE_LOGGER.getChild("runtime")
    logger.info("Starting QApplication event loop.")
    app = QtWidgets.QApplication(sys.argv)
    window = ProcessMiningApp()
    window.show()
    exit_code = app.exec()
    logger.info("Application closed with exit code %s", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
