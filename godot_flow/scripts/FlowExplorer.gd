extends Node2D

@export_file("*.json") var payload_path: String = "res://data/sample_flow_payload.json"
@export_range(80.0, 400.0, 10.0) var base_radius: float = 240.0

var _node_positions: Dictionary[String, Vector2] = {}
var _edges: Array[Dictionary] = []
var _metadata: Dictionary = {}
var _max_frequency: float = 1.0

func _ready() -> void:
	_load_payload()

func _load_payload() -> void:
	if payload_path.is_empty():
		push_warning("No payload path provided.")
		return
	if not FileAccess.file_exists(payload_path):
		push_warning("Payload file not found: %s" % payload_path)
		return

	var file := FileAccess.open(payload_path, FileAccess.READ)
	if file == null:
		push_error("Unable to open payload: %s" % payload_path)
		return

	var text := file.get_as_text()
	file.close()
	var parsed: Variant = JSON.parse_string(text)
	if typeof(parsed) != TYPE_DICTIONARY:
		push_error("Invalid JSON payload structure.")
		return

	var parsed_dict: Dictionary = parsed as Dictionary

	var metadata_variant: Variant = parsed_dict.get("metadata", {})
	if metadata_variant is Dictionary:
		var metadata_dict: Dictionary = metadata_variant as Dictionary
		_metadata = metadata_dict
	else:
		_metadata = {}

	var edges_typed: Array[Dictionary] = []
	var raw_edges_variant: Variant = parsed_dict.get("edges", [])
	if raw_edges_variant is Array:
		var raw_edges: Array = raw_edges_variant as Array
		for edge_variant in raw_edges:
			if edge_variant is Dictionary:
				edges_typed.append(edge_variant as Dictionary)
	_edges = edges_typed

	var nodes_variant: Variant = parsed_dict.get("nodes", [])
	if nodes_variant is Array:
		var nodes_array: Array = nodes_variant as Array
		_layout_nodes(nodes_array)
	else:
		_layout_nodes([])
	queue_redraw()

func _layout_nodes(nodes: Array) -> void:
	for child in get_children():
		if child.is_in_group("flow_node"):
			child.queue_free()

	_node_positions.clear()
	_max_frequency = 1.0
	if nodes.is_empty():
		return

	nodes.sort_custom(Callable(self, "_sort_nodes"))
	var first_node: Dictionary = nodes[0] as Dictionary
	if first_node:
		_max_frequency = max(1.0, float(first_node.get("frequency", 1)))

	var total: int = max(1, nodes.size())
	var ring_radius := base_radius
	for idx in range(total):
		var node_dict: Dictionary = nodes[idx] as Dictionary
		if node_dict == null:
			continue
		var angle := TAU * float(idx) / float(total)
		var node_pos := Vector2(ring_radius * cos(angle), ring_radius * sin(angle))
		var node_id := str(node_dict.get("id", str(idx)))
		_node_positions[node_id] = node_pos
		_spawn_node_marker(node_dict, node_pos)

func _spawn_node_marker(node: Dictionary, node_pos: Vector2) -> void:
	var marker := Node2D.new()
	marker.position = node_pos
	marker.add_to_group("flow_node")
	add_child(marker)

	var label := Label.new()
	label.text = "%s\n%s cases" % [node.get("label", node.get("id", "")), node.get("frequency", 0)]
	label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	label.position = Vector2(-70, -14)
	label.size = Vector2(140, 28)
	label.add_theme_color_override("font_color", Color(0.87, 0.91, 1.0, 1.0))
	marker.add_child(label)

func _draw() -> void:
	_draw_edges()
	_draw_nodes()

func _draw_edges() -> void:
	for edge_index: int in range(_edges.size()):
		var edge_variant: Variant = _edges[edge_index]
		if not (edge_variant is Dictionary):
			continue
		var edge: Dictionary = edge_variant as Dictionary
		if edge == null:
			continue
		var src_id: String = str(edge.get("source", ""))
		var dst_id: String = str(edge.get("target", ""))
		if not _node_positions.has(src_id) or not _node_positions.has(dst_id):
			continue
		var src: Vector2 = _node_positions[src_id]
		var dst: Vector2 = _node_positions[dst_id]
		var is_rework: bool = bool(edge.get("is_rework", false))
		var volume: float = float(edge.get("frequency", 0))
		var weight: float = lerp(1.5, 6.0, clamp(volume / _max_frequency, 0.0, 1.0))
		var color: Color = Color(0.35, 0.72, 0.98)
		if is_rework:
			color = Color(0.53, 0.55, 0.65, 0.85)
		elif edge.get("duration_seconds", 0.0) > 0.0 and _metadata.has("max_edge_duration"):
			var max_duration: float = max(0.001, float(_metadata.get("max_edge_duration", 0.0)))
			var duration_seconds: float = float(edge.get("duration_seconds", 0.0))
			var ratio: float = clamp(duration_seconds / max_duration, 0.0, 1.0)
			color = Color(lerp(0.27, 0.95, ratio), lerp(0.88, 0.33, ratio), lerp(0.94, 0.18, ratio))
		draw_line(src, dst, color, weight, true)

func _draw_nodes() -> void:
	for key: String in _node_positions.keys():
		var id: String = key
		var pos: Vector2 = _node_positions[id]
		var fill := Color(0.32, 0.47, 0.92, 0.95)
		if id == "__start__":
			fill = Color(0.38, 0.93, 0.69, 0.95)
		elif id == "__end__":
			fill = Color(0.94, 0.37, 0.36, 0.95)
		draw_circle(pos, 22.0, fill)
		draw_arc(pos, 22.0, 0.0, TAU, 48, Color(0.06, 0.08, 0.14), 2.2)

func _sort_nodes(a: Dictionary, b: Dictionary) -> bool:
	return int(a.get("frequency", 0)) > int(b.get("frequency", 0))
