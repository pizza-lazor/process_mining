from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd
from pandas.errors import ParserError
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.util import sorting
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer


class LogFormatError(Exception):
    """Raised when an event log cannot be parsed or converted."""


@dataclass
class EventLogContainer:
    """
    Wrapper that keeps the raw dataframe and the pm4py event log in sync.

    The dataframe uses `case_id`, `activity`, and `timestamp` as canonical columns.
    """

    df: pd.DataFrame
    event_log: EventLog

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "EventLogContainer":
        df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
        event_log = _dataframe_to_event_log(df)
        return cls(df=df, event_log=event_log)

    @property
    def case_ids(self) -> Iterable[str]:
        return self.df["case_id"].unique()

    @property
    def activities(self) -> Iterable[str]:
        return self.df["activity"].unique()

    def to_xes_bytes(self) -> bytes:
        """Export the current log snapshot to an XES byte string."""
        xes_string = xes_exporter.serialize(self.event_log)
        return xes_string.encode("utf-8")


def _normalise_dataframe(
    df: pd.DataFrame,
    case_id_col: str,
    activity_col: str,
    timestamp_col: str,
    resource_col: Optional[str] = None,
    lifecycle_col: Optional[str] = None,
    variant_separator: Optional[str] = None,
) -> pd.DataFrame:
    """
    Standardise column names and types so that pm4py can convert the log reliably.
    """
    missing = {case_id_col, activity_col, timestamp_col} - set(df.columns)
    if missing:
        raise LogFormatError(f"Missing required columns: {', '.join(sorted(missing))}")

    renamed = {
        case_id_col: "case_id",
        activity_col: "activity",
        timestamp_col: "timestamp",
    }

    if resource_col and resource_col in df.columns:
        renamed[resource_col] = "resource"
    if lifecycle_col and lifecycle_col in df.columns:
        renamed[lifecycle_col] = "lifecycle"

    normalised = df.rename(columns=renamed).copy()
    normalised["case_id"] = normalised["case_id"].astype(str)
    normalised["activity"] = normalised["activity"].astype(str)
    normalised["timestamp"] = pd.to_datetime(normalised["timestamp"], utc=True)

    # Duplicate canonical columns into pm4py default XES keys so downstream conversions work.
    normalised["case:concept:name"] = normalised["case_id"]
    normalised["concept:name"] = normalised["activity"]
    normalised["time:timestamp"] = normalised["timestamp"]

    if "resource" in normalised.columns:
        normalised["org:resource"] = normalised["resource"]
    if "cost" in normalised.columns:
        normalised["cost:total"] = normalised["cost"]
    if "lifecycle" in normalised.columns:
        normalised["lifecycle:transition"] = normalised["lifecycle"]

    if variant_separator:
        normalised["variant_key"] = (
            normalised.groupby("case_id")["activity"]
            .transform(lambda acts: variant_separator.join(acts))
            .astype(str)
        )

    normalised = normalised.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    return normalised


def _dataframe_to_event_log(df: pd.DataFrame) -> EventLog:
    """
    Convert the canonical dataframe into a pm4py `EventLog`.
    """
    parameters = {
        log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name",
        log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ATTRIBUTE_PREFIX: "case:",
    }

    event_log = log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)
    sorting.sort_timestamp(event_log)
    return event_log


def load_log_from_xes(file_bytes: bytes) -> EventLogContainer:
    """
    Load an XES file into both pm4py event log and pandas dataframe representations.
    """
    xes_string = file_bytes.decode("utf-8")
    event_log = xes_importer.deserialize(xes_string)
    df = log_converter.apply(event_log, variant=log_converter.Variants.TO_DATA_FRAME)
    df = _normalise_dataframe(df, "case:concept:name", "concept:name", "time:timestamp")
    event_log = _dataframe_to_event_log(df)
    return EventLogContainer(df=df, event_log=event_log)


def load_log_from_csv(
    file_bytes: bytes,
    case_id_col: str,
    activity_col: str,
    timestamp_col: str,
    resource_col: Optional[str] = None,
    lifecycle_col: Optional[str] = None,
) -> EventLogContainer:
    """
    Load a CSV file that contains an event log in tabular form.
    """
    buffer = io.BytesIO(file_bytes)
    try:
        df = pd.read_csv(buffer)
    except ParserError as exc:
        buffer.seek(0)
        try:
            df = pd.read_csv(buffer, sep=None, engine="python")
        except ParserError as exc_second:
            raise LogFormatError(
                "Unable to parse CSV content. Ensure the file uses a consistent delimiter (e.g., comma or semicolon) "
                "and that embedded commas are quoted."
            ) from exc_second
    df = _normalise_dataframe(
        df,
        case_id_col=case_id_col,
        activity_col=activity_col,
        timestamp_col=timestamp_col,
        resource_col=resource_col,
        lifecycle_col=lifecycle_col,
    )
    event_log = _dataframe_to_event_log(df)
    return EventLogContainer(df=df, event_log=event_log)


def try_auto_detect_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    """
    Best-effort detection of case/activity/timestamp columns for CSV uploads.
    """
    lowered = {col.lower(): col for col in df.columns}

    def pick(options: Iterable[str]) -> Optional[str]:
        for option in options:
            if option in lowered:
                return lowered[option]
        return None

    candidates_case = ("case:concept:name", "case_id", "case", "caseid", "case concept name")
    candidates_act = ("concept:name", "activity", "event", "task")
    candidates_ts = ("time:timestamp", "timestamp", "time", "datetime")

    case_id_col = pick(candidates_case)
    activity_col = pick(candidates_act)
    timestamp_col = pick(candidates_ts)

    if not all([case_id_col, activity_col, timestamp_col]):
        raise LogFormatError("Could not auto-detect case/activity/timestamp columns.")

    return case_id_col, activity_col, timestamp_col


def read_csv_summary(file_bytes: bytes) -> pd.DataFrame:
    """
    Read CSV content into a dataframe without normalisation.
    Useful for previewing column names to let the user map fields.
    """
    buffer = io.BytesIO(file_bytes)
    try:
        return pd.read_csv(buffer, nrows=50)
    except ParserError:
        buffer.seek(0)
        try:
            return pd.read_csv(buffer, nrows=50, sep=None, engine="python", on_bad_lines="skip")
        except ParserError as exc_second:
            raise LogFormatError(
                "Unable to parse CSV preview. Check for mixed delimiters or irregular quoting."
            ) from exc_second
