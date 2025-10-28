from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional, Sequence
import pandas as pd

from .log_loader import EventLogContainer


def filter_by_case_ids(log_container: EventLogContainer, case_ids: Sequence[str]) -> EventLogContainer:
    subset = log_container.df[log_container.df["case_id"].isin(case_ids)].copy()
    return EventLogContainer.from_dataframe(subset)


def filter_by_activity(log_container: EventLogContainer, activities: Sequence[str]) -> EventLogContainer:
    subset = log_container.df[log_container.df["activity"].isin(activities)].copy()
    return EventLogContainer.from_dataframe(subset)


def filter_by_time_range(
    log_container: EventLogContainer, start: Optional[datetime] = None, end: Optional[datetime] = None
) -> EventLogContainer:
    subset = log_container.df
    if start:
        subset = subset[subset["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        subset = subset[subset["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    subset = subset.copy()
    return EventLogContainer.from_dataframe(subset)


def filter_by_attribute(
    log_container: EventLogContainer, attribute_name: str, allowed_values: Iterable[str]
) -> EventLogContainer:
    if attribute_name not in log_container.df.columns:
        return log_container
    subset = log_container.df[log_container.df[attribute_name].astype(str).isin(set(map(str, allowed_values)))].copy()
    return EventLogContainer.from_dataframe(subset)
