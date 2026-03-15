"""Tests for dashboard configuration data structures."""

from __future__ import annotations

import pytest

from tracking.src.dashboards import (
    DashboardConfig,
    PanelDef,
    get_collapse_dashboard,
    get_overview_dashboard,
    get_safety_dashboard,
)


class TestPanelDef:
    def test_defaults(self) -> None:
        p = PanelDef(title="Test", metric_keys=["m/key"])
        assert p.panel_type == "line"
        assert p.section == "default"

    def test_custom_values(self) -> None:
        p = PanelDef(title="Bar", metric_keys=["a", "b"], panel_type="bar", section="sec")
        assert p.title == "Bar"
        assert p.metric_keys == ["a", "b"]
        assert p.panel_type == "bar"
        assert p.section == "sec"


class TestDashboardConfig:
    def test_to_dict_basic(self) -> None:
        cfg = DashboardConfig(name="test", description="desc")
        d = cfg.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "desc"
        assert d["panels"] == []

    def test_to_dict_with_panels(self) -> None:
        cfg = DashboardConfig(
            name="test",
            panels=[
                PanelDef(title="P1", metric_keys=["k1"]),
                PanelDef(title="P2", metric_keys=["k2", "k3"], panel_type="bar"),
            ],
        )
        d = cfg.to_dict()
        assert len(d["panels"]) == 2
        assert d["panels"][0]["title"] == "P1"
        assert d["panels"][0]["metric_keys"] == ["k1"]
        assert d["panels"][1]["panel_type"] == "bar"


class TestGetSafetyDashboard:
    def test_returns_dict(self) -> None:
        result = get_safety_dashboard()
        assert isinstance(result, dict)

    def test_has_name_and_panels(self) -> None:
        result = get_safety_dashboard()
        assert "name" in result
        assert "panels" in result
        assert result["name"] == "Safety Dashboard"

    def test_has_description(self) -> None:
        result = get_safety_dashboard()
        assert "description" in result
        assert len(result["description"]) > 0

    def test_panels_are_list(self) -> None:
        result = get_safety_dashboard()
        assert isinstance(result["panels"], list)
        assert len(result["panels"]) > 0

    def test_panel_structure(self) -> None:
        result = get_safety_dashboard()
        for panel in result["panels"]:
            assert "title" in panel
            assert "metric_keys" in panel
            assert "panel_type" in panel
            assert "section" in panel

    def test_safety_metrics_present(self) -> None:
        result = get_safety_dashboard()
        all_keys = []
        for panel in result["panels"]:
            all_keys.extend(panel["metric_keys"])
        assert any("goal_drift" in k for k in all_keys)
        assert any("semantic_drift" in k for k in all_keys)


class TestGetCollapseDashboard:
    def test_returns_dict(self) -> None:
        result = get_collapse_dashboard()
        assert isinstance(result, dict)

    def test_has_name_and_panels(self) -> None:
        result = get_collapse_dashboard()
        assert "name" in result
        assert "panels" in result
        assert result["name"] == "Collapse Dashboard"

    def test_panels_not_empty(self) -> None:
        result = get_collapse_dashboard()
        assert len(result["panels"]) > 0

    def test_collapse_metrics_present(self) -> None:
        result = get_collapse_dashboard()
        all_keys = []
        for panel in result["panels"]:
            all_keys.extend(panel["metric_keys"])
        assert any("entropy" in k for k in all_keys)
        assert any("perplexity" in k for k in all_keys)


class TestGetOverviewDashboard:
    def test_returns_dict(self) -> None:
        result = get_overview_dashboard()
        assert isinstance(result, dict)

    def test_has_name_and_panels(self) -> None:
        result = get_overview_dashboard()
        assert "name" in result
        assert "panels" in result
        assert result["name"] == "Overview Dashboard"

    def test_panels_not_empty(self) -> None:
        result = get_overview_dashboard()
        assert len(result["panels"]) > 0

    def test_has_training_and_safety_sections(self) -> None:
        result = get_overview_dashboard()
        sections = {panel["section"] for panel in result["panels"]}
        assert "training" in sections
        assert "safety" in sections

    def test_overview_metrics_present(self) -> None:
        result = get_overview_dashboard()
        all_keys = []
        for panel in result["panels"]:
            all_keys.extend(panel["metric_keys"])
        assert any("loss" in k for k in all_keys)
        assert any("accuracy" in k for k in all_keys)
        assert any("goal_drift" in k for k in all_keys)
