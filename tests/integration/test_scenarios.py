"""End-to-end regression suite: every simulated scenario through the full pipeline.

These encode the behaviour we promise: falls are confirmed quickly, normal activities
never raise a confirmed fall, and each activity is recognised for most of its duration.
"""

from collections import Counter

import pytest

from motionsense.events.models import EventType
from motionsense.simulation.scenarios import SCENARIOS

MAX_CONFIRM_LATENCY_S = 3.5


def fall_events(pipe, kind: EventType):
    return [e for e in pipe.bus.history(limit=0) if e.type == kind]


@pytest.mark.parametrize("name", [n for n, s in SCENARIOS.items() if s.has_fall])
@pytest.mark.parametrize("seed", [1, 2])
def test_falls_are_confirmed_quickly(run_scenario, name, seed):
    pipe, _ = run_scenario(name, noise_px=2.0, seed=seed)
    confirmed = fall_events(pipe, EventType.FALL_CONFIRMED)
    assert confirmed, f"{name}: fall was not confirmed"
    latency = confirmed[0].ts - SCENARIOS[name].fall_time
    assert 0 <= latency <= MAX_CONFIRM_LATENCY_S
    assert confirmed[0].confidence >= 0.6
    assert confirmed[0].snapshots_url == f"/api/v1/events/{confirmed[0].id}/snapshots"


@pytest.mark.parametrize(
    "name",
    [n for n, s in SCENARIOS.items() if not s.has_fall and n not in ("demo", "lie_down_slowly")],
)
@pytest.mark.parametrize("seed", [1, 2])
def test_normal_activities_never_confirm_a_fall(run_scenario, name, seed):
    pipe, _ = run_scenario(name, noise_px=2.0, seed=seed)
    assert not fall_events(pipe, EventType.FALL_CONFIRMED)
    assert not fall_events(pipe, EventType.INACTIVE_ON_FLOOR)


@pytest.mark.parametrize("name", [n for n, s in SCENARIOS.items() if s.expected_labels])
def test_activity_is_recognised_for_most_of_its_segment(run_scenario, name):
    sc = SCENARIOS[name]
    _, states = run_scenario(name)
    expected = sc.expected_labels[0]
    counts: Counter[str] = Counter()
    for st in states:
        if st.ts < 1.5 or sc.label_at(st.ts) != expected:  # skip warm-up and other segments
            continue
        for p in st.persons:
            counts[p.top_activity] += 1
    total = sum(counts.values())
    assert total > 0
    assert counts[expected] / total >= 0.7, f"{name}: {dict(counts)}"


def test_lying_still_on_floor_escalates_without_witnessed_fall(run_scenario):
    pipe, _ = run_scenario("lie_down_slowly")
    assert not fall_events(pipe, EventType.FALL_CONFIRMED)
    assert len(fall_events(pipe, EventType.INACTIVE_ON_FLOOR)) == 1


def test_fall_then_get_up_reports_recovery(run_scenario):
    pipe, states = run_scenario("fall_get_up")
    assert fall_events(pipe, EventType.FALL_RECOVERED)
    assert states[-1].persons[0].fall_state == "monitoring"


def test_reps_are_counted(run_scenario):
    _, states = run_scenario("push_ups")
    assert states[-1].persons[0].reps["push_up"] == 6
    _, states = run_scenario("squats")
    assert states[-1].persons[0].reps["squat"] == 5


def test_looking_around_attribute(run_scenario):
    _, states = run_scenario("looking_around")
    flags = [st.persons[0].attributes["looking_around"] for st in states if st.ts > 2.5]
    assert sum(flags) / len(flags) > 0.8


def test_demo_scenario_runs_cleanly(run_scenario):
    pipe, states = run_scenario("demo")
    seen = {p.top_activity for st in states for p in st.persons}
    assert {"standing", "walking", "running", "sitting", "squatting", "push_up", "lying"} <= seen
    assert fall_events(pipe, EventType.FALL_CONFIRMED)
