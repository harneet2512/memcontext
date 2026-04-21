from __future__ import annotations

from memcontext.event_bus import CLAIM_CREATED, TURN_ADDED, EventBus


def test_subscribe_and_publish():
    bus = EventBus()
    received: list[dict] = []
    bus.subscribe(TURN_ADDED, lambda p: received.append(p))

    bus.publish(TURN_ADDED, {"turn_id": "tu_123", "session_id": "s1"})
    assert len(received) == 1
    assert received[0]["turn_id"] == "tu_123"


def test_unsubscribe():
    bus = EventBus()
    received: list[dict] = []
    callback = lambda p: received.append(p)
    bus.subscribe(TURN_ADDED, callback)
    bus.unsubscribe(TURN_ADDED, callback)

    bus.publish(TURN_ADDED, {"turn_id": "tu_123"})
    assert len(received) == 0


def test_subscriber_exception_does_not_crash():
    bus = EventBus()
    calls: list[str] = []

    def bad_callback(payload: dict):
        raise ValueError("boom")

    def good_callback(payload: dict):
        calls.append("ok")

    bus.subscribe(CLAIM_CREATED, bad_callback)
    bus.subscribe(CLAIM_CREATED, good_callback)

    bus.publish(CLAIM_CREATED, {"claim_id": "cl_123"})
    assert "ok" in calls


def test_multiple_subscribers():
    bus = EventBus()
    r1: list[dict] = []
    r2: list[dict] = []
    bus.subscribe(TURN_ADDED, lambda p: r1.append(p))
    bus.subscribe(TURN_ADDED, lambda p: r2.append(p))

    bus.publish(TURN_ADDED, {"turn_id": "tu_456"})
    assert len(r1) == 1
    assert len(r2) == 1
