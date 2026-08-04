"""Tests for the Planner tool."""

from data_harness.data.tools.planner import Planner


class TestPlannerBasic:
    def test_add_and_list(self):
        p = Planner()
        p.add(items=["task 1", "task 2"])
        result = p.list()
        assert "task 1" in result
        assert "task 2" in result

    def test_update_status(self):
        p = Planner()
        p.add(items=["task 1"])
        items = p._items
        item_id = items[0]["id"]
        p.update(id=item_id, status="done")
        result = p.list()
        assert "done" in result.lower()

    def test_add_update_list_flow(self):
        p = Planner()
        p.add(items=["step A", "step B"])
        ids = [item["id"] for item in p._items]
        p.update(id=ids[0], status="done")
        result = p.list()
        assert "step A" in result
        assert "step B" in result


class TestPlannerReminders:
    def test_no_reminder_when_no_pending(self):
        p = Planner()
        p.add(items=["task"])
        # Mark done immediately
        p.update(id=p._items[0]["id"], status="done")
        hook = p.reminder_hook
        result = hook(5, 25)
        assert result is None

    def test_gentle_nag_at_4_turns(self):
        p = Planner()
        p.add(items=["pending task"])
        hook = p.reminder_hook
        # Simulate 4 turns without update
        p._turns_since_update = 4
        result = hook(4, 25)
        assert result is not None
        assert len(result) > 0

    def test_firm_nag_at_8_turns(self):
        p = Planner()
        p.add(items=["pending task"])
        p._turns_since_update = 8
        # Get nag at 8
        p._turns_since_update = 8
        result_8 = p.reminder_hook(8, 25)
        assert result_8 is not None

    def test_urgent_nag_at_12_turns(self):
        p = Planner()
        p.add(items=["pending task"])
        p._turns_since_update = 12
        result = p.reminder_hook(12, 25)
        assert result is not None

    def test_reset_on_update(self):
        p = Planner()
        p.add(items=["task"])
        p._turns_since_update = 10
        p.update(id=p._items[0]["id"], status="in_progress")
        assert p._turns_since_update == 0

    def test_reset_on_add(self):
        p = Planner()
        p._turns_since_update = 10
        p.add(items=["new task"])
        assert p._turns_since_update == 0

    def test_no_reminder_below_threshold(self):
        p = Planner()
        p.add(items=["pending"])
        p._turns_since_update = 3
        result = p.reminder_hook(3, 25)
        assert result is None

    def test_turns_since_update_increments_on_reminder_call(self):
        p = Planner()
        p.add(items=["task"])
        p._turns_since_update = 3
        p.reminder_hook(3, 25)
        assert p._turns_since_update == 4
