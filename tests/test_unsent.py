"""Copies of records PS.com did not get yet, kept in the work folder and sent again later."""
import json

import pytest

from physical_sandbox import unsent
from physical_sandbox.errors import CatalogError
from physical_sandbox.push import PushError

RECORD = {"schemaVersion": 3, "name": "デモ", "equipment": {"devices": []}, "traces": []}


class Sender:
    """A stand-in for PS.com: ``answers`` maps a ref to what sending to it does (an exception to raise or a version
    to answer); every call is logged with the record it carried."""

    def __init__(self, **answers):
        self.answers = answers
        self.calls = []

    def __call__(self, path, ref, mode, base):
        self.calls.append((ref, mode, base, json.loads(path.read_text())))
        answer = self.answers.get(ref, "1")
        if isinstance(answer, Exception):
            raise answer
        return f"{ref}:{answer}"


def test_keep_writes_the_record_and_where_it_goes_and_pending_lists_it(tmp_path):
    name = unsent.keep(tmp_path, RECORD, ref="ada/dance", mode="over", base="2")
    assert name == "dance"
    assert json.loads((tmp_path / "unsent" / "dance.record.json").read_text()) == RECORD
    assert json.loads((tmp_path / "unsent" / "dance.send.json").read_text()) == {
        "ref": "ada/dance", "mode": "over", "base": "2"}
    assert unsent.pending(tmp_path) == [{"name": "dance", "ref": "ada/dance", "mode": "over", "base": "2"}]


def test_a_second_copy_of_the_same_name_is_kept_beside_the_first(tmp_path):
    assert unsent.keep(tmp_path, RECORD, ref="ada/dance", mode="over", base="2") == "dance"
    assert unsent.keep(tmp_path, RECORD, ref="ada/dance", mode="over", base="2") == "dance-2"
    assert unsent.keep(tmp_path, RECORD, ref="walk", mode="new", base=None) == "walk"  # no handle while signed out
    assert [p["name"] for p in unsent.pending(tmp_path)] == ["dance", "dance-2", "walk"]


def test_pending_skips_what_is_not_a_whole_pair(tmp_path):
    assert unsent.pending(tmp_path) == []
    unsent.keep(tmp_path, RECORD, ref="ada/dance", mode="new", base=None)
    (tmp_path / "unsent" / "lost.send.json").write_text(json.dumps({"ref": "ada/lost", "mode": "new", "base": None}))
    (tmp_path / "unsent" / "broken.record.json").write_text("{}")
    (tmp_path / "unsent" / "broken.send.json").write_text("{not json")
    assert [p["name"] for p in unsent.pending(tmp_path)] == ["dance"]


def test_send_all_sends_each_and_removes_what_went(tmp_path):
    unsent.keep(tmp_path, RECORD, ref="ada/dance", mode="over", base="2")
    unsent.keep(tmp_path, RECORD, ref="ada/walk", mode="new", base=None)
    sender = Sender(**{"ada/dance": "3"})
    sent, left, notes = unsent.send_all(tmp_path, sender)
    assert sent == [{"name": "dance", "ref": "ada/dance", "version": "3", "mode": "over"},
                    {"name": "walk", "ref": "ada/walk", "version": "1", "mode": "new"}]
    assert left == [] and notes == []
    assert [c[:3] for c in sender.calls] == [("ada/dance", "over", "2"), ("ada/walk", "new", None)]
    assert unsent.pending(tmp_path) == [] and list((tmp_path / "unsent").iterdir()) == []


def test_offline_keeps_every_copy_and_tries_no_more(tmp_path):
    unsent.keep(tmp_path, RECORD, ref="ada/dance", mode="over", base="2")
    unsent.keep(tmp_path, RECORD, ref="ada/walk", mode="new", base=None)
    sender = Sender(**{"ada/dance": CatalogError("connection refused")})
    sent, left, notes = unsent.send_all(tmp_path, sender)
    assert sent == [] and [p["name"] for p in left] == ["dance", "walk"]
    assert len(sender.calls) == 1 and "PS.com に接続できませんでした" in notes[0]
    assert len(unsent.pending(tmp_path)) == 2


def test_a_record_saved_over_elsewhere_goes_as_new_under_the_first_free_suffix(tmp_path):
    unsent.keep(tmp_path, RECORD, ref="ada/dance", mode="over", base="2")
    sender = Sender(**{"ada/dance": PushError("moved", 409, latest="3"),
                       "ada/dance-2": PushError("taken", 409)})
    sent, left, notes = unsent.send_all(tmp_path, sender)
    assert sent == [{"name": "dance", "ref": "ada/dance-3", "version": "1", "mode": "new"}] and left == []
    assert notes == ["ada/dance は別の場所で変わっていたため、ada/dance-3 として新しく保存しました"]
    assert [c[:3] for c in sender.calls] == [("ada/dance", "over", "2"), ("ada/dance-2", "new", None),
                                             ("ada/dance-3", "new", None)]
    assert sender.calls[-1][3] == {**RECORD, "basedOn": "ada/dance:2"}  # the new record says what it came from
    assert unsent.pending(tmp_path) == []


def test_a_refusal_keeps_the_copy_with_the_reason_and_the_next_one_still_goes(tmp_path):
    unsent.keep(tmp_path, RECORD, ref="ada/dance", mode="new", base=None)
    unsent.keep(tmp_path, RECORD, ref="ada/walk", mode="new", base=None)
    sender = Sender(**{"ada/dance": PushError("too-large", 413, note="記録が大きすぎます")})
    sent, left, notes = unsent.send_all(tmp_path, sender)
    assert [s["name"] for s in sent] == ["walk"] and [p["name"] for p in left] == ["dance"]
    assert notes == ["dance を送れませんでした（記録が大きすぎます）"]
    assert [p["name"] for p in unsent.pending(tmp_path)] == ["dance"]


def test_signed_out_on_the_server_keeps_every_copy_and_says_how_to_sign_in(tmp_path):
    unsent.keep(tmp_path, RECORD, ref="ada/dance", mode="over", base="2")
    unsent.keep(tmp_path, RECORD, ref="ada/walk", mode="new", base=None)
    sent, left, notes = unsent.send_all(tmp_path, Sender(**{"ada/dance": PushError("signin", 401)}))
    assert sent == [] and len(left) == 2 and notes == ["psbx login でログインすると送れます"]


def test_a_moved_record_that_cannot_go_as_new_offline_stays_as_it_was(tmp_path):
    unsent.keep(tmp_path, RECORD, ref="ada/dance", mode="over", base="2")
    sender = Sender(**{"ada/dance": PushError("moved", 409, latest="3"), "ada/dance-2": CatalogError("gone")})
    sent, left, notes = unsent.send_all(tmp_path, sender)
    assert sent == [] and left == [{"name": "dance", "ref": "ada/dance", "mode": "over", "base": "2"}]
    assert json.loads((tmp_path / "unsent" / "dance.record.json").read_text()) == RECORD  # no basedOn added


@pytest.mark.parametrize("name", ["a" * 39, "a" * 38])
def test_the_suffixed_name_stays_a_catalog_name(tmp_path, name):
    unsent.keep(tmp_path, RECORD, ref=f"ada/{name}", mode="over", base="1")
    sent, _, _ = unsent.send_all(tmp_path, Sender(**{f"ada/{name}": PushError("moved", 409, latest="2")}))
    written = sent[0]["ref"].split("/")[1]
    assert written.endswith("-2") and len(written) <= 39
