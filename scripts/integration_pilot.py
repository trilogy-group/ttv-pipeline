"""Emit an offline pilot using real TTV services and deterministic test providers."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.contracts.generation_v1 import canonical_bytes, digest, seal
from tests.test_generation_integration import FakeProvider, approval, request, service


def main(root):
    root = Path(root).resolve()
    svc = service(root / "ttv")
    req = request()
    req["scenes"][2]["continuity"]["mode"] = "continue_from_previous"
    board = {
        "approved": True,
        "project_id": req["project_id"],
        "revision_id": "storyboard-pilot",
        "scenes": copy.deepcopy(req["scenes"]),
    }
    req["storyboard"] = {"revision_id": board["revision_id"], "sha256": digest(board)}
    req = seal(req)
    plan = svc.plan(req)
    ap = approval(plan)
    job = svc.approve(ap)
    result = svc.run(job["id"])
    assert result["terminal_status"] == "succeeded"
    assert svc.approve(ap)["id"] == job["id"]
    assert len(FakeProvider.calls) == 6
    regen = copy.deepcopy(req)
    regen.update(
        request_id="regeneration",
        idempotency_key="regeneration",
        revision=2,
        supersedes_request_id=req["request_id"],
    )
    regen["scenes"] = [regen["scenes"][1]]
    regen["regeneration"] = {
        "base_result_ids": [result["result_id"]],
        "supersedes_take_ids": [result["scenes"][1]["takes"][0]["take_id"]],
        "reason": "Replace middle scene",
    }
    regen = seal(regen)
    new_plan = svc.plan(regen)
    new_ap = approval(new_plan)
    new_ap.update(approval_id="regen-approval", idempotency_key="regen-approval")
    new_ap = seal(new_ap)
    replacement = svc.run(svc.approve(new_ap)["id"])
    assert replacement["terminal_status"] == "succeeded"
    assert len(FakeProvider.calls) == 9
    out = root / "handoff"
    out.mkdir(parents=True, exist_ok=True)
    docs = {
        "storyboard": board,
        "request": req,
        "plan": plan,
        "approval": ap,
        "result": result,
        "regeneration-request": regen,
        "regeneration-plan": new_plan,
        "regeneration-approval": new_ap,
        "regeneration-result": replacement,
    }
    for name, doc in docs.items():
        (out / f"{name}.json").write_bytes(canonical_bytes(doc))
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
