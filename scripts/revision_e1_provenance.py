"""Verify retained E1 sources against private historical records; emit safe metadata.

Private transcript paths are CLI inputs, never embedded in the release output.
This verifies the authors' records, not independent registration or clock accuracy.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path


def normalize(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def replay(records, lines):
    content = None
    events = []
    for line in lines:
        record = records[line - 1]
        blocks = [b for b in record["message"]["content"]
                  if b.get("type") == "tool_use" and b.get("name") in ("Write", "Edit")]
        assert len(blocks) == 1
        block = blocks[0]
        args = block["input"]
        if block["name"] == "Write":
            assert content is None
            content = normalize(args["content"])
        else:
            old, new = normalize(args["old_string"]), normalize(args["new_string"])
            assert content is not None and old in content
            if not args.get("replace_all", False):
                assert content.count(old) == 1
                content = content.replace(old, new, 1)
            else:
                content = content.replace(old, new)
        events.append({"operation": block["name"], "timestamp_utc": record["timestamp"]})
    return content, events


def load_records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def verify(source_history, launch_history, root):
    records = load_records(source_history)
    output = {"evidence_type": "author-maintained historical records",
              "verification_scope": "original retained files before release path sanitization; not a comparison against the sanitized runner",
              "independent_registration": False, "newline_normalization": "CRLF/CR to LF",
              "files": []}
    for name, lines in [("scripts/moneyshot/e1_valselect_mwnet.py", [40, 106]),
                        ("docs/E1_PREREG.md", [70, 352, 365, 367, 369])]:
        content, events = replay(records, lines)
        assert content == normalize((Path(root) / name).read_text(encoding="utf-8")), name
        output["files"].append({"path": name, "normalized_content_matches": True, "events": events})
    launches = load_records(launch_history)
    command = launches[19109]
    result = launches[19110]
    command_text = "\n".join(b.get("input", {}).get("command", "")
                             for b in command["message"]["content"])
    result_text = "\n".join(b.get("content", "") for b in result["message"]["content"])
    safe_command = "nohup bash e1_launch.sh 12 > e1_master.log 2>&1 &"
    safe_result = "E1 LAUNCHED, master pid 17055\n== phase 0: repeat-floor gate on THIS machine =="
    assert safe_command in command_text and safe_result in result_text
    output["launch"] = {"command_timestamp_utc": command["timestamp"],
                        "output_timestamp_utc": result["timestamp"],
                        "command_excerpt": safe_command, "output_excerpt": safe_result,
                        "redactions": "remote address, account and surrounding filesystem commands omitted"}
    freeze = datetime.fromisoformat(output["files"][1]["events"][-1]["timestamp_utc"].replace("Z", "+00:00"))
    launch = datetime.fromisoformat(command["timestamp"].replace("Z", "+00:00"))
    output["protocol_to_recorded_launch_hours"] = round((launch - freeze).total_seconds() / 3600, 6)
    return output


def check_release(root):
    """Check public chronology consistency, not authenticity of private records."""
    data = json.loads((Path(root) / "revision_results/e1_provenance.json").read_text(encoding="utf-8"))
    assert data["independent_registration"] is False
    assert len(data["files"]) == 2
    for item in data["files"]:
        assert (Path(root) / item["path"]).is_file()
        assert item["normalized_content_matches"] is True
    freeze = datetime.fromisoformat(data["files"][1]["events"][-1]["timestamp_utc"].replace("Z", "+00:00"))
    launch = datetime.fromisoformat(data["launch"]["command_timestamp_utc"].replace("Z", "+00:00"))
    elapsed = round((launch - freeze).total_seconds() / 3600, 6)
    assert elapsed == data["protocol_to_recorded_launch_hours"] and elapsed > 0
    return {"public_chronology_consistent": True, "private_records_reverified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-history")
    parser.add_argument("--launch-history")
    parser.add_argument("--check", action="store_true", help="Check released chronology without private inputs")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    if args.check:
        result = check_release(args.root)
    else:
        if not args.source_history or not args.launch_history:
            parser.error("supply both historical inputs, or --check")
        result = verify(args.source_history, args.launch_history, args.root)
    print(json.dumps(result, indent=2))
