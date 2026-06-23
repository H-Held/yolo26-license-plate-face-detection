#!/usr/bin/env python
"""
============================================================================
 run_campaign.py  —  unattended 2-model training campaign
============================================================================
Trains TWO large (YOLO26l) models on ALL GPUs of the profile, each with the
full pipeline (real batch probe -> coarse train until early stop -> fine-tune
the best checkpoint -> evaluate on test -> log per-class best confidence):

  1) campaign_l_1280 : 1280x1280 tiles, uncompressed
  2) campaign_l_640  : the same tiles down-scaled ("compressed") to 640x640

It is designed to KEEP RUNNING IF THE CONNECTION DROPS:
  * launch it detached:  nohup python run_campaign.py > campaign.log 2>&1 &
  * every stage of every run records state to disk; relaunching the exact same
    command resumes from where it stopped (already-finished stages are skipped).

Run it directly as a script (NOT inside a notebook) so multi-GPU DDP works.
"""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import train_pipeline as P
import build_resized_dataset as RS

CAMPAIGN_STATE = C.ROOT / "runs" / "campaign_state.json"
SRC_DATASET = C.ROOT / "dataset_face_lp"            # 1280 tiles (built by 03b)
DATASET_640 = C.ROOT / "dataset_face_lp_640"        # built here if missing


def _load():
    return json.loads(CAMPAIGN_STATE.read_text()) if CAMPAIGN_STATE.exists() else {"runs": {}}

def _save(s):
    CAMPAIGN_STATE.parent.mkdir(parents=True, exist_ok=True)
    CAMPAIGN_STATE.write_text(json.dumps(s, indent=2))


def main():
    P.log("############ CAMPAIGN START ############")
    P.log(f"python={sys.executable}")
    device, nproc = P.resolve_devices()
    P.log(f"using device={device}  ({nproc} GPU)")
    assert SRC_DATASET.joinpath("dataset.yaml").exists(), \
        f"missing {SRC_DATASET}/dataset.yaml — run notebooks 00..03b first"

    state = _load()

    # ----- build the 640 ("compressed") dataset once -----
    if not DATASET_640.joinpath("dataset.yaml").exists():
        P.log("building 640x640 compressed dataset (one-time)...")
        RS.build(str(SRC_DATASET), str(DATASET_640), 640)
    else:
        P.log(f"640 dataset already present: {DATASET_640}")

    # ----- the two runs -----
    specs = [
        P.make_spec("campaign_l_1280", data=SRC_DATASET / "dataset.yaml",
                    imgsz=1280, model_size="l", cache="auto"),
        P.make_spec("campaign_l_640", data=DATASET_640 / "dataset.yaml",
                    imgsz=640, model_size="l", cache="auto"),
    ]

    # Each run is retried several times. A retry RESUMES the interrupted training
    # from its last checkpoint (see train_pipeline._resume_if_possible), so a
    # transient crash mid-training never throws away progress and never aborts the
    # campaign. Crucially, we ALWAYS move on to the next model, even if one run
    # keeps failing — so the 2nd model is never skipped.
    MAX_ATTEMPTS = 8
    for spec in specs:
        name = spec["name"]
        if state["runs"].get(name, {}).get("complete"):
            P.log(f"--- {name}: already complete, skipping ---")
            continue
        for attempt in range(1, MAX_ATTEMPTS + 1):
            P.log(f"################ {name}: attempt {attempt}/{MAX_ATTEMPTS} ################")
            t0 = time.time()
            try:
                res = P.run_full(spec)
                state["runs"][name] = {"complete": True, "final_best": res["final_best"],
                                       "attempts": attempt,
                                       "minutes": round((time.time() - t0) / 60, 1)}
                _save(state)
                P.log(f"################ finished {name} in "
                      f"{state['runs'][name]['minutes']} min ################")
                break
            except Exception as e:
                state["runs"][name] = {"complete": False, "attempts": attempt, "error": repr(e)}
                _save(state)
                P.log(f"!!!! {name} attempt {attempt} failed: {e!r}")
                if attempt < MAX_ATTEMPTS:
                    P.log(f"     retrying {name} in 30s (will RESUME from last checkpoint)...")
                    time.sleep(30)
                else:
                    P.log(f"!!!! {name} gave up after {MAX_ATTEMPTS} attempts — "
                          f"MOVING ON to the next model")

    incomplete = [n for n, r in state["runs"].items() if not r.get("complete")] \
        + [s["name"] for s in specs if s["name"] not in state["runs"]]
    if incomplete:
        P.log(f"############ CAMPAIGN INCOMPLETE: {incomplete} ############")
        P.log("(re-launch / supervisor will resume these)")
        sys.exit(1)
    P.log("############ CAMPAIGN COMPLETE ############")
    P.log("per-class best confidence log: " + str(C.ROOT / "runs" / "best_conf_log.csv"))


if __name__ == "__main__":
    main()
