"""Run the scripted simulated-user verification of the recommender adaptation.

SIMULATED USERS ONLY. Software verification; not evidence of effectiveness.
Output: data/processed/neurotwin_simulation_report.json (NEW file)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.neurotwin import config, simulation  # noqa: E402


def main() -> None:
    rep = simulation.run()
    config.SIMULATION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    config.SIMULATION_REPORT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"Wrote {config.SIMULATION_REPORT}")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
