"""CLI: python -m datagen [--data PATH] <command> [steps...]

    --data PATH          where data is generated and read (default: ./data; or $COUNTERPOINT_DATA)

    preview              small version of every pool + report in data/preview/ (look at examples)
    sources              download inputs, build exclusion lists, set U (§12)
    build [steps...]     render datasets; steps: shared e1 t0 e2 (default: all)
    export [e1|e2]       write training files (default: both)
    check                CI checks (§10.1) -> data/reports/ci_report.json
    audit                render-audit contact sheets (§10.2) -> data/reports/audit/
    stats                statistics + plots -> data/reports/stats/index.html
    view [port]          browser viewer on localhost (tunnel with ssh -L): browse/verify, blind audit (§10.3), stats
    report               static report bundle (stats, galleries, contact sheets, CI) -> data/reports.tar.gz
    all                  sources, build, export, check, report
"""

import os
import sys
from pathlib import Path


def take_data_arg(argv):
    """Strip --data PATH / --data=PATH from argv and export it before any datagen module is imported."""
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--data" and i + 1 < len(argv):
            os.environ["COUNTERPOINT_DATA"] = str(Path(argv[i + 1]).expanduser().resolve())
            i += 2
            continue
        if a.startswith("--data="):
            os.environ["COUNTERPOINT_DATA"] = str(Path(a.split("=", 1)[1]).expanduser().resolve())
        else:
            out.append(a)
        i += 1
    return out


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "sources":
        from .sources import fetch_all
        fetch_all()
    elif cmd == "build":
        from .build import build
        build(rest or None)
    elif cmd == "export":
        from .export import export
        export(rest or ("e1", "e2"))
    elif cmd == "check":
        from .checks import run_checks
        return 0 if run_checks() else 1
    elif cmd == "audit":
        from .checks import audit_sheets
        audit_sheets()
    elif cmd == "stats":
        from .stats import run_stats
        run_stats()
    elif cmd == "preview":
        from .preview import preview
        preview()
    elif cmd == "report":
        from .report import build_report
        build_report()
    elif cmd == "view":
        from .viewer import view
        view(int(rest[0]) if rest else 7860)
    elif cmd == "all":
        for c in (["sources"], ["build"], ["export"], ["check"], ["report"]):
            rc = main(c)
            if rc:
                return rc
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(take_data_arg(sys.argv[1:])))
