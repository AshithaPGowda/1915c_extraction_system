"""
Review step for the services ADK app: show the extracted services to the user,
let them approve / change / reject, then write the approved rows.

Services are a two-level tree (a parent service and its child services), so
the review uses two-level numbers: a parent is "2", its third child is "2.3".
A parent number covers the parent and ALL its children; a child number covers
just that child.

The source of truth is data/outputs/<KEY>/services.json (written by
finalize_and_save). Applying a review rewrites that file (the original is kept
as services.pre_review.json), refreshes the SQLite rows and, by default,
merges the waiver's rows into the services CSV.
"""

import copy
import datetime
import json

from waiver_level_extraction_agent.shared_libraries.paths import OUTPUTS_DIR

_YEAR_SUFFIXES = ["capitation", "unit", "users", "avg_units_per_user",
                  "avg_cost_per_unit", "component_cost", "total_cost"]
_PARENT_FIELDS = ["hcbs_taxonomy", "service_type", "service_delivery_method",
                  "provider_eligibility", "provider_specifications",
                  "self_direction_authority", "information_assistance"]
_CHILD_FIELDS = ["hcbs_taxonomy"]
_EVIDENCE_ALIASES = {"service_delivery_method": ["service_delivery_method", "delivery_method"]}


# ---------------------------------------------------------------- helpers

def _files(waiver_key: str):
    base = OUTPUTS_DIR / waiver_key
    return base / "services.json", base / "services_evidence.json"


def _load(waiver_key: str):
    sj, ev = _files(waiver_key)
    if not sj.exists():
        return None, None, f"No services.json for {waiver_key} - run the extraction (finalize_and_save) first"
    doc = json.loads(sj.read_text())
    evidence = json.loads(ev.read_text()) if ev.exists() else {}
    return doc, evidence, None


def _cell(v, width: int) -> str:
    t = "" if v is None else str(v).replace("\n", " ").replace("|", "/").strip()
    return t if len(t) <= width else t[: width - 1] + "..."


def _num(v):
    try:
        return float(str(v).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return None


def _money(v) -> str:
    n = _num(v)
    if n is None:
        return _cell(v, 14) if v not in (None, "") else ""
    return f"{n:,.0f}"


def _services(doc) -> list:
    return doc.get("services") or []


def _evidence_for(evidence: dict, index: int, name: str) -> dict:
    items = (evidence or {}).get("services_evidence") or []
    if index < len(items) and items[index].get("service_name") == name:
        return items[index]
    for it in items:
        if it.get("service_name") == name:
            return it
    return {}


def _flags_for(evidence: dict, name: str) -> list:
    crit = (evidence or {}).get("critique") or {}
    out = []
    seen = set()
    for key in ("soft_flag", "human_queue"):
        for f in crit.get(key, []) or []:
            if isinstance(f, dict) and f.get("service") == name:
                sig = (f.get("field"), f.get("reason"))
                if sig not in seen:
                    seen.add(sig)
                    out.append(f)
    return out


def _evidence_text(ev: dict, key: str) -> str:
    for k in _EVIDENCE_ALIASES.get(key, [key]):
        if ev.get(k) not in (None, ""):
            return str(ev[k])
    return ""


def _all_field_keys(row: dict) -> list:
    return [k for k in row if k not in ("row_type", "parent_service")]


def _resolve_field(text: str, row: dict):
    """Match what the user typed to a field key of this row. Accepts the key
    (year_1_unit), the sheet label ('Unit, Year 1') or 'year 1 unit'."""
    from waiver_level_extraction_agent.storage.services_store import sheet_columns

    t = str(text or "").strip().lower().replace(" ", "_").replace(",", "")
    keys = set(_all_field_keys(row)) | set(_PARENT_FIELDS) | {
        f"year_{y}_{s}" for y in range(1, 6) for s in _YEAR_SUFFIXES} | {"service_name"}
    for k in keys:
        if k.lower() == t:
            return k
    if t == "delivery_method":
        return "service_delivery_method"
    # 'Unit, Year 1' / 'unit year 1' -> year_1_unit
    raw = str(text or "").strip().lower()
    for header, internal in sheet_columns():
        if internal and (header.lower() == raw or header.lower().startswith(raw) and len(raw) > 8):
            return "service_delivery_method" if internal == "delivery_method" else internal
    import re
    m = re.match(r"^(?:year[_ ]?(\d)[_ ]+)(\w+)$", t.replace("__", "_")) or re.match(r"^(\w+?)[_ ]+year[_ ]?(\d)$", t)
    if m:
        a, b = m.groups()
        yr, suf = (a, b) if a.isdigit() else (b, a)
        cand = f"year_{yr}_{suf}"
        cand = {"year_%s_cost" % yr: f"year_{yr}_total_cost"}.get(cand, cand)
        if cand in keys:
            return cand
    return None


def _parse_refs(text: str, services: list):
    """'1-3, 2.1, 4.2-4.5' -> (parents:set[int], children:set[(pi,ci)], bad:list). Zero-based."""
    parents, children, bad = set(), set(), []

    def one(tok):
        a, _, b = tok.partition(".")
        if not a.isdigit():
            raise ValueError
        pi = int(a) - 1
        if not 0 <= pi < len(services):
            raise ValueError
        if not b:
            return pi, None
        if not b.isdigit():
            raise ValueError
        ci = int(b) - 1
        if not 0 <= ci < len(services[pi].get("children") or []):
            raise ValueError
        return pi, ci

    for tok in str(text or "").replace(";", ",").split(","):
        tok = tok.strip().lstrip("#")
        if not tok:
            continue
        try:
            if "-" in tok:
                left, right = (x.strip() for x in tok.split("-", 1))
                (p1, c1), (p2, c2) = one(left), one(right)
                if c1 is None and c2 is None:
                    parents.update(range(min(p1, p2), max(p1, p2) + 1))
                elif p1 == p2 and c1 is not None and c2 is not None:
                    children.update((p1, c) for c in range(min(c1, c2), max(c1, c2) + 1))
                else:
                    raise ValueError
            else:
                pi, ci = one(tok)
                (parents.add(pi) if ci is None else children.add((pi, ci)))
        except ValueError:
            bad.append(tok)
    return parents, children, bad


def _coerce(v):
    t = str(v).strip()
    return None if t.lower() in ("null", "none", "") else t


def _year_table(row: dict) -> list:
    lines = ["| Year | Capitation | Unit | Users | Avg units/user | Avg cost/unit | Component cost | Total cost |",
             "|---|---|---|---|---|---|---|---|"]
    n = 0
    for y in range(1, 6):
        vals = [row.get(f"year_{y}_{s}") for s in _YEAR_SUFFIXES]
        if all(v in (None, "") for v in vals):
            continue
        n += 1
        lines.append(f"| {y} | {_cell(vals[0], 10)} | {_cell(vals[1], 18)} | {_cell(vals[2], 10)} | "
                     f"{_cell(vals[3], 10)} | {_cell(vals[4], 10)} | {_money(vals[5])} | {_money(vals[6])} |")
    return lines if n else []


def _next_hint(offset, limit, total):
    if offset + limit < total:
        return f'More services: say "show next page" (services {offset + limit + 1}-{min(offset + 2 * limit, total)}).'
    return "That is the last page."


# ---------------------------------------------------------------- review

def review_services(waiver_key: str, show: str = "", search: str = "",
                    offset: int = 0, limit: int = 50) -> dict:
    """Show the extracted services for review, as numbered markdown tables.
    Print the returned `table` text to the user exactly as given.

    Services are a tree: a parent service ("2") can have child services
    ("2.1", "2.2", ...). With no arguments you get one line per parent service.

    Args:
        waiver_key: The waiver identifier
        show: "" = the table of parent services. "2" = parent 2 in full (its own
            fields, a per-year cost table, flags, and a table of its children).
            "2.3" = child 3 of parent 2 in full.
        search: Only services/children whose name or taxonomy contains any of
            these comma-separated words (case-insensitive); shows each match's number
        offset: First parent (0-based) of the page when listing parents
        limit: Parents per page (default 50). Use limit=0 ONLY when the user asks
            to "show all" - it returns every parent in one table
    """
    doc, evidence, err = _load(waiver_key)
    if err:
        return {"status": "error", "error": err}
    services = _services(doc)
    total_children = sum(len(s.get("children") or []) for s in services)
    head = (f"Waiver {waiver_key}: {len(services)} parent services, {total_children} child services "
            f"(status {doc.get('final_status')}, accuracy {doc.get('accuracy')}).")

    if show:
        parents, children, bad = _parse_refs(show, services)
        if bad or len(parents) + len(children) != 1:
            return {"status": "error", "error": f"Give one service number such as 2 or 2.3 (got: {show})"}
        if parents:
            pi = next(iter(parents))
            svc = services[pi]
            p = svc.get("parent") or {}
            name = p.get("service_name")
            ev = _evidence_for(evidence, pi, name).get("parent_evidence", {})
            lines = [f"### {pi + 1}. {name}", "", "| Field | Value | Evidence |", "|---|---|---|"]
            for k in ["service_name"] + _PARENT_FIELDS:
                v = p.get(k)
                lines.append(f"| `{k}` | {_cell(v, 60) if v not in (None, '') else '*(empty)*'} | "
                             f"{_cell(_evidence_text(ev, k), 70)} |")
            pages = ", ".join(f"{lab} p.{ev[k]}" for k, lab in
                              (("appendix_c_page", "C"), ("appendix_e_page", "E"), ("appendix_j_page", "J")) if ev.get(k))
            if pages:
                lines += ["", f"Evidence pages: {pages}"]
            yt = _year_table(p)
            if yt:
                lines += ["", "**Costs by year** (edit a cell with its key, e.g. `year_1_unit`):", ""] + yt
            flags = _flags_for(evidence, name)
            if flags:
                lines += ["", f"**Flagged by the checker ({len(flags)}):**"]
                lines += [f"- `{f.get('field')}`: {_cell(f.get('reason'), 140)}" for f in flags[:12]]
            kids = svc.get("children") or []
            if kids:
                lines += ["", f"**Child services ({len(kids)}):**", "",
                          "| # | Child service | Taxonomy | Yr 1 unit | Yr 1 users | Yr 1 total cost |", "|---|---|---|---|---|---|"]
                for ci, c in enumerate(kids, 1):
                    lines.append(f"| {pi + 1}.{ci} | {_cell(c.get('service_name'), 50)} | {_cell(c.get('hcbs_taxonomy'), 14)} | "
                                 f"{_cell(c.get('year_1_unit'), 16)} | {_cell(c.get('year_1_users'), 8)} | {_money(c.get('year_1_total_cost'))} |")
            else:
                lines += ["", "No child services."]
            nxt = (f'Open a child with "show {pi + 1}.1". ' if kids else "") + 'Back to the list: "show services".'
            return {"status": "success", "waiver_key": waiver_key, "summary": head, "service_name": name,
                    "next_hint": nxt, "table": "\n".join(lines)}
        pi, ci = next(iter(children))
        svc = services[pi]
        c = svc["children"][ci]
        pname = (svc.get("parent") or {}).get("service_name")
        evs = _evidence_for(evidence, pi, pname).get("children_evidence") or []
        ev = evs[ci] if ci < len(evs) and isinstance(evs[ci], dict) else {}
        lines = [f"### {pi + 1}.{ci + 1}  {c.get('service_name')}  (child of {pi + 1}. {pname})", "",
                 "| Field | Value | Evidence |", "|---|---|---|"]
        for k in ["service_name"] + _CHILD_FIELDS:
            v = c.get(k)
            lines.append(f"| `{k}` | {_cell(v, 60) if v not in (None, '') else '*(empty)*'} | {_cell(_evidence_text(ev, k), 70)} |")
        yt = _year_table(c)
        if yt:
            lines += ["", "**Costs by year** (edit a cell with its key, e.g. `year_1_unit`):", ""] + yt
        return {"status": "success", "waiver_key": waiver_key, "summary": head, "service_name": c.get("service_name"),
                "next_hint": f'Back to its parent: "show {pi + 1}".', "table": "\n".join(lines)}

    if search:
        terms = [t.strip().lower() for t in search.split(",") if t.strip()]
        lines = ["| # | Service | Taxonomy | Yr 1 total cost |", "|---|---|---|---|"]
        hits = 0
        for pi, svc in enumerate(services):
            p = svc.get("parent") or {}
            rows = [(f"{pi + 1}", p)] + [(f"{pi + 1}.{ci + 1}", c) for ci, c in enumerate(svc.get("children") or [])]
            for ref, r in rows:
                hay = f"{r.get('service_name', '')} {r.get('hcbs_taxonomy', '')}".lower()
                if any(t in hay for t in terms):
                    hits += 1
                    lines.append(f"| {ref} | {_cell(r.get('service_name'), 55)} | {_cell(r.get('hcbs_taxonomy'), 14)} | {_money(r.get('year_1_total_cost'))} |")
        return {"status": "success", "waiver_key": waiver_key, "summary": f"{head} {hits} matches for '{search}'.",
                "next_hint": 'Open any match with "show <number>".' if hits else "", "table": "\n".join(lines)}

    if limit <= 0:
        limit = max(len(services), 1)
    page = services[offset: offset + limit]
    lines = ["| # | Service | Type | Delivery | Taxonomy | Children | Yr 1 total cost | Page (C) | Flags |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, svc in enumerate(page, offset + 1):
        p = svc.get("parent") or {}
        name = p.get("service_name")
        ev = _evidence_for(evidence, i - 1, name).get("parent_evidence", {})
        flags = len(_flags_for(evidence, name))
        lines.append(f"| {i} | {_cell(name, 45)} | {_cell(p.get('service_type'), 4)} | {_cell(p.get('service_delivery_method'), 4)} | "
                     f"{_cell(p.get('hcbs_taxonomy'), 14)} | {len(svc.get('children') or [])} | {_money(p.get('year_1_total_cost'))} | "
                     f"{ev.get('appendix_c_page', '')} | {flags or ''} |")
    summary = (f"{head} Showing services {offset + 1}-{offset + len(page)} of {len(services)}. "
               f"Numbers: a parent is N, its children are N.1, N.2, ...")
    return {"status": "success", "waiver_key": waiver_key, "total_parents": len(services), "total_children": total_children,
            "summary": summary, "next_hint": _next_hint(offset, limit, len(services)) +
            ' To see a service and its children say "show 2"; a single child "show 2.1".',
            "next_offset": offset + limit if offset + limit < len(services) else None, "table": "\n".join(lines)}


# ---------------------------------------------------------------- apply

def apply_services_review(waiver_key: str, reject: str = "", approve_only: str = "",
                          edits_json: str = "", write_csv: bool = True,
                          output_csv: str = "", dry_run: bool = False) -> dict:
    """Apply the user's review decisions, then write the approved rows to the
    services CSV. Numbers are the ones in the review tables: "2" is a parent
    service and ALL its children, "2.3" is one child.

    Args:
        waiver_key: The waiver identifier
        reject: Numbers to remove, e.g. "3, 5.2, 7-9". Rejecting a parent also
            removes all of its children.
        approve_only: If given, ONLY these are kept and everything else is
            removed. "2" keeps parent 2 with all its children; "2.1-2.4" keeps
            parent 2's row with just those children.
        edits_json: Field corrections as JSON: row number -> {field: new value},
            e.g. '{"2.2": {"year_1_unit": "hour"}, "3": {"hcbs_taxonomy": "08010"}}'.
            Fields: service_name, hcbs_taxonomy, service_type, service_delivery_method,
            provider_eligibility, provider_specifications, self_direction_authority,
            information_assistance, year_N_capitation/unit/users/avg_units_per_user/
            avg_cost_per_unit/component_cost/total_cost.
        write_csv: Write the reviewed rows to the CSV (default True)
        output_csv: Optional different CSV (file name -> data/outputs/, or a path)
        dry_run: Report what would change without writing anything
    """
    from waiver_level_extraction_agent.shared_libraries.file_utils import log_event
    from waiver_level_extraction_agent.sub_agents.services_agent import flatten_services_to_rows
    from waiver_level_extraction_agent.storage.csv_merge import merge_services_into_csv
    from waiver_level_extraction_agent.storage.services_store import save_services, save_services_run

    doc, evidence, err = _load(waiver_key)
    if err:
        return {"status": "error", "error": err}
    services = _services(doc)

    rej_p, rej_c, bad1 = _parse_refs(reject, services)
    keep_p, keep_c, bad2 = _parse_refs(approve_only, services)
    try:
        edits_raw = json.loads(edits_json) if edits_json else {}
        if not isinstance(edits_raw, dict):
            raise ValueError("must be an object")
    except ValueError as e:
        return {"status": "error", "error": f"edits_json is not valid JSON: {e}"}
    bad3 = []
    new_services = copy.deepcopy(services)
    edited = []
    for ref, changes in edits_raw.items():
        eps, ecs, bad = _parse_refs(str(ref), services)
        if bad or len(eps) + len(ecs) != 1 or not isinstance(changes, dict):
            bad3.append(ref)
            continue
        if eps:
            pi = next(iter(eps))
            row = new_services[pi]["parent"]
        else:
            pi, ci = next(iter(ecs))
            row = new_services[pi]["children"][ci]
        for fld, val in changes.items():
            key = _resolve_field(fld, row)
            if not key:
                bad3.append(f"{ref}:{fld}")
                continue
            old = row.get(key)
            row[key] = _coerce(val)
            if key == "service_name" and eps:
                for c in new_services[pi].get("children") or []:
                    c["parent_service"] = row[key]
            edited.append({"#": ref, "field": key, "old": _cell(old, 40), "new": _cell(row[key], 40)})
    if bad1 or bad2 or bad3:
        return {"status": "error", "error": f"Could not understand these numbers/fields: {bad1 + bad2 + bad3}"}

    # decide what survives (original numbering)
    removed, kept_services = [], []
    for pi, svc in enumerate(new_services):
        pname = (svc.get("parent") or {}).get("service_name")
        kids = svc.get("children") or []
        if approve_only:
            if pi in keep_p:
                keep_ci = set(range(len(kids)))
            elif any(p == pi for p, _ in keep_c):
                keep_ci = {c for p, c in keep_c if p == pi}
            else:
                removed.append({"#": f"{pi + 1}", "service": pname, "children_removed": len(kids)})
                continue
        else:
            keep_ci = set(range(len(kids)))
        if pi in rej_p:
            removed.append({"#": f"{pi + 1}", "service": pname, "children_removed": len(kids)})
            continue
        kept_kids = []
        for ci, c in enumerate(kids):
            if ci in keep_ci and (pi, ci) not in rej_c:
                kept_kids.append(c)
            else:
                removed.append({"#": f"{pi + 1}.{ci + 1}", "service": c.get("service_name"), "child_of": pname})
        svc["children"] = kept_kids
        svc["_orig_index"] = pi
        svc["_kept_children"] = [ci for ci in range(len(kids)) if ci in keep_ci and (pi, ci) not in rej_c]
        kept_services.append(svc)

    if not kept_services:
        return {"status": "error", "error": "That would remove every service. Nothing was changed."}

    summary = {
        "waiver_key": waiver_key,
        "parent_services_kept": len(kept_services),
        "child_services_kept": sum(len(s["children"]) for s in kept_services),
        "removed": removed,
        "edited": edited,
    }
    if dry_run:
        return {"status": "dry_run", **summary}

    # keep the evidence file aligned with the services that remain
    ev_items = (evidence or {}).get("services_evidence") or []
    new_ev = []
    for svc in kept_services:
        oi = svc.pop("_orig_index")
        kept_ci = svc.pop("_kept_children")
        name = (services[oi].get("parent") or {}).get("service_name")
        item = copy.deepcopy(_evidence_for(evidence, oi, name))
        if item and isinstance(item.get("children_evidence"), list) and item["children_evidence"]:
            item["children_evidence"] = [item["children_evidence"][c] for c in kept_ci
                                         if c < len(item["children_evidence"])]
        if item:
            new_ev.append(item)

    sj, evp = _files(waiver_key)
    backup = sj.with_name("services.pre_review.json")
    if not backup.exists():
        backup.write_text(sj.read_text())
    ev_backup = evp.with_name("services_evidence.pre_review.json")
    if evp.exists() and not ev_backup.exists():
        ev_backup.write_text(evp.read_text())

    doc["services"] = kept_services
    doc["final_status"] = "human_reviewed"
    sj.write_text(json.dumps(doc, indent=2, default=str))
    if evp.exists():
        evidence["services_evidence"] = new_ev
        evp.write_text(json.dumps(evidence, indent=2, default=str))
    (sj.parent / "services_review.json").write_text(json.dumps({
        "reviewed_at": datetime.datetime.utcnow().isoformat(),
        "reject": reject, "approve_only": approve_only, "edits": edits_raw,
    }, indent=2))
    log_event(waiver_key, {"agent": "services_human_review", "action": "review_applied",
                           "parents_kept": summary["parent_services_kept"],
                           "children_kept": summary["child_services_kept"],
                           "removed": len(removed), "edited": len(edited)})

    try:  # keep SQLite in step with what was approved
        rows = flatten_services_to_rows(
            services_data=kept_services, waiver_key=waiver_key, state=doc.get("state"),
            waiver_title=doc.get("waiver_title"), file_name=doc.get("file_name"))
        save_services(waiver_key, rows)
        save_services_run(waiver_key, doc.get("accuracy"), doc.get("attempts") or 1,
                          "human_reviewed", len(kept_services))
        summary["db_updated"] = True
    except Exception as e:  # noqa: BLE001
        summary["db_updated"] = f"skipped: {e}"

    summary.update(status="success", services_json=str(sj), original_kept_at=str(backup))
    if write_csv:
        summary["csv"] = merge_services_into_csv(waiver_key, output_csv=output_csv)
    summary["note"] = ("Numbers may have changed because removed services are gone; "
                       "this is the refreshed list.")
    summary["refreshed_review"] = review_services(waiver_key).get("table", "")
    return summary
