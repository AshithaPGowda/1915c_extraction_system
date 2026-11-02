"""
Per-service AcroForm parsing and service-name correlation.

The CMS 1915(c) application form stores per-service checkboxes in
position-indexed arrays (position 0..N-1), not by service name:

    svapdxC1_1_prnt:dtServicesPrnt:{pos}:svapdxC1_1a_prnt:
        rbSvcIncInAprvdW    service included in approved waiver
        svcDelMthPtcDir     delivery method: participant-directed
        svcDelMthPvdrMgd    delivery method: provider managed
        svcPvdLegResp       may be provided by legally responsible person
        svcPvdRel           may be provided by relative
        svcPvdLegGrd        may be provided by legal guardian

    svapdxE1_6:dtPDServices:{k}:svcEmpAuth / svcBudAuth
        E-1g self-direction authority, indexed over ONLY the
        participant-directed services (delivery method 1 or 3),
        in the same relative order as Appendix C.

    svapdxE1_9:dtPDInfoServices:{k}:svcInfoAsst
        E-1j information & assistance, indexed over all services.

    svapdxJ2_4 .. svapdxJ2_8 : dtEstFctrD:{row}:cndEstDInCap
        J-2d capitation checkbox for years 1..5, one row per service.
        Not present in every PDF (some Appendix J tables are flat text).

Correlation between positions and service names uses the order of first
appearance of each service name in the PDF text (the form renders
positions sequentially), verified by count match against the AcroForm.
"""

import re

from .acroform_tree import AcroFormTree


# AcroForm subtree names
_C1_ROOT = "svapdxC1_1_prnt"
_C1_TABLE = "dtServicesPrnt"
_C1_ENTRY = "svapdxC1_1a_prnt"
_E1G_ROOT = "svapdxE1_6"
_E1G_TABLE = "dtPDServices"
_E1J_ROOT = "svapdxE1_9"
_E1J_TABLE = "dtPDInfoServices"
_J2_YEAR_ROOTS = ["svapdxJ2_4", "svapdxJ2_5", "svapdxJ2_6", "svapdxJ2_7", "svapdxJ2_8"]
_J2_TABLE = "dtEstFctrD"
_J2_CAP = "cndEstDInCap"


def _children(node: dict) -> dict:
    return node.get("children") or {}


def _leaf_value(node: dict, name: str):
    child = _children(node).get(name)
    if child is None:
        return None
    return child.get("value")


def _delivery_method_code(ptc_dir, pvdr_mgd) -> int | None:
    """Map the two delivery-method checkboxes to the sheet's 1/2/3 coding."""
    if ptc_dir and pvdr_mgd:
        return 3
    if ptc_dir:
        return 1
    if pvdr_mgd:
        return 2
    return None


def _provider_eligibility_digits(leg_resp, rel, leg_grd) -> str | None:
    """
    Provider-eligibility digits present in the AcroForm, e.g. "2" or "123".

    The sheet codes only the applicable digits (Relative-only rows appear
    as "2", not "020"), so comparison should be set-based — use
    provider_eligibility_set() for that.
    """
    digits = ""
    if leg_resp:
        digits += "1"
    if rel:
        digits += "2"
    if leg_grd:
        digits += "3"
    return digits or None


def provider_eligibility_set(value) -> frozenset:
    """Digits 1-3 present in a provider-eligibility value, however coded."""
    if value is None:
        return frozenset()
    return frozenset(c for c in str(value) if c in "123")


def extract_services_acroform(pdf_path: str) -> dict:
    """
    Parse all per-service AcroForm fields from a waiver PDF.

    Returns a dict:
        has_acroform         bool
        service_count        number of positions in the C-1 service table
        services             {position(int): {included, delivery_method,
                              provider_eligibility}}
        self_direction       [ {employer, budget, authority} ] in PD-table order
        info_assistance      [bool] in table order (usually one per service)
        capitation           {year(int 1-5): [0/1, ...]} in table row order,
                             only for years present in the AcroForm
    """
    result = {
        "has_acroform": False,
        "service_count": 0,
        "services": {},
        "self_direction": [],
        "info_assistance": [],
        "capitation": {},
    }

    tree = AcroFormTree(pdf_path)
    if not tree.has_acroform:
        return result
    result["has_acroform"] = True

    root = tree.to_dict()["tree"]

    # --- Appendix C-1 service table -------------------------------------
    c1 = root.get(_C1_ROOT)
    table = _children(_children(c1).get(_C1_TABLE, {})) if c1 else {}
    for pos_str, pos_node in table.items():
        try:
            pos = int(pos_str)
        except ValueError:
            continue
        entry = _children(pos_node).get(_C1_ENTRY)
        if entry is None:
            continue
        result["services"][pos] = {
            "included": bool(_leaf_value(entry, "rbSvcIncInAprvdW")),
            "delivery_method": _delivery_method_code(
                _leaf_value(entry, "svcDelMthPtcDir"),
                _leaf_value(entry, "svcDelMthPvdrMgd"),
            ),
            "provider_eligibility": _provider_eligibility_digits(
                _leaf_value(entry, "svcPvdLegResp"),
                _leaf_value(entry, "svcPvdRel"),
                _leaf_value(entry, "svcPvdLegGrd"),
            ),
        }
    result["service_count"] = len(result["services"])

    # --- E-1g self-direction authority (participant-directed subset) ----
    e1g = root.get(_E1G_ROOT)
    pd_table = _children(_children(e1g).get(_E1G_TABLE, {})) if e1g else {}
    for pos_str in sorted(pd_table, key=lambda s: int(s) if s.isdigit() else 0):
        entry = pd_table[pos_str]
        emp = bool(_leaf_value(entry, "svcEmpAuth"))
        bud = bool(_leaf_value(entry, "svcBudAuth"))
        authority = 3 if (emp and bud) else 1 if emp else 2 if bud else None
        result["self_direction"].append(
            {"employer": emp, "budget": bud, "authority": authority}
        )

    # --- E-1j information & assistance -----------------------------------
    e1j = root.get(_E1J_ROOT)
    ia_table = _children(_children(e1j).get(_E1J_TABLE, {})) if e1j else {}
    for pos_str in sorted(ia_table, key=lambda s: int(s) if s.isdigit() else 0):
        result["info_assistance"].append(
            bool(_leaf_value(ia_table[pos_str], "svcInfoAsst"))
        )

    # --- J-2d capitation per year ----------------------------------------
    for year_idx, root_name in enumerate(_J2_YEAR_ROOTS, start=1):
        j_node = root.get(root_name)
        if j_node is None:
            continue
        j_table = _children(_children(j_node).get(_J2_TABLE, {}))
        rows = []
        for pos_str in sorted(j_table, key=lambda s: int(s) if s.isdigit() else 0):
            cap = _leaf_value(j_table[pos_str], _J2_CAP)
            if cap is None:
                continue
            rows.append(1 if cap else 0)
        if rows:
            result["capitation"][year_idx] = rows

    return result


_SPEC_MARKER = "C-1/C-3: Service Specification"


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).lower()
    # PDF text extraction sometimes inserts a space after a line-wrapped
    # "/" (e.g. "Family/ Consumer Training" vs the CSV's "Family/Consumer
    # Training") - collapse it so that's never the reason a name fails to
    # match (see correlate_services / _spec_block_order).
    return re.sub(r"\s*/\s*", "/", text)


# Official HCBS Taxonomy category labels (MMRR 2014, Exhibit 1). The PDF
# form prints categories with a numeric code and its own casing
# ("17 Other Services", "08 Home-Based Services"); the output column always
# uses these canonical labels.
CANONICAL_CATEGORIES = [
    "Case management",
    "Round-the-clock services",
    "Supported employment",
    "Day services",
    "Nursing",
    "Home-delivered meals",
    "Rent and food expenses for live-in caregiver",
    "Home-based services",
    "Caregiver support",
    "Other mental health and behavioral services",
    "Other health and therapeutic services",
    "Services supporting participant direction",
    "Participant training",
    "Equipment, technology, and modifications",
    "Nonmedical transportation",
    "Community transition services",
    "Other services",
    "Unknown",
]

_CANONICAL_BY_KEY = {
    re.sub(r"[^a-z]", "", c.lower()): c for c in CANONICAL_CATEGORIES
}


def canonical_category(raw: str | None) -> str | None:
    """
    Normalize a PDF taxonomy category to its official label.

    "17 Other Services" -> "Other services"
    "08 Home-Based Services" -> "Home-based services"
    Unrecognized values are returned with the leading code stripped.
    """
    if not raw:
        return None
    stripped = re.sub(r"^\s*\d+\s*", "", str(raw)).strip()
    if not stripped:
        return None
    key = re.sub(r"[^a-z]", "", stripped.lower())
    return _CANONICAL_BY_KEY.get(key, stripped)


_TAXONOMY_PAIR_RE = re.compile(
    r"Category\s+(\d)\s*:\s*\n(.*?)\nSub-Category\s+\1\s*:\s*\n(.*?)\n",
    re.DOTALL,
)
_SERVICE_NAME_LINE_RE = re.compile(
    r"(?:Alternate Service Title \(if any\)|Service Title|Service)\s*:\s*\n\s*(.+?)\s*\n",
)


def extract_service_taxonomies(pdf_path: str, service_names: list[str]) -> dict:
    """
    Deterministically parse the HCBS Taxonomy tables from the C-1/C-3
    Service Specification blocks.

    The form renders, per service block:

        HCBS Taxonomy:
        Category 1:
        04 Day Services
        Sub-Category 1:
        04020 day habilitation
        ...

    Old waivers (pre-taxonomy forms) render the same table with empty
    values - those services genuinely have NO taxonomy in the PDF, and the
    parsed result is None (LLM values for them are hallucinations).

    Returns:
        { service_name: {"taxonomy": str | None,
                         "detail": [{"category", "subcategory"}, ...],
                         "block_found": bool} }
    """
    result = {
        name: {"taxonomy": None, "detail": [], "block_found": False}
        for name in service_names
    }

    try:
        import fitz  # pymupdf
    except ImportError:
        return result

    try:
        doc = fitz.open(pdf_path)
        page_texts = [doc[i].get_text() for i in range(len(doc))]
        doc.close()
    except Exception as e:
        print(f"Warning: taxonomy parse failed: {e}")
        return result

    spec_pages = [i for i, t in enumerate(page_texts) if _SPEC_MARKER in t]
    if not spec_pages:
        return result
    region_text = "\n".join(page_texts[spec_pages[0]: spec_pages[-1] + 3])

    # Each service's specification starts at the marker; a block runs until
    # the next marker.
    blocks = region_text.split(_SPEC_MARKER)[1:]
    norm_names = {_normalize_text(n): n for n in service_names}

    for block in blocks:
        # Identify the service this block belongs to via its title lines.
        block_names = [
            _normalize_text(m) for m in _SERVICE_NAME_LINE_RE.findall(block)
        ]
        matched = None
        for bn in block_names:
            if bn in norm_names:
                matched = norm_names[bn]
                break
        if matched is None:
            # Substring fallback (alternate titles with extra whitespace etc.)
            block_norm = _normalize_text(block[:2000])
            candidates = [
                orig for norm, orig in norm_names.items() if norm in block_norm
            ]
            if len(candidates) == 1:
                matched = candidates[0]
        if matched is None or result[matched]["block_found"]:
            continue

        entry = result[matched]
        entry["block_found"] = True

        tax_idx = block.find("HCBS Taxonomy")
        if tax_idx < 0:
            continue
        # Taxonomy table sits between the header and the service definition.
        end_idx = block.find("Service Definition", tax_idx)
        tax_text = block[tax_idx: end_idx if end_idx > 0 else tax_idx + 2000]

        categories: list[str] = []
        for _, category, subcategory in _TAXONOMY_PAIR_RE.findall(tax_text):
            category = re.sub(r"\s+", " ", category).strip()
            subcategory = re.sub(r"\s+", " ", subcategory).strip()
            if not category:
                continue
            # Keep the raw form-printed value (with code) in the detail;
            # the output column uses the official label.
            entry["detail"].append(
                {"category": category, "subcategory": subcategory or None}
            )
            label = canonical_category(category)
            if label and label not in categories:
                categories.append(label)

        entry["taxonomy"] = "; ".join(categories) if categories else None

    return result


# Page footer/header text that pymupdf interleaves into the extracted text
# whenever a label and its value straddle a page break, e.g.:
#   "Provider Category:\nApplication for 1915(c) HCBS Waiver: CT.0993... \n
#    Page 67 of 216\n05/21/2025\n\nIndividual"
# Stripping it before parsing prevents label/value pairs from being missed
# purely because of where the page happened to break.
_PAGE_FOOTER_RE = re.compile(
    r"Application for 1915\(c\) HCBS Waiver:.*?\nPage \d+ of \d+\n[\d/]+\n?"
)

_PROVIDER_SPEC_MARKER = "C-1/C-3: Provider Specifications for Service"
_PROVIDER_SPEC_NAME_RE = re.compile(r"Service Name\s*:\s*\n?\s*(.+?)\s*\n")
_PROVIDER_CATEGORY_RE = re.compile(
    r"Provider Category\s*:\s*\n?\s*(Agency|Individual)", re.IGNORECASE
)
_SERVICE_TYPE_RE = re.compile(r"Service Type\s*:\s*\n?\s*(.+?)\s*\n")
_SERVICE_TYPE_LABELS = {
    "statutory service": 1,
    "extended state plan service": 2,
    "supports for participant direction": 3,
    "other service": 4,
}


def _load_clean_page_text(pdf_path: str) -> str | None:
    """Full document text, joined across pages, with page-footer boilerplate
    stripped. Returns None if the PDF can't be opened."""
    try:
        import fitz  # pymupdf
    except ImportError:
        return None
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(doc[i].get_text() for i in range(len(doc)))
        doc.close()
    except Exception as e:
        print(f"Warning: PDF text load failed: {e}")
        return None
    return _PAGE_FOOTER_RE.sub("", text)


def extract_provider_specifications(pdf_path: str, service_names: list[str]) -> dict:
    """
    Deterministically parse `provider_specifications` (1 = agency;
    2 = individual; 3 = both) from the per-service "C-1/C-3: Provider
    Specifications for Service" blocks.

    The form renders one such block per (service, provider category) row:

        C-1/C-3: Provider Specifications for Service
        Service Type: Statutory Service
        Service Name: Respite
        Provider Category:
        Agency
        Provider Type:
        Private providers

    A service with both an Agency row and an Individual row elsewhere in
    the document gets both categories unioned together (-> 3).

    Returns:
        { service_name: {"value": 1|2|3|None, "categories": [...],
                         "block_found": bool} }
    """
    result = {
        name: {"value": None, "categories": [], "block_found": False}
        for name in service_names
    }
    text = _load_clean_page_text(pdf_path)
    if text is None:
        return result

    chunks = text.split(_PROVIDER_SPEC_MARKER)[1:]
    norm_names = {_normalize_text(n): n for n in service_names}
    found: dict[str, set] = {}

    for chunk in chunks:
        name_m = _PROVIDER_SPEC_NAME_RE.search(chunk)
        cat_m = _PROVIDER_CATEGORY_RE.search(chunk)
        if not name_m or not cat_m:
            continue
        matched = norm_names.get(_normalize_text(name_m.group(1)))
        if not matched:
            continue
        found.setdefault(matched, set()).add(cat_m.group(1).capitalize())

    for name, categories in found.items():
        entry = result[name]
        entry["block_found"] = True
        entry["categories"] = sorted(categories)
        if categories == {"Agency"}:
            entry["value"] = 1
        elif categories == {"Individual"}:
            entry["value"] = 2
        elif categories == {"Agency", "Individual"}:
            entry["value"] = 3

    return result


def extract_service_type(pdf_path: str, service_names: list[str]) -> dict:
    """
    Deterministically parse `service_type` (1 = Statutory Service;
    2 = Extended State Plan Service; 3 = Supports for Participant Direction;
    4 = Other Service) from the "Service Type:" label printed at the top of
    each service's C-1/C-3 Service Specification block (the same block
    `extract_service_taxonomies` reads for the HCBS Taxonomy table).

    Returns:
        { service_name: {"value": 1-4|None, "raw_label": str|None,
                         "block_found": bool} }
    """
    result = {
        name: {"value": None, "raw_label": None, "block_found": False}
        for name in service_names
    }
    text = _load_clean_page_text(pdf_path)
    if text is None:
        return result

    blocks = text.split(_SPEC_MARKER)[1:]
    norm_names = {_normalize_text(n): n for n in service_names}

    for block in blocks:
        head = block[:1000]
        block_names = [
            _normalize_text(m) for m in _SERVICE_NAME_LINE_RE.findall(head)
        ]
        matched = None
        for bn in block_names:
            if bn in norm_names:
                matched = norm_names[bn]
                break
        if matched is None or result[matched]["block_found"]:
            continue

        entry = result[matched]
        entry["block_found"] = True

        type_m = _SERVICE_TYPE_RE.search(block[:500])
        if type_m:
            raw = re.sub(r"\s+", " ", type_m.group(1)).strip()
            entry["raw_label"] = raw
            entry["value"] = _SERVICE_TYPE_LABELS.get(raw.lower())

    return result


def _spec_block_order(pdf_path: str, names: list[str]) -> list[str | None]:
    """
    Service names in document order, one per "C-1/C-3: Service
    Specification" block - the same block-splitting + per-block
    "Service:"/"Service Title:" line lookup already proven correct in
    extract_service_type/extract_provider_specifications/
    extract_service_taxonomies, reused here as the position-correlation
    signal instead of a raw substring search.

    CONFIRMED FIX (2026-10-02): the previous signal (a per-name search for
    the first character offset where that name's text appears ANYWHERE in
    the whole C-1/C-3 region) silently misorders names whenever one
    service's name is mentioned incidentally - in another service's
    narrative, a cross-reference, a child name containing the parent's
    name as a substring, etc. - before its own spec block. Confirmed wrong
    this way for AL0001R1000 ("Supported Living Service" landed at
    position 17 instead of the correct 23, verified against a rendered PDF
    page) and for multiple services each in CA0336R0510/MT0148R0700/
    MN0166R0801. Each service has exactly ONE spec block, in the same
    document order as its AcroForm position, so anchoring on that block's
    own labeled name line removes the substring-search ambiguity entirely
    - verified to produce zero missing/duplicate matches across all 4 of
    those waivers once _normalize_text also collapses "/ " spacing.

    Returns one entry per spec block found (which should equal
    len(names) when the document is well-formed, but is NOT truncated or
    padded to match - callers must check the length themselves), each
    either a matched name from `names` or None if that block's labeled
    name didn't match any known name (e.g. a wording/spacing mismatch).
    """
    text = _load_clean_page_text(pdf_path)
    if text is None:
        return []
    blocks = text.split(_SPEC_MARKER)[1:]
    norm_names = {_normalize_text(n): n for n in names}
    order: list[str | None] = []
    for block in blocks:
        head = block[:1000]
        block_names = [_normalize_text(m) for m in _SERVICE_NAME_LINE_RE.findall(head)]
        matched = None
        for bn in block_names:
            if bn in norm_names:
                matched = norm_names[bn]
                break
        order.append(matched)
    return order


def _spec_block_start_pages(pdf_path: str) -> list[int]:
    """Page index where each "C-1/C-3: Service Specification" block starts,
    in document order - aligned 1:1 with _spec_block_order()'s return list
    (both come from the same page-by-page scan, same order, just one
    returns the matched name and this one the page it started on)."""
    try:
        import fitz
    except ImportError:
        return []
    try:
        doc = fitz.open(pdf_path)
        page_texts = [doc[i].get_text() for i in range(len(doc))]
        doc.close()
    except Exception:
        return []
    starts: list[int] = []
    for i, t in enumerate(page_texts):
        starts.extend([i] * t.count(_SPEC_MARKER))
    return starts


def _position_widget_pages(pdf_path: str) -> dict[int, int]:
    """Page index of each AcroForm position's own service-eligibility
    widgets (svcPvdRel - present exactly once per position in every waiver
    checked so far). Used ONLY as an independent cross-check signal inside
    correlate_services() - never as a correlation method on its own. The
    two page numbers it's compared against (a block's text-anchored start
    page, and the AcroForm's own widget page) come from two completely
    unrelated extraction paths, so they agreeing is real corroboration,
    not circular reasoning."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return {}
    try:
        reader = PdfReader(pdf_path)
    except Exception:
        return {}
    pos_to_page: dict[int, int] = {}
    for pageno, page in enumerate(reader.pages):
        annots = page.get("/Annots")
        if not annots:
            continue
        for a in annots:
            obj = a.get_object()
            fq = []
            cur = obj
            while cur is not None:
                t = cur.get("/T")
                if t:
                    fq.insert(0, str(t))
                cur = cur.get("/Parent")
                if cur is not None:
                    cur = cur.get_object()
            fqname = ":".join(fq)
            if fqname.endswith("svcPvdRel") and _C1_TABLE in fqname:
                parts = fqname.split(":")
                try:
                    pos = int(parts[2])
                except (IndexError, ValueError):
                    continue
                pos_to_page.setdefault(pos, pageno)
    return pos_to_page


def correlate_services(
    service_names: list[str],
    pdf_path: str,
    acroform: dict | None = None,
) -> dict:
    """
    Correlate service names with AcroForm positions.

    The C-1 form table renders positions sequentially, and so does each
    service's "C-1/C-3: Service Specification" block - see
    _spec_block_order. Confidence is "high" only when every name matched a
    block's own labeled name, the block count equals the AcroForm position
    count, AND (see position_range_check below) every position's own
    checkbox widgets actually fall within that position's own block's page
    range - a self-verification pass CONFIRMED (2026-10-02, DC0334R0503)
    to correctly tolerate services whose own narrative section spans many
    pages (so a widget can be several pages after its block's own "Service:"
    label - checked against the NEXT block's start, not an exact-page
    match) while still catching a genuinely wrong position assignment
    automatically, without a human having to notice a surprising value
    first (which is how every correlation bug this far was actually
    found).

    Returns:
        {
          "confidence": "high" | "medium" | "none",
          "count_match": bool,
          "position_range_check": {"performed": bool,
                                    "out_of_range_services": [...]},
          "hints": { service_name: {position, delivery_method,
                                    provider_eligibility,
                                    self_direction_authority,
                                    information_assistance,
                                    capitation: {year: 0/1}} }
        }
    """
    if acroform is None:
        acroform = extract_services_acroform(pdf_path)

    # CONFIRMED BUG (2026-10-05, batch run on CT0426R0402): position_range_check
    # used to only get added to `result` on the happy path, past every early
    # "bail out, can't trust positions" return below - so a caller reading
    # result["position_range_check"] unconditionally (as it should be able
    # to, per this function's own docstring) got a KeyError on exactly the
    # waivers where correlation failed outright, i.e. the ones most in need
    # of being handled, not crashed on. Always present now, from the first
    # line, so every return path - happy or bailed-out - has it.
    result = {
        "confidence": "none",
        "count_match": False,
        "position_range_check": {"performed": False, "out_of_range_services": []},
        "hints": {},
    }
    if not acroform["has_acroform"] or not acroform["services"]:
        return result

    count_match = len(service_names) == acroform["service_count"]
    result["count_match"] = count_match

    if not count_match:
        # Positions cannot be trusted one-to-one; bail out rather than guess.
        return result

    block_order = _spec_block_order(pdf_path, service_names)
    if len(block_order) != len(service_names):
        # Block count doesn't line up with the position count either - the
        # document doesn't have exactly one spec block per service, so rank
        # i's block can't be trusted to mean position i. Bail rather than
        # guess, same as the count_match check above.
        return result

    matched = [n for n in block_order if n is not None]
    all_found = len(set(matched)) == len(service_names)
    result["confidence"] = "high" if all_found else "medium"

    # Fill any block that didn't match a known name (wording/spacing
    # mismatch) or duplicated an already-used name with whichever service
    # name is still unaccounted for, in original list order - a narrower,
    # more targeted guess than the old approach's "dump every unfound name
    # at the very end" since it only affects the specific slot(s) that
    # didn't match, not the whole ordering.
    remaining = [n for n in service_names if n not in set(matched)]
    ordered: list[str] = []
    seen: set[str] = set()
    for slot in block_order:
        if slot is not None and slot not in seen:
            ordered.append(slot)
            seen.add(slot)
        else:
            ordered.append(remaining.pop(0))

    positions = sorted(acroform["services"])
    # Participant-directed subset, in position order, for the E-1g table.
    pd_positions = [
        p for p in positions
        if acroform["services"][p].get("delivery_method") in (1, 3)
    ]

    # Independent cross-check: does each position's own checkbox widget
    # actually sit on a page within that SAME rank's spec-block page range
    # (its own block's start page up to the next block's start page)? A
    # mismatch means the name<->position pairing itself is wrong, not just
    # uncertain - downgrade confidence automatically instead of only
    # catching it when a human happens to notice a suspicious value.
    block_start_pages = _spec_block_start_pages(pdf_path)
    widget_pages = _position_widget_pages(pdf_path)
    out_of_range: list[str] = []
    range_check_performed = bool(block_start_pages) and bool(widget_pages) and len(block_start_pages) == len(block_order)
    if range_check_performed:
        for rank, name in enumerate(ordered):
            pos = positions[rank]
            wp = widget_pages.get(pos)
            if wp is None:
                continue
            start = block_start_pages[rank]
            # Upper bound is INCLUSIVE of the next block's own start page -
            # CONFIRMED (2026-10-02, AL0001R1000 "Speech and Language
            # Therapy", MN0166R0801 "Extended Community First Services and
            # Supports") a service's checkbox widgets can legitimately land
            # on the exact same page where the NEXT service's own "Service:"
            # label text also begins (both earlier-appearing text and later
            # widgets can share one physical page) - an exclusive upper
            # bound flagged both as false positives despite already being
            # confirmed correct. Still catches a real multi-page/position
            # jump like the original "Supported Living Service" bug, which
            # was off by 6 positions and many pages, not one.
            end = block_start_pages[rank + 1] if rank + 1 < len(block_start_pages) else None
            if wp < start or (end is not None and wp > end):
                out_of_range.append(name)
    result["position_range_check"] = {
        "performed": range_check_performed,
        "out_of_range_services": out_of_range,
    }
    if out_of_range and result["confidence"] == "high":
        result["confidence"] = "medium"

    for rank, name in enumerate(ordered):
        pos = positions[rank]
        svc = acroform["services"][pos]
        hint = {
            "position": pos,
            "delivery_method": svc.get("delivery_method"),
            "provider_eligibility": svc.get("provider_eligibility"),
            "self_direction_authority": None,
            "information_assistance": None,
            "capitation": {},
        }
        if pos in pd_positions:
            k = pd_positions.index(pos)
            if k < len(acroform["self_direction"]):
                hint["self_direction_authority"] = (
                    acroform["self_direction"][k]["authority"]
                )
        if len(acroform["info_assistance"]) == len(positions):
            hint["information_assistance"] = (
                1 if acroform["info_assistance"][rank] else None
            )
        for year, rows in acroform["capitation"].items():
            if len(rows) == len(positions):
                hint["capitation"][year] = rows[rank]
        result["hints"][name] = hint

    return result
