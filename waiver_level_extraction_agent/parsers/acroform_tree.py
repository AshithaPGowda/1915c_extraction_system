"""
AcroForm Tree Parser.

Depth-n hierarchical parser for CMS 1915(c) waiver PDF AcroForm fields.
Builds a tree structure from colon-separated field paths and provides
query methods for validation, conflict detection, and LLM critic support.

Field name structure:
    svapdxAQ_1:dtPMPrnt:0:svQmPerfMeas:dtDataSource:0:dsRpSma
    └─section─┘└─table──┘│└──nested───┘└───table───┘│└─leaf─┘
                         └─row index                └─row index

Max observed depth: 6 colons (7 segments), designed for up to 8.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TreeNode:
    """
    A node in the AcroForm field tree.

    Attributes:
        name: Segment name at this level (e.g., "locHosp")
        full_path: Full colon-separated path (e.g., "svloc:locHosp")
        value: Normalized value - True for /Yes, False for /Off,
               string for other values, None if branch node
        raw_value: Original value from PDF before normalization
        is_dynamic_index: True if this segment is a numeric row index
        children: Child nodes keyed by segment name
        parent: Reference to parent node (None for root)
    """
    name: str
    full_path: str
    value: bool | str | None = None
    raw_value: str | None = None
    is_dynamic_index: bool = False
    children: dict[str, TreeNode] = field(default_factory=dict)
    parent: TreeNode | None = field(default=None, repr=False)

    def is_leaf(self) -> bool:
        """Return True if this node has no children (leaf node)."""
        return len(self.children) == 0

    def is_active(self) -> bool:
        """Return True if this node's value is True/checked."""
        return self.value is True

    def __repr__(self) -> str:
        val_str = f"={self.value}" if self.value is not None else ""
        children_str = f", {len(self.children)} children" if self.children else ""
        return f"TreeNode({self.full_path}{val_str}{children_str})"


def _normalize_value(raw_value: Any) -> bool | str | None:
    """
    Normalize AcroForm field value to bool or string.

    Args:
        raw_value: Raw value from pypdf field

    Returns:
        True for /Yes, /on, /1, etc.
        False for /Off, empty, /0, etc.
        Original string for other text values
        None if no value
    """
    if raw_value is None:
        return None

    # Handle pypdf name objects
    if hasattr(raw_value, "name"):
        raw_value = raw_value.name

    val_str = str(raw_value).strip()
    val_lower = val_str.lower().lstrip("/")

    # Check for True values
    if val_lower in ("yes", "on", "1", "true"):
        return True

    # Check for False values
    if val_lower in ("off", "0", "no", "false", "", "none"):
        return False

    # Check for radio button selection patterns (field name as value)
    # e.g., /svapdxA1_1:adminOpAgMed indicates selection
    if val_str.startswith("/") and ":" in val_str:
        return True  # A specific option is selected

    # Return as string for other values
    return val_str if val_str else None


def _is_numeric_segment(segment: str) -> bool:
    """Check if a path segment is a numeric row index."""
    return segment.isdigit()


class AcroFormTree:
    """
    Hierarchical tree parser for AcroForm fields.

    Parses colon-separated field names into a navigable tree structure,
    supporting dynamic row detection, ancestor chain validation, and
    conflict detection against LLM-extracted values.

    Example:
        tree = AcroFormTree("waiver.pdf")

        # Direct value lookup
        is_hospital = tree.get_value("svloc:locHosp")  # False

        # Check if field is valid (all ancestors active)
        is_valid = tree.is_ancestor_chain_active("svapdxB4_1:elgGrp435Type")

        # Get all active descendants
        active = tree.get_active_descendants("svapdxB4_1")

        # Validate LLM extraction
        conflicts = tree.get_conflicts({"svloc:locHosp": 1})
    """

    def __init__(self, pdf_path: str | Path):
        """
        Initialize AcroFormTree from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        The tree is built from AcroForm fields. If no AcroForm exists,
        an empty tree is created (no exception raised).
        """
        self.pdf_path = Path(pdf_path)
        self.root = TreeNode(name="__root__", full_path="")
        self._flat_fields: dict[str, TreeNode] = {}
        self._field_count = 0
        self._max_depth = 0
        self._has_acroform = False
        self._metadata: dict[str, Any] = {}

        if self.pdf_path.exists():
            self._parse_pdf()

    def _parse_pdf(self) -> None:
        """Parse AcroForm fields from PDF and build tree."""
        try:
            from pypdf import PdfReader
        except ImportError:
            return

        try:
            reader = PdfReader(str(self.pdf_path))

            # Store metadata
            if reader.metadata:
                self._metadata = dict(reader.metadata)

            # Check if PDF has AcroForm
            try:
                root_obj = reader.trailer["/Root"]
                if hasattr(root_obj, "get_object"):
                    root_obj = root_obj.get_object()
                self._has_acroform = "/AcroForm" in root_obj if hasattr(root_obj, "__contains__") else False
            except Exception:
                self._has_acroform = False

            if not self._has_acroform:
                return

            # Collect all fields
            raw_fields: dict[str, Any] = {}

            # Method 1: Use reader.get_fields() (handles most cases)
            fields = reader.get_fields()
            if fields:
                for field_name, field_data in fields.items():
                    # Get value - try /V first, then /AS
                    value = field_data.get("/V") or field_data.get("/AS") or ""
                    raw_fields[field_name] = value

            # Method 2: Scan page annotations for additional fields
            for page in reader.pages:
                try:
                    if "/Annots" not in page:
                        continue

                    annots = page["/Annots"]
                    if hasattr(annots, "get_object"):
                        annots = annots.get_object()

                    if not annots:
                        continue

                    for annot in annots:
                        try:
                            annot_obj = annot.get_object() if hasattr(annot, "get_object") else annot

                            # Check for Widget annotation
                            if annot_obj.get("/Subtype") != "/Widget":
                                continue

                            # Get field name
                            field_name = annot_obj.get("/T", "")
                            if hasattr(field_name, "get_object"):
                                field_name = field_name.get_object()
                            field_name = str(field_name) if field_name else ""

                            if not field_name or field_name in raw_fields:
                                continue

                            # Get value
                            value = annot_obj.get("/V") or annot_obj.get("/AS") or ""
                            raw_fields[field_name] = value

                        except Exception:
                            continue

                except Exception:
                    continue

            # Build tree from collected fields
            for field_name, raw_value in raw_fields.items():
                self._add_field(field_name, raw_value)

            self._field_count = len(raw_fields)

        except Exception:
            # Graceful failure - empty tree
            pass

    def _add_field(self, full_path: str, raw_value: Any) -> None:
        """
        Add a field to the tree, creating intermediate nodes as needed.

        Args:
            full_path: Full colon-separated field path
            raw_value: Raw value from PDF
        """
        segments = full_path.split(":")
        depth = len(segments)
        self._max_depth = max(self._max_depth, depth)

        current = self.root
        path_so_far = ""

        for i, segment in enumerate(segments):
            path_so_far = segment if not path_so_far else f"{path_so_far}:{segment}"
            is_last = (i == len(segments) - 1)
            is_dynamic = _is_numeric_segment(segment)

            if segment not in current.children:
                # Create new node
                node = TreeNode(
                    name=segment,
                    full_path=path_so_far,
                    is_dynamic_index=is_dynamic,
                    parent=current
                )
                current.children[segment] = node

            current = current.children[segment]

            # Set value on leaf node
            if is_last:
                normalized = _normalize_value(raw_value)
                current.value = normalized
                current.raw_value = str(raw_value) if raw_value else None
                self._flat_fields[full_path] = current

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_value(self, path: str) -> bool | str | None:
        """
        Get the value of a field by its full colon-separated path.

        Args:
            path: Full field path (e.g., "svloc:locHosp")

        Returns:
            True for checked/yes fields
            False for unchecked/off fields
            String for text values
            None if field not found or has no value
        """
        node = self._flat_fields.get(path)
        return node.value if node else None

    def get_node(self, path: str) -> TreeNode | None:
        """
        Get the TreeNode for a given path.

        Args:
            path: Full or partial colon-separated path

        Returns:
            TreeNode if found, None otherwise
        """
        # Check flat lookup first (for leaf nodes)
        if path in self._flat_fields:
            return self._flat_fields[path]

        # Walk the tree for partial paths
        segments = path.split(":")
        current = self.root

        for segment in segments:
            if segment not in current.children:
                return None
            current = current.children[segment]

        return current

    def get_children(self, path: str) -> dict[str, TreeNode]:
        """
        Get all direct children of a node.

        Args:
            path: Path to parent node (e.g., "svconcurrentOp")

        Returns:
            Dict mapping child names to TreeNodes
            Empty dict if path not found or has no children
        """
        node = self.get_node(path)
        return dict(node.children) if node else {}

    def get_subtree(self, path: str) -> dict:
        """
        Get full nested dict of everything under a path.

        Useful for sending to LLM as context for a specific section.

        Args:
            path: Path to subtree root

        Returns:
            Nested dict with structure:
            {
                "name": segment_name,
                "path": full_path,
                "value": normalized_value,
                "children": {child_name: {...}, ...}
            }
        """
        node = self.get_node(path)
        if not node:
            return {}

        return self._node_to_dict(node)

    def _node_to_dict(self, node: TreeNode) -> dict:
        """Convert a TreeNode to nested dict representation."""
        result = {
            "name": node.name,
            "path": node.full_path,
            "value": node.value,
        }

        if node.children:
            result["children"] = {
                name: self._node_to_dict(child)
                for name, child in node.children.items()
            }

        return result

    def get_active_descendants(self, path: str) -> list[str]:
        """
        Get all descendant LEAF paths where value is True AND all ancestors are True.

        This is the key critic method - finds all legitimately active fields
        under a section where the entire ancestor chain supports the value.

        Args:
            path: Path to start from

        Returns:
            List of full paths to active leaf nodes with active ancestor chains
        """
        node = self.get_node(path)
        if not node:
            return []

        active_leaves: list[str] = []
        self._collect_active_descendants(node, active_leaves, ancestor_active=True)
        return active_leaves

    def _collect_active_descendants(
        self,
        node: TreeNode,
        results: list[str],
        ancestor_active: bool
    ) -> None:
        """Recursively collect active descendants."""
        # Determine if this node is active
        # For branch nodes (no value), we consider them active if they exist
        node_active = node.value is True if node.value is not None else True

        # Chain is only active if both this node and ancestors are active
        chain_active = ancestor_active and node_active

        # If this is a leaf and the chain is active and node is True
        if node.is_leaf() and node.value is True and ancestor_active:
            results.append(node.full_path)

        # Recurse to children
        for child in node.children.values():
            self._collect_active_descendants(child, results, chain_active)

    def get_row_count(self, pattern: str) -> int:
        """
        Count dynamic rows under a pattern.

        Args:
            pattern: Path prefix containing dynamic rows
                     (e.g., "svapdxC1_1_prnt:dtServicesPrnt")

        Returns:
            Number of unique row indices found (0 if none)
        """
        node = self.get_node(pattern)
        if not node:
            return 0

        # Count children that are numeric indices
        count = sum(1 for name in node.children.keys() if _is_numeric_segment(name))
        return count

    def get_row(self, pattern: str, index: int) -> dict[str, TreeNode]:
        """
        Get all fields for a specific dynamic row.

        Args:
            pattern: Path prefix containing dynamic rows
            index: Row index (0-based)

        Returns:
            Dict mapping relative field names to TreeNodes for that row
            Empty dict if row not found
        """
        row_path = f"{pattern}:{index}"
        node = self.get_node(row_path)

        if not node:
            return {}

        # Collect all leaf descendants
        result: dict[str, TreeNode] = {}
        self._collect_leaves(node, result, prefix="")
        return result

    def _collect_leaves(
        self,
        node: TreeNode,
        result: dict[str, TreeNode],
        prefix: str
    ) -> None:
        """Recursively collect all leaf nodes under a node."""
        for name, child in node.children.items():
            child_prefix = f"{prefix}:{name}" if prefix else name

            if child.is_leaf():
                result[child_prefix] = child
            else:
                self._collect_leaves(child, result, child_prefix)

    def is_ancestor_chain_active(self, path: str) -> bool:
        """
        Check if the entire ancestor chain from root to this node is active.

        Critical for detecting hallucinated fields - if any ancestor is /Off,
        the descendant values are irrelevant/inactive.

        Args:
            path: Full path to check

        Returns:
            True if ALL ancestors have value True or None (branch nodes)
            False if ANY ancestor has value False
        """
        node = self.get_node(path)
        if not node:
            return False

        # Walk up the tree checking each ancestor
        current = node.parent
        while current and current.full_path:  # Stop at root
            # If ancestor has an explicit False value, chain is inactive
            if current.value is False:
                return False
            current = current.parent

        return True

    def get_ancestor_chain(self, path: str) -> list[tuple[str, bool | str | None]]:
        """
        Get the full ancestor chain for a path.

        Args:
            path: Full path to trace

        Returns:
            List of (path, value) tuples from root to the node
        """
        node = self.get_node(path)
        if not node:
            return []

        chain: list[tuple[str, bool | str | None]] = []
        current = node

        while current and current.full_path:
            chain.append((current.full_path, current.value))
            current = current.parent

        chain.reverse()
        return chain

    def get_section(self, prefix: str) -> dict[str, TreeNode]:
        """
        Get all fields under a section prefix.

        Args:
            prefix: Section prefix (e.g., "svloc", "svapdxB4_1")

        Returns:
            Dict mapping full paths to TreeNodes for all fields under prefix
        """
        result: dict[str, TreeNode] = {}

        for path, node in self._flat_fields.items():
            if path.startswith(prefix + ":") or path == prefix:
                result[path] = node

        return result

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def to_dict(self) -> dict:
        """
        Convert entire tree to nested dict for debugging.

        Returns:
            Full tree as nested dict structure
        """
        return {
            "pdf_path": str(self.pdf_path),
            "field_count": self._field_count,
            "max_depth": self._max_depth,
            "has_acroform": self._has_acroform,
            "tree": {
                name: self._node_to_dict(child)
                for name, child in self.root.children.items()
            }
        }

    def get_conflicts(
        self,
        extracted_fields: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Validate LLM-extracted fields against AcroForm values.

        Args:
            extracted_fields: Dict of {field_name: extracted_value} from LLM

        Returns:
            List of conflict dicts:
            {
                "field": str,
                "acroform_value": bool | str | None,
                "llm_value": Any,
                "conflict_type": "wrong_value" | "hallucinated" |
                                 "inactive_parent" | "missing_child",
                "ancestor_path": list[str],
                "severity": "hard" | "soft"
            }
        """
        conflicts: list[dict[str, Any]] = []

        for field_name, llm_value in extracted_fields.items():
            conflict = self._check_field_conflict(field_name, llm_value)
            if conflict:
                conflicts.append(conflict)

        # Also check for missing children (fields in AcroForm but not extracted)
        # This is soft severity
        for path, node in self._flat_fields.items():
            if node.value is True and path not in extracted_fields:
                # Check if we should have extracted this
                if self.is_ancestor_chain_active(path):
                    # Only flag if it looks like a schema field
                    # (Skip internal AcroForm fields)
                    if not any(x in path.lower() for x in ["dtpm", "qm", "dsfda"]):
                        conflicts.append({
                            "field": path,
                            "acroform_value": True,
                            "llm_value": None,
                            "conflict_type": "missing_child",
                            "ancestor_path": [p for p, _ in self.get_ancestor_chain(path)],
                            "severity": "soft"
                        })

        return conflicts

    def _check_field_conflict(
        self,
        field_name: str,
        llm_value: Any
    ) -> dict[str, Any] | None:
        """Check a single field for conflicts."""
        node = self._flat_fields.get(field_name)

        # Build ancestor path for context
        ancestor_path = [p for p, _ in self.get_ancestor_chain(field_name)]

        # Case 1: Field not in AcroForm at all
        if node is None:
            # Check if this could be a body-text field (not expected in AcroForm)
            # Don't flag these as hallucinated
            return None

        acro_value = node.value

        # Normalize LLM value for comparison
        llm_bool = None
        if isinstance(llm_value, bool):
            llm_bool = llm_value
        elif isinstance(llm_value, (int, float)):
            llm_bool = llm_value == 1 or llm_value == 1.0
        elif isinstance(llm_value, str):
            llm_lower = llm_value.lower().strip()
            if llm_lower in ("1", "yes", "true"):
                llm_bool = True
            elif llm_lower in ("0", "no", "false", ""):
                llm_bool = False

        # Case 2: Ancestor chain is inactive but LLM extracted a value
        if not self.is_ancestor_chain_active(field_name):
            if llm_bool is True or (llm_value is not None and llm_value != 0):
                return {
                    "field": field_name,
                    "acroform_value": acro_value,
                    "llm_value": llm_value,
                    "conflict_type": "inactive_parent",
                    "ancestor_path": ancestor_path,
                    "severity": "hard"
                }

        # Case 3: Values disagree
        if acro_value is not None and llm_bool is not None:
            if acro_value != llm_bool:
                # AcroForm says one thing, LLM says another
                severity = "hard"

                # Soft severity if AcroForm is /Off but LLM has value
                # (could be body text override)
                if acro_value is False and llm_bool is True:
                    severity = "soft"

                return {
                    "field": field_name,
                    "acroform_value": acro_value,
                    "llm_value": llm_value,
                    "conflict_type": "wrong_value",
                    "ancestor_path": ancestor_path,
                    "severity": severity
                }

        return None

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def field_count(self) -> int:
        """Total number of AcroForm fields parsed."""
        return self._field_count

    @property
    def max_depth(self) -> int:
        """Maximum nesting depth (number of segments) found."""
        return self._max_depth

    @property
    def has_acroform(self) -> bool:
        """Whether the PDF has AcroForm fields."""
        return self._has_acroform

    @property
    def static_field_count(self) -> int:
        """Number of fields without dynamic row indices."""
        return sum(
            1 for path in self._flat_fields.keys()
            if not any(_is_numeric_segment(s) for s in path.split(":"))
        )

    @property
    def dynamic_pattern_count(self) -> int:
        """Number of unique dynamic row patterns."""
        patterns = set()
        for path in self._flat_fields.keys():
            # Replace numeric segments with {n}
            pattern = re.sub(r':(\d+):', ':{n}:', path)
            if '{n}' in pattern:
                patterns.add(pattern)
        return len(patterns)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_metadata(pdf_path: str | Path) -> dict[str, Any]:
    """
    Extract metadata from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dict with:
        - title: str
        - author: str
        - producer: str
        - keywords: str
        - waiver_number: str | None (extracted from title)
        - is_cms_form: bool (Author==CMS && iText in Producer)
        - effective_date: str | None (if parseable from title)
    """
    result = {
        "title": "",
        "author": "",
        "producer": "",
        "keywords": "",
        "waiver_number": None,
        "is_cms_form": False,
        "effective_date": None,
    }

    try:
        from pypdf import PdfReader
    except ImportError:
        return result

    try:
        reader = PdfReader(str(pdf_path))
        metadata = reader.metadata or {}

        result["title"] = str(metadata.get("/Title", "") or "")
        result["author"] = str(metadata.get("/Author", "") or "")
        result["producer"] = str(metadata.get("/Producer", "") or "")
        result["keywords"] = str(metadata.get("/Keywords", "") or "")

        # Check if CMS form
        is_cms = result["author"] == "CMS"
        has_itext = "itext" in result["producer"].lower()
        result["is_cms_form"] = is_cms and has_itext

        # Extract waiver number from title (pattern: XX.XXXX.RXX.XX)
        title = result["title"]
        waiver_match = re.search(r'([A-Z]{2})[.\s]*(\d{4})[.\s]*R(\d{2})[.\s]*(\d{2})', title)
        if waiver_match:
            state, num, rev, sub = waiver_match.groups()
            result["waiver_number"] = f"{state}.{num}.R{rev}.{sub}"

        # Try to extract effective date from title
        # Common patterns: "Eff. 01/01/2025", "Effective 1-1-2025", etc.
        date_patterns = [
            r'[Ee]ff(?:ective)?[.\s:]*(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})',
            r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})',
        ]
        for pattern in date_patterns:
            date_match = re.search(pattern, title)
            if date_match:
                m, d, y = date_match.groups()
                result["effective_date"] = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
                break

    except Exception:
        pass

    return result


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Test PDF path
    test_pdf = Path("pdfs_267_latest/AK0260R0607.PDF")

    if not test_pdf.exists():
        print(f"Test PDF not found: {test_pdf}")
        print("Please run from project root directory")
        sys.exit(1)

    print("=" * 70)
    print("ACROFORM TREE PARSER TESTS")
    print("=" * 70)
    print(f"\nTest PDF: {test_pdf}")

    # Parse the tree
    print("\n1. Parsing PDF...")
    tree = AcroFormTree(test_pdf)

    print(f"\n2. BASIC STATS")
    print(f"   Total fields parsed: {tree.field_count}")
    print(f"   Expected: ~1117")
    print(f"   Max depth found: {tree.max_depth}")
    print(f"   Expected: 6")
    print(f"   Static fields: {tree.static_field_count}")
    print(f"   Expected: ~295")
    print(f"   Dynamic patterns: {tree.dynamic_pattern_count}")
    print(f"   Expected: ~381")
    print(f"   Has AcroForm: {tree.has_acroform}")

    print(f"\n3. VALUE LOOKUPS")
    # Test known values
    loc_icfmr = tree.get_value("svloc:locICFMR")
    print(f"   svloc:locICFMR = {loc_icfmr}")
    print(f"   Expected: True")

    loc_hosp = tree.get_value("svloc:locHosp")
    print(f"   svloc:locHosp = {loc_hosp}")
    print(f"   Expected: False")

    miller = tree.get_value("svapdxB4_1:millTrStYN")
    print(f"   svapdxB4_1:millTrStYN = {miller}")

    print(f"\n4. ANCESTOR CHAIN TESTS")
    # Test ancestor chain for active field
    active_path = "svapdxB4_1:elgGrpSpecHomCom"
    is_active_chain = tree.is_ancestor_chain_active(active_path)
    print(f"   is_ancestor_chain_active('{active_path}')")
    print(f"   Result: {is_active_chain}")
    print(f"   Chain: {tree.get_ancestor_chain(active_path)}")

    # Test ancestor chain for field under inactive parent
    print(f"\n   Checking concurrent operation children...")
    conc_children = tree.get_children("svconcurrentOp")
    for name, node in list(conc_children.items())[:5]:
        chain_active = tree.is_ancestor_chain_active(node.full_path)
        print(f"   {name}: value={node.value}, chain_active={chain_active}")

    print(f"\n5. DYNAMIC ROWS")
    # Test row count
    service_prefix = "svapdxC1_1_prnt:dtServicesPrnt"
    row_count = tree.get_row_count(service_prefix)
    print(f"   get_row_count('{service_prefix}') = {row_count}")

    # Get row 0
    row0 = tree.get_row(service_prefix, 0)
    print(f"   Row 0 fields: {len(row0)}")
    for rel_path, node in list(row0.items())[:3]:
        print(f"      {rel_path}: {node.value}")

    print(f"\n6. ACTIVE DESCENDANTS")
    # Get active descendants for B4 section
    b4_active = tree.get_active_descendants("svapdxB4_1")
    print(f"   get_active_descendants('svapdxB4_1')")
    print(f"   Found {len(b4_active)} active leaf nodes")
    for path in b4_active[:5]:
        print(f"      {path}")
    if len(b4_active) > 5:
        print(f"      ... and {len(b4_active) - 5} more")

    print(f"\n7. CONFLICT DETECTION")
    # Test conflict detection with sample extracted values
    test_extracted = {
        # Correct value
        "svloc:locICFMR": 1,
        # Wrong value (AcroForm says False)
        "svloc:locHosp": 1,
        # Hallucinated (inactive parent)
        "svconcurrentOp:conc1915b1": 1,
    }

    print(f"   Testing with: {test_extracted}")
    conflicts = tree.get_conflicts(test_extracted)
    print(f"   Found {len(conflicts)} conflicts:")
    for c in conflicts:
        if c["conflict_type"] != "missing_child":  # Skip soft missing warnings
            print(f"      {c['field']}: {c['conflict_type']} (severity={c['severity']})")
            print(f"         AcroForm={c['acroform_value']}, LLM={c['llm_value']}")

    print(f"\n8. SECTION LOOKUP")
    svloc = tree.get_section("svloc")
    print(f"   get_section('svloc'): {len(svloc)} fields")
    for path, node in svloc.items():
        print(f"      {path}: {node.value}")

    print(f"\n9. SUBTREE")
    subtree = tree.get_subtree("svloc")
    print(f"   get_subtree('svloc'):")
    print(f"   {subtree}")

    print(f"\n10. METADATA")
    metadata = get_metadata(test_pdf)
    print(f"   get_metadata('{test_pdf}'):")
    for k, v in metadata.items():
        print(f"      {k}: {v}")

    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)
