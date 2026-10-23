"""PCB values retain their original edition, conditions, units and table context."""

from __future__ import annotations

import re

STANDARD = re.compile(
    r"(?:GB\s*/?\s*T?|SJ\s*/?\s*T?|IPC\s*-?\s*[A-Z]*|ISO|IEC)[\s-]*\d+(?:\.\d+)*(?:[-:]\d{4}|[A-Z])?", re.I
)
VALUE = re.compile(
    r"(?P<comparator>不小于|不低于|不大于|不超过|至少|最多|小于|大于|≥|≤|>=|<=|>|<|应为|为)?\s*"
    r"(?P<value>[-+]?\d+(?:\.\d+)?(?:\s*[~～—–至-]\s*\d+(?:\.\d+)?)?)\s*"
    r"(?P<unit>μm|µm|um|mm|mil|oz|nm|N/mm|N/m|MPa|GPa|℃|°C|%|GHz|MHz|kHz|ps|ns|ms|Ω|ohm|V|A|W)",
    re.I,
)
COMPARATORS = {
    "不小于": ">=",
    "不低于": ">=",
    "至少": ">=",
    "≥": ">=",
    "不大于": "<=",
    "不超过": "<=",
    "最多": "<=",
    "≤": "<=",
    "小于": "<",
    "大于": ">",
    "应为": "=",
    "为": "=",
}
CELL_VALUE = re.compile(
    r"^\s*(?P<comparator>不小于|不低于|不大于|不超过|至少|最多|小于|大于|≥|≤|>=|<=|>|<|=)?\s*"
    r"(?P<value>[-+]?\d+(?:\.\d+)?(?:\s*[~～—–至-]\s*\d+(?:\.\d+)?)?)\s*(?:\[?[*a-z†‡]\]?)?\s*$"
)
HEADER_UNIT = re.compile(
    r"[（(\[/]\s*(N/mm|N/m|μm|µm|um|mm|mil|oz|nm|MPa|GPa|℃|°C|%|GHz|MHz|kHz|ps|ns|ms|Ω|ohm|V|A|W)\s*[）)\]]?", re.I
)


def _cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _separator(line):
    return "|" in line and all(re.fullmatch(r":?-{2,}:?", cell) for cell in _cells(line))


def table_context(text):
    lines = text.splitlines()
    headers, footnotes = [], []
    for index, line in enumerate(lines):
        if "|" in line and index + 1 < len(lines) and _separator(lines[index + 1]):
            headers.append(line.strip())
        if re.match(r"\s*(?:注\s*\d*\s*[:：]|注\s*[:：]?\s|\[\^|[*†‡]|[a-z]\))", line):
            footnotes.append(line.strip())
    return {"table_headers": headers, "table_footnotes": footnotes}


def enrich_table_child(child, parent):
    context = table_context(parent)
    if "|" not in child:
        return child, {"table_headers": [], "table_footnotes": []}
    if len(context["table_headers"]) > 1:
        # Select the table containing this child's data rows; never attach
        # unrelated column headers merely because tables share a parent.
        lines = parent.splitlines()
        starts = [index for index in range(len(lines) - 1) if "|" in lines[index] and _separator(lines[index + 1])]
        matching = []
        child_rows = {row.strip() for row in child.splitlines() if "|" in row and not _separator(row)}
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else len(lines)
            block = lines[start:end]
            if child_rows.intersection(line.strip() for line in block[2:]):
                matching.append("\n".join(block))
        context = table_context("\n\n".join(matching))
    headers = [line for line in context["table_headers"] if line not in child]
    notes = [line for line in context["table_footnotes"] if line not in child]
    prefixed = []
    for header in headers:
        prefixed.extend([header, "| " + " | ".join("---" for _ in _cells(header)) + " |"])
    return "\n".join(prefixed + [child] + notes), context


def link_continued_tables(documents):
    """Carry page locators only for explicitly marked continued PDF tables."""
    previous = {}
    for document in documents:
        metadata = document.metadata
        source = str(metadata.get("source_path") or metadata.get("file_path") or "")
        context = table_context(document.text)
        prior = previous.get(source)
        if prior and re.search(r"续表|表\s*\d+\s*[（(]续[）)]|table\s+\d+.*continued", document.text, re.I):
            if not context["table_headers"] and "|" in document.text:
                header = prior["table_headers"][-1]
                prefix = header + "\n| " + " | ".join("---" for _ in _cells(header)) + " |\n"
                text = prefix + document.text + "\n" + "\n".join(prior["table_footnotes"])
                if hasattr(document, "set_content"):
                    document.set_content(text)
                else:
                    document.text = text
                metadata.update(
                    table_headers=[header],
                    table_footnotes=prior["table_footnotes"],
                    table_context_page=prior["page"],
                    continued_from_page=prior["page"],
                )
                context = table_context(document.text)
        if context["table_headers"]:
            previous[source] = dict(context, page=str(metadata.get("page_label") or metadata.get("page") or ""))
        else:
            previous.pop(source, None)
    return documents


def extract_qualified_facts(text, metadata=None):
    metadata = metadata or {}
    standard_match = STANDARD.search(str(metadata.get("source_path", ""))) or STANDARD.search(text)
    standard = standard_match.group(0).strip() if standard_match else str(metadata.get("standard", ""))
    edition = str(metadata.get("edition") or metadata.get("standard_version") or "")
    version = re.search(r"[-:](\d{4})$|(?<=\d)([A-Z])$", standard)
    if not edition and version:
        edition = next(group for group in version.groups() if group)
    context = table_context(text)
    for key in context:
        context[key] = list(dict.fromkeys(list(metadata.get(key) or []) + context[key]))
    facts = []
    active_header = []
    lines = re.split(r"(?<=[。；;])|\n", text)
    active_conditions = str(metadata.get("conditions") or "")
    for index, sentence in enumerate(lines):
        local_standard = STANDARD.search(sentence)
        if local_standard:
            standard = local_standard.group(0).strip()
            version = re.search(r"[-:](\d{4})$|(?<=\d)([A-Z])$", standard, re.I)
            edition = (
                next((group for group in version.groups() if group), "")
                if version
                else str(metadata.get("edition") or "")
            )
        condition = re.search(r"(?:当.+?时|在.+?(?:条件下|情况下)|对于[^，,。]+)", sentence)
        if condition:
            active_conditions = condition.group(0)
        if "|" in sentence and index + 1 < len(lines) and _separator(lines[index + 1]):
            active_header = _cells(sentence)
            continue
        if _separator(sentence):
            continue
        base = {
            "standard": standard,
            "edition": edition,
            "clause": str(metadata.get("section") or metadata.get("parent_title") or ""),
            "scope": str(metadata.get("scope") or ""),
            "conditions": active_conditions,
            "source_path": str(metadata.get("source_path") or ""),
            "page": str(metadata.get("page_label") or metadata.get("page") or ""),
            **context,
        }
        for match in VALUE.finditer(sentence):
            prefix = sentence[: match.start()].strip(" \t|，,:：")
            parameter = re.split(r"[，,：:|]", prefix)[-1].strip()[-80:]
            parameter = re.sub(r"^(?:规定|要求)\s*", "", STANDARD.sub("", parameter).strip())
            raw_op = match.group("comparator") or ""
            facts.append(
                {
                    **base,
                    "parameter": parameter,
                    "comparator": COMPARATORS.get(raw_op, raw_op),
                    "value": match.group("value"),
                    "unit": match.group("unit"),
                    "original_value": match.group(0).strip(),
                    "evidence": sentence.strip(),
                }
            )
        if "|" not in sentence:
            continue
        header = active_header or (_cells(context["table_headers"][0]) if len(context["table_headers"]) == 1 else [])
        cells = _cells(sentence)
        if len(cells) != len(header):
            continue
        for column, (label, cell) in enumerate(zip(header, cells)):
            unit_match, number = HEADER_UNIT.search(label), CELL_VALUE.fullmatch(cell)
            if not unit_match or not number:
                continue
            raw_op = number.group("comparator") or ""
            row_scope = cells[0] if column and not CELL_VALUE.fullmatch(cells[0]) else base["scope"]
            facts.append(
                {
                    **base,
                    "scope": row_scope,
                    "parameter": HEADER_UNIT.sub("", label).strip(" ()（）[]/"),
                    "comparator": COMPARATORS.get(raw_op, raw_op),
                    "value": number.group("value"),
                    "unit": unit_match.group(1),
                    "original_value": cell,
                    "evidence": sentence.strip(),
                    "column_header": label,
                    "table_column": column,
                    "row_label": cells[0],
                    "unit_source": "column_header",
                }
            )
    return facts


def detect_fact_conflicts(facts):
    """Expose differing requirements; do not collapse editions/conditions."""
    groups = {}
    for fact in facts:
        key = (fact.get("parameter"), fact.get("unit"))
        groups.setdefault(key, []).append(fact)
    return [
        values
        for values in groups.values()
        if len(
            {
                (
                    item.get("standard"),
                    item.get("edition"),
                    item.get("scope"),
                    item.get("conditions"),
                    item.get("comparator"),
                    item.get("value"),
                )
                for item in values
            }
        )
        > 1
    ]
