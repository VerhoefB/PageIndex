import math
import os
import json
import re

from .utils import llm_completion, extract_json, count_tokens, get_text_of_pdf_pages

DEBUG_PROGRESS = True

def progress(message):
    if DEBUG_PROGRESS:
        print(message, flush=True)

def split_large_section_with_llm(title, text, max_tokens, model=None):
    current_tokens = count_tokens(text, model=model)
    expected_chunks = max(2, math.ceil(current_tokens / max_tokens))

    prompt = f"""
You are given a document section.

Current section token count: {current_tokens}
Maximum allowed tokens per child section: {max_tokens}
Expected number of child sections: approximately {expected_chunks}

Your objective is to split this section into the FEWEST number of coherent child sections such that:

1. Every resulting child section is approximately below {max_tokens} tokens.
2. The number of child sections is minimized.
3. Use approximately {expected_chunks} child sections.
4. Do not create significantly more child sections unless absolutely necessary.
5. Splits should occur at natural topic or paragraph boundaries.
6. Do not create very small sections.
7. Do not create sections containing only a title or a few sentences.
8. Preserve the original document meaning and structure.
9. Prefer larger chunks over smaller chunks.

Do NOT invent a deep hierarchy.
Do NOT summarize.
Do NOT rewrite text.
Do NOT return the text itself.

Instead identify where new chunks should start.

Return JSON only.

For each chunk:
- Use a short descriptive title.
- The title should reflect the main topic of the chunk.
- Do NOT use generic titles such as "Part 1", "Part 2", "Section A", "Chunk 1", "Chunk 2" etc.
- Prefer titles similar to document section headings.
- Use terminology from the original section text.

Example:

[
  {{
    "title": "ESG Risk Drivers and Transmission Channels",
    "start_phrase": "exact phrase where this chunk starts"
  }},
  {{
    "title": "Materiality Assessment Methodology",
    "start_phrase": "exact phrase where this chunk starts"
  }}
]

Section title:
{title}

Section text:
{text}
"""

    response = llm_completion(model=model, prompt=prompt)
    result = extract_json(response)

    if isinstance(result, list):
        return result

    return []

def generate_node_summary(title, text, model=None, summary_token_threshold=200):
    """
    Generate a concise retrieval summary for a node/chunk.

    Used for:
    - normal leaf nodes
    - generated intro nodes
    - LLM-split child nodes
    """
    text = (text or "").strip()
    title = title or "Untitled section"

    if not text:
        return ""

    token_count = count_tokens(text, model=model)

    progress(
        f"[summary check] tokens={token_count} "
        f"threshold={summary_token_threshold} "
        f"title={title!r}"
    )

    if token_count <= summary_token_threshold:
        return text

    # PageIndex-like behavior: if text is already short, use it directly.
    if count_tokens(text, model=model) <= summary_token_threshold:
        return text

    prompt = f"""
You are given a section from a PDF document.

Write a concise retrieval summary for this section.

The summary will be used in a hierarchical document index. Another LLM will read the node title and this summary to decide whether this node is relevant to a user question.

Rules:
- Write 1 concise paragraph.
- Be specific and factual.
- Mention the main topics, entities, metrics, risks, policies, tables, or concepts covered.
- Do not add information that is not in the section text.
- Do not use bullet points.
- Return only the summary.

Node title:
{title}

Section text:
{text}
"""

    progress(f"[LLM summary start] tokens={token_count} title={title!r}")
    summary = llm_completion(model=model, prompt=prompt).strip()
    progress(f"[LLM summary done] title={title!r} chars={len(summary)}")
    return summary


def generate_parent_summary_from_children(title, child_nodes, model=None):
    """
    Generate a summary for a parent node that was split into children.

    The parent no longer has its own chunk text, so summarize from child summaries.
    """
    title = title or "Untitled section"

    child_context = "\n".join(
        f"- {child.get('title', '')}: {child.get('summary', '')}"
        for child in child_nodes
        if child.get("summary")
    )

    if not child_context.strip():
        return ""

    prompt = f"""
You are given a parent section from a PDF document and summaries of its child sections.

Write a concise retrieval summary for the parent section.

Rules:
- Write 1 concise paragraph.
- Base the summary only on the child titles and summaries.
- Mention the main themes covered by the children.
- Do not add information that is not present.
- Do not use bullet points.
- Return only the summary.

Parent title:
{title}

Child sections:
{child_context}
"""

    return llm_completion(model=model, prompt=prompt).strip()

def normalize_for_match(text):
    text = text or ""
    text = text.lower()
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"^\s*\d+\s*", "", text)
    return re.sub(r"[^a-z0-9]", "", text)


def normalize_with_map(text):
    normalized = ""
    mapping = []

    for i, ch in enumerate(text or ""):
        ch = ch.lower()
        ch = ch.replace("ﬁ", "fi").replace("ﬂ", "fl")

        for c in ch:
            if c.isalnum():
                normalized += c
                mapping.append(i)

    return normalized, mapping


def find_phrase_position(text, phrase):
    if not text or not phrase:
        return -1

    text_key, mapping = normalize_with_map(text)
    phrase_key, _ = normalize_with_map(phrase)

    if not phrase_key:
        return -1

    pos = text_key.find(phrase_key)

    if pos == -1:
        return -1

    return mapping[pos]

def slice_text_by_llm_boundaries(text, boundaries):
    valid = []

    for item in boundaries:
        title = item.get("title", "Untitled subsection")
        start_phrase = item.get("start_phrase", "")

        pos = find_phrase_position(text, start_phrase)

        if pos == -1:
            title_positions = find_title_heading_candidates(text, title)
            if title_positions:
                pos = title_positions[0]

        if pos != -1:
            valid.append({
                "title": title,
                "pos": pos,
            })

    deduped = {}
    for item in valid:
        deduped[item["pos"]] = item

    valid = sorted(deduped.values(), key=lambda x: x["pos"])

    chunks = []

    for i, item in enumerate(valid):
        start = item["pos"]
        end = valid[i + 1]["pos"] if i + 1 < len(valid) else len(text)
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "title": item["title"],
                "text": chunk_text,
            })

    return chunks


def split_until_within_limit(title, text, max_tokens, model=None, depth=0, max_depth=10):
    token_count = count_tokens(text, model=model)

    if token_count <= max_tokens:
        return [{
            "title": title,
            "text": text,
        }]

    if depth >= max_depth:
        raise ValueError(
            f"Could not split section below token limit after {max_depth} iterations. "
            f"Title: {title}, tokens: {token_count}, limit: {max_tokens}"
        )

    boundaries = split_large_section_with_llm(
        title=title,
        text=text,
        max_tokens=max_tokens,
        model=model,
    )

    chunks = slice_text_by_llm_boundaries(text, boundaries)

    if len(chunks) < 2:
        raise ValueError(
            f"LLM failed to split section into multiple chunks. "
            f"Title: {title}, tokens: {token_count}, limit: {max_tokens}"
        )

    final_chunks = []

    for chunk in chunks:
        final_chunks.extend(
            split_until_within_limit(
                title=chunk["title"],
                text=chunk["text"],
                max_tokens=max_tokens,
                model=model,
                depth=depth + 1,
                max_depth=max_depth,
            )
        )

    return final_chunks


def get_leaf_nodes_in_order(structure):
    leaf_nodes = []

    def walk(nodes):
        for node in nodes:
            if node.get("nodes"):
                walk(node["nodes"])
            else:
                leaf_nodes.append(node)

    walk(structure)
    return leaf_nodes

def get_all_nodes_in_order(structure):
    all_nodes = []

    def walk(nodes):
        for node in nodes:
            all_nodes.append(node)
            if node.get("nodes"):
                walk(node["nodes"])

    walk(structure)
    return all_nodes


def get_next_boundary_node(current_node, all_nodes):
    current_id = current_node.get("node_id")

    for i, node in enumerate(all_nodes):
        if node.get("node_id") == current_id:
            if i + 1 < len(all_nodes):
                return all_nodes[i + 1]
            return None

    return None

def find_title(text, title):
    if not text or not title:
        return -1

    title_key = normalize_for_match(title)
    if not title_key:
        return -1

    lines = text.splitlines()
    char_pos = 0

    for i in range(len(lines)):
        combined = ""

        for j in range(i, min(i + 4, len(lines))):
            combined += " " + lines[j].strip()
            combined_key = normalize_for_match(combined)

            if combined_key == title_key or combined_key.startswith(title_key) or title_key in combined_key:
                return char_pos + max(lines[i].find(lines[i].strip()), 0)

        char_pos += len(lines[i]) + 1

    # fallback: heading appears mid-line
    text_key = normalize_for_match(text)
    pos_key = text_key.find(title_key)

    if pos_key == -1:
        return -1

    # approximate original character position
    pattern = r"\s*[-–—]?\s*".join(re.escape(w) for w in re.findall(r"[A-Za-z0-9]+", title))
    match = re.search(pattern, text, flags=re.IGNORECASE)

    if match:
        return match.start()

    return -1

def find_title_strict_heading(text, title):
    if not text or not title:
        return -1

    title_key = normalize_for_match(title)
    if not title_key:
        return -1

    lines = text.splitlines()
    char_pos = 0

    for i in range(len(lines)):
        combined = ""

        for j in range(i, min(i + 4, len(lines))):
            combined += " " + lines[j].strip()

            cleaned = re.sub(
                r"^\s*\d+(\.\d+)*\.?\s*",
                "",
                combined.strip()
            )

            combined_key = normalize_for_match(cleaned)

            if combined_key == title_key or combined_key.startswith(title_key):
                return char_pos + max(lines[i].find(lines[i].strip()), 0)

        char_pos += len(lines[i]) + 1

    return -1

def estimate_title_end_pos(text, title, start_pos):
    """
    Estimate where a matched title ends in the original text.

    This is used so the next node can start searching after the previous
    title, not after an arbitrary offset like +5 or +50.
    """
    if start_pos == -1 or not title:
        return start_pos

    search_window = text[start_pos:start_pos + max(len(title) * 4, 300)]

    words = re.findall(r"[A-Za-z0-9]+", title or "")
    if not words:
        return start_pos

    pattern = r"\s*[-–—.]?\s*".join(re.escape(w) for w in words)
    match = re.search(pattern, search_window, flags=re.IGNORECASE)

    if match:
        return start_pos + match.end()

    # fallback
    return start_pos + len(title)

def find_normalized_occurrences(text, needle):
    text_key, mapping = normalize_with_map(text)
    needle_key, _ = normalize_with_map(needle)

    if not text_key or not needle_key:
        return []

    positions = []
    start = 0

    while True:
        pos = text_key.find(needle_key, start)
        if pos == -1:
            break

        positions.append(mapping[pos])
        start = pos + 1

    return positions

def choose_title_near_phrase(title_candidates, phrase_pos, window=1200):
    if phrase_pos == -1:
        return title_candidates[0] if title_candidates else -1

    before_phrase = [
        pos for pos in title_candidates
        if pos <= phrase_pos and phrase_pos - pos <= window
    ]

    if before_phrase:
        return max(before_phrase)

    return -1

def has_usable_phrase(value):
    if not value:
        return False
    key, _ = normalize_with_map(value)
    return bool(key)


def find_specific_title_before_phrase(text, title, phrase_pos, lookback=700):
    if phrase_pos == -1 or not title:
        return -1

    start = max(0, phrase_pos - lookback)
    window_text = text[start:phrase_pos]

    words = re.findall(r"[A-Za-z0-9]+", title or "")
    if not words:
        return -1

    pattern = r"\s*[-–—.]?\s*".join(re.escape(w) for w in words)
    matches = list(re.finditer(pattern, window_text, flags=re.IGNORECASE))

    if matches:
        return start + matches[-1].start()

    return -1

def find_intro_parent_start(text, parent_node, min_pos=0, lookback=1200):
    parent_title = get_node_heading_for_match(parent_node)
    parent_phrase = parent_node.get("start_phrase", "")

    search_text = text[min_pos:]

    title_candidates = find_title_candidates(search_text, parent_title)
    title_candidates = [p + min_pos for p in title_candidates]

    phrase_pos = find_phrase_position(search_text, parent_phrase)
    phrase_pos = phrase_pos + min_pos if phrase_pos != -1 else -1

    if phrase_pos != -1:
        before_phrase = [
            p for p in title_candidates
            if p <= phrase_pos and phrase_pos - p <= lookback
        ]

        if before_phrase:
            return max(before_phrase)  # closest title before phrase

    if title_candidates:
        return title_candidates[0]

    return -1


def validate_chunk_starts(rows, max_preview_chars=250):
    suspicious = []

    for row in rows:
        title = row.get("title", "")
        text = (row.get("text") or "").strip()

        title_key = normalize_for_match(title)
        preview_key = normalize_for_match(text[:max_preview_chars])

        if title_key and title_key not in preview_key:
            suspicious.append({
                "chunk_id": row.get("chunk_id"),
                "title": title,
                "start_index": row.get("start_index"),
                "preview": text[:180].replace("\n", " "),
            })

    return suspicious

def strip_heading_numbering(line):
    """
    Remove common heading numbering:
    1
    1.
    1.2
    1.2.3.
    7.4.1
    """
    return re.sub(
        r"^\s*\d+(\.\d+)*\.?\s*",
        "",
        line or ""
    ).strip()


def get_line_span_at_pos(text, pos):
    """
    Return start, end, and text of the line containing pos.
    """
    if pos < 0:
        return -1, -1, ""

    line_start = text.rfind("\n", 0, pos) + 1
    line_end = text.find("\n", pos)

    if line_end == -1:
        line_end = len(text)

    return line_start, line_end, text[line_start:line_end]


def title_words(text):
    return re.findall(r"[A-Za-z0-9]+", text or "")


def title_matches_as_words(line, title):
    """
    Whole-word-ish title match.

    For short titles, use regex.
    For very long titles, avoid huge regex patterns because they can become very slow.
    """
    words = title_words(title)
    if not words:
        return False

    cleaned = strip_heading_numbering(line)

    # Fast path for very long titles.
    # Long risk-factor titles make huge regex patterns and can hang.
    if len(words) > 18 or len(title) > 180:
        line_key = normalize_for_match(cleaned)
        title_key = normalize_for_match(title)

        if not line_key or not title_key:
            return False

        return (
            line_key == title_key
            or line_key.startswith(title_key)
            or title_key.startswith(line_key)
        )

    pattern = r"\b" + r"\s*[-–—:./]?\s*".join(
        re.escape(w) for w in words
    ) + r"\b"

    return re.search(pattern, cleaned, flags=re.IGNORECASE) is not None


def line_heading_score(line, title):
    """
    Score how much a line looks like a standalone heading for title.

    This avoids hardcoded generic title lists.
    """
    cleaned = strip_heading_numbering(line).strip()

    if not cleaned or not title:
        return 0

    title_key = normalize_for_match(title)
    line_key = normalize_for_match(cleaned)

    if not title_key or not line_key:
        return 0

    if not title_matches_as_words(cleaned, title):
        return 0

    score = 0

    # Exact standalone heading.
    if line_key == title_key:
        score += 100

    # Heading line starts with the title.
    elif line_key.startswith(title_key):
        score += 75

        remainder = cleaned[len(title):].strip(" \t\r\n:-–—.")
        remainder_words = title_words(remainder)

        # If the line continues as a sentence/paragraph, penalize.
        if len(remainder_words) > 8:
            score -= 50

        if len(cleaned) > max(len(title) * 4, len(title) + 160):
            score -= 50

    else:
        # Title appears inside the line, but not at the beginning.
        # Usually body text, so this is weak.
        score += 20

    # Long lines are usually paragraphs, not headings.
    if len(cleaned) > max(len(title) * 5, len(title) + 220):
        score -= 40

    # Sentence punctuation often means body text.
    if cleaned.endswith((".", ",", ";")):
        score -= 15

    return max(score, 0)


def find_title_heading_candidates(text, title):
    """
    Find likely heading candidates using line structure.

    This is stricter than normalized occurrence matching, but not so strict
    that it misses headings split over several PDF-extracted lines.
    """
    if not text or not title:
        return []

    candidates = []
    lines = text.splitlines()
    char_pos = 0

    for i in range(len(lines)):
        combined = ""

        for j in range(i, min(i + 4, len(lines))):
            part = lines[j].strip()

            if not part:
                break

            combined = f"{combined} {part}".strip()

            score = line_heading_score(combined, title)

            if score >= 75:
                line_start_offset = max(lines[i].find(lines[i].strip()), 0)
                candidates.append(char_pos + line_start_offset)
                break

        char_pos += len(lines[i]) + 1

    return sorted(set(candidates))


def find_title_candidates(text, title):
    """
    Backwards-compatible candidate finder.

    Returns likely title positions. This version prefers heading-like candidates,
    but also includes normalized occurrences as lower-confidence candidates.
    """
    if not text or not title:
        return []

    candidates = set()

    for pos in find_title_heading_candidates(text, title):
        candidates.add(pos)

    for pos in find_normalized_occurrences(text, title):
        candidates.add(pos)

    return sorted(candidates)

def collect_title_candidates(text, title):
    """
    Collect both strict heading candidates and broader normalized candidates.

    For very long titles, avoid broad normalized occurrence scanning because it is slow.
    """
    candidates = {}

    for pos in find_title_heading_candidates(text, title):
        candidates[pos] = {
            "pos": pos,
            "source": "strict_heading",
        }

    title_word_count = len(title_words(title))
    title_key = normalize_for_match(title)

    # Long titles are expensive and usually should be found as headings,
    # not by broad normalized occurrence search.
    if title_word_count > 18 or len(title_key) > 180:
        return list(candidates.values())

    for pos in find_normalized_occurrences(text, title):
        if pos not in candidates:
            candidates[pos] = {
                "pos": pos,
                "source": "normalized_occurrence",
            }

    return list(candidates.values())


def score_title_candidate(text, candidate_pos, title, phrase_pos=-1, window=1500):
    """
    Score a possible title boundary.

    Strongest signal:
        title heading shortly before start_phrase

    Also supports headings glued to previous text, but only when they
    are anchored by the start_phrase.
    """
    _, _, line = get_line_span_at_pos(text, candidate_pos)

    score = line_heading_score(line, title)

    # If the whole line does not look like a heading, the title may still
    # be glued to the end of the previous paragraph/footer.
    if score <= 0:
        if title_matches_at_position(text, candidate_pos, title):
            score = 35
        else:
            return 0

    if phrase_pos != -1:
        distance = abs(phrase_pos - candidate_pos)

        if candidate_pos <= phrase_pos:
            if distance <= window:
                # Best case: title before phrase.
                score += 90 - min(70, distance / 20)
            else:
                score -= 50
        else:
            if distance <= 250:
                # Sometimes extraction puts phrase before title,
                # but only trust this if very close.
                score += 30 - min(25, distance / 20)
            else:
                score -= 40

    return max(score, 0)

def normalize_heading_compare(text):
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def remove_next_heading_from_previous_rows(rows, model=None, max_tail_chars=1200, max_after_normalized_chars=40):
    """
    Remove a next chunk heading if it is accidentally glued to the end
    of the previous chunk.

    Example:
        row[i].text ends with:
            "... Integrated Annual Report 2024.
             Climate risk identification, monitoring and reporting"

        row[i+1].title is:
            "Climate risk identification, monitoring and reporting - Introduction"

    Then the heading is removed from row[i].text.
    """

    for i in range(len(rows) - 1):
        current = rows[i]
        next_row = rows[i + 1]

        current_text = current.get("text") or ""
        next_title = next_row.get("title") or ""

        if not current_text.strip() or not next_title.strip():
            continue

        # Intro rows have titles like:
        # "Climate risk identification, monitoring and reporting - Introduction"
        next_heading = re.sub(
            r"\s*-\s*Introduction\s*$",
            "",
            next_title,
            flags=re.IGNORECASE
        ).strip()

        if not next_heading:
            continue

        heading_key = normalize_for_match(next_heading)

        # Too short = unsafe.
        if len(heading_key) < 8:
            continue

        tail_start = max(0, len(current_text) - max_tail_chars)
        tail = current_text[tail_start:]

        tail_key, tail_map = normalize_with_map(tail)

        if not tail_key or not tail_map:
            continue

        # Find the LAST occurrence of next heading in the tail.
        pos_key = tail_key.rfind(heading_key)

        if pos_key == -1:
            continue

        after_key = tail_key[pos_key + len(heading_key):]

        # Only remove when the heading is basically at the end.
        # This avoids deleting legitimate mentions inside the content.
        if len(after_key) > max_after_normalized_chars:
            continue

        original_cut_pos = tail_start + tail_map[pos_key]

        current["text"] = current_text[:original_cut_pos].rstrip()

        if model is not None:
            current["token_count"] = count_tokens(current["text"], model=model)
        else:
            current["token_count"] = count_tokens(current["text"])

    return rows

def find_glued_heading_after_header(text, title, min_pos=0, max_prefix_chars=180):
    """
    Find headings glued to a running header/page label.

    Example:
        2024 ANNUAL REPORT - Sustainability Reporting 193Policies

    This should match:
        Policies

    But avoid matching normal prose:
        ... our policies include ...
    """
    if not text or not title:
        return -1

    min_pos = max(0, min_pos)

    title_key = normalize_for_match(title)
    if not title_key:
        return -1

    text_key, text_map = normalize_with_map(text[min_pos:])
    if not text_key or not text_map:
        return -1

    search_from = 0

    while True:
        pos_key = text_key.find(title_key, search_from)

        if pos_key == -1:
            return -1

        original_pos = min_pos + text_map[pos_key]

        line_start, line_end, line = get_line_span_at_pos(text, original_pos)
        before_on_line = text[line_start:original_pos].strip()
        after_on_line = text[original_pos:line_end].strip()

        # Must start exactly at this original position.
        if not title_matches_at_position(text, original_pos, title):
            search_from = pos_key + 1
            continue

        # Accept if the prefix looks like page header / page number noise.
        before_key = normalize_for_match(before_on_line)

        has_page_number = bool(re.search(r"\d{1,5}\s*$", before_on_line))
        short_prefix = len(before_on_line) <= max_prefix_chars

        # Avoid accepting ordinary prose before the title.
        prefix_has_sentence_punctuation = bool(re.search(r"[.;:!?]\s", before_on_line))

        # Heading itself should not continue as a long paragraph on same line.
        title_words_count = len(title_words(title))
        after_words = title_words(after_on_line)

        after_is_reasonable = (
            len(after_words) <= max(title_words_count + 6, 10)
            or normalize_for_match(after_on_line).startswith(title_key)
        )

        if short_prefix and has_page_number and not prefix_has_sentence_punctuation and after_is_reasonable:
            return original_pos

        search_from = pos_key + 1

def choose_boundary_from_title_and_phrase(
    title_candidates,
    phrase_pos,
    before_window=1500,
    after_window=250,
    min_title_score=45,
    require_phrase_agreement=False,
):
    """
    Boundary decision logic.

    If require_phrase_agreement=True, only accept title candidates that are
    close to the start_phrase. This is used on pages before the expected
    next start page to avoid early false cutoffs.
    """
    good_titles = [
        c for c in title_candidates
        if c.get("score", 0) >= min_title_score
    ]

    if phrase_pos != -1 and good_titles:
        before_phrase = [
            c for c in good_titles
            if c["pos"] <= phrase_pos
            and phrase_pos - c["pos"] <= before_window
        ]

        if before_phrase:
            return max(before_phrase, key=lambda c: c["pos"])["pos"]

        after_phrase = [
            c for c in good_titles
            if c["pos"] > phrase_pos
            and c["pos"] - phrase_pos <= after_window
        ]

        if after_phrase:
            return min(after_phrase, key=lambda c: c["pos"])["pos"]

    # On early pages, do not accept standalone title-only matches.
    # This prevents cutting Roles and responsibilities on page 20/21
    # when Environment is expected on page 22.
    if require_phrase_agreement:
        return -1

    if good_titles:
        return sorted(good_titles, key=lambda c: (-c["score"], c["pos"]))[0]["pos"]

    if phrase_pos != -1:
        return phrase_pos

    return -1

def find_intro_start_in_full_text(text, intro_node, min_pos=0):
    """
    Find where an intro node starts.

    Intro node represents the parent section's own text, so its start should
    be found using the parent heading/title/start_phrase, not the generated
    '- Introduction' title.
    """
    if not text or not intro_node:
        return -1

    source_heading = intro_node.get("source_heading") or intro_node.get("heading") or ""
    source_title = intro_node.get("source_title") or ""
    start_phrase = intro_node.get("start_phrase") or ""

    heading_pos = find_heading_position(
        text=text,
        heading=source_heading,
        start_phrase=start_phrase,
        min_pos=min_pos,
        window=3000,
    )

    if heading_pos != -1:
        return heading_pos

    if source_title:
        candidates = collect_title_candidates(text[min_pos:], source_title)
        positions = [
            c["pos"] + min_pos
            for c in candidates
            if c["pos"] + min_pos >= min_pos
        ]

        if positions:
            return min(positions)

    if start_phrase:
        local_phrase_pos = find_phrase_position(text[min_pos:], start_phrase)
        if local_phrase_pos != -1:
            return min_pos + local_phrase_pos

    return -1

def find_intro_child_boundary_in_full_text(text, intro_node, child_node, start_pos):
    """
    Find where an intro node should stop.

    For a parent intro, the boundary is the parent's first real child.
    This is more reliable than normal next-leaf logic because the first child
    may itself be a parent and not a leaf.

    Strategy:
    1. Try full child heading.
    2. Try plain child title.
    3. Try child start_phrase and look backwards for the child title.
    4. If parent and child start_phrase are the same, do not use the phrase
       as a boundary.
    """
    if not text or not child_node:
        return -1

    min_pos = max(0, start_pos + 1)

    child_heading = get_node_heading_for_match(child_node)
    child_title = (
        child_node.get("source_title")
        or child_node.get("title")
        or ""
    )
    child_phrase = child_node.get("start_phrase", "")
    parent_phrase = intro_node.get("start_phrase", "")

    # 1. Full numbered heading, e.g. "5.1 Own workforce"
    heading_pos = find_heading_position(
        text=text,
        heading=child_heading,
        start_phrase=child_phrase,
        min_pos=min_pos,
        window=3000,
    )

    if heading_pos != -1 and heading_pos > start_pos:
        return heading_pos

    # 2. Plain title heading, e.g. "Own workforce"
    title_candidates = find_title_heading_candidates(
        text[min_pos:],
        child_title,
    )

    if title_candidates:
        title_candidates = [p + min_pos for p in title_candidates]
        title_candidates = [p for p in title_candidates if p > start_pos]

        if title_candidates:
            return min(title_candidates)

    # 3. Use child start_phrase, but prefer the title directly before it.
    phrase_pos = -1
    if child_phrase:
        search_text = text[min_pos:]
        local_phrase_pos = find_phrase_position(search_text, child_phrase)

        if local_phrase_pos != -1:
            phrase_pos = min_pos + local_phrase_pos

    if phrase_pos != -1 and phrase_pos > start_pos:
        # Try to move boundary back from body sentence to child title.
        title_before_phrase = find_specific_title_before_phrase(
            text=text,
            title=child_title,
            phrase_pos=phrase_pos,
            lookback=900,
        )

        if title_before_phrase != -1 and title_before_phrase > start_pos:
            return title_before_phrase

        # Only use the phrase itself if it is not the same as the parent phrase.
        if not phrases_match(parent_phrase, child_phrase):
            return phrase_pos

    return -1

def find_next_boundary(
    text,
    next_node,
    min_pos=50,
    window=1500,
    require_phrase_agreement=False,
):
    heading = get_node_heading_for_match(next_node)
    fallback_title = (
        next_node.get("source_title")
        or next_node.get("title")
        or ""
    )

    if next_node.get("is_intro_node"):
        fallback_title = (
            next_node.get("source_title")
            or re.sub(r"\s*-\s*Introduction\s*$", "", next_node.get("title", ""), flags=re.I)
        )
    next_phrase = next_node.get("start_phrase", "")

    search_text = text[min_pos:]

    heading_pos = find_heading_position(
        text=text,
        heading=heading,
        start_phrase=next_phrase,
        min_pos=min_pos,
        window=2500,
    )

    if heading_pos != -1:
        return heading_pos

    phrase_pos = find_phrase_position(search_text, next_phrase)
    phrase_pos = phrase_pos + min_pos if phrase_pos != -1 else -1

    if accept_heading_match(
        text=text,
        heading=heading,
        heading_pos=heading_pos,
        phrase_pos=phrase_pos,
        window=window,
    ):
        return heading_pos

    # 2. Old fallback logic: title + start_phrase agreement.
    raw_candidates = collect_title_candidates(search_text, fallback_title)

    scored = []

    for cand in raw_candidates:
        absolute_pos = cand["pos"] + min_pos

        score = score_title_candidate(
            text=text,
            candidate_pos=absolute_pos,
            title=fallback_title,
            phrase_pos=phrase_pos,
            window=window,
        )

        if cand["source"] == "strict_heading":
            score += 20

        if score > 0:
            scored.append({
                "pos": absolute_pos,
                "score": score,
                "source": cand["source"],
            })

    min_score = 45 if not next_node.get("is_llm_split_node") else 20

    return choose_boundary_from_title_and_phrase(
        title_candidates=scored,
        phrase_pos=phrase_pos,
        before_window=window,
        after_window=250,
        min_title_score=min_score,
        require_phrase_agreement=require_phrase_agreement,
    )

def get_node_heading_for_match(node):
    """
    Prefer full numbered heading for matching.
    Example: "2.b Impact, risk and opportunity management"

    Fallbacks:
    - source_heading
    - structure + title
    - source_title/title
    """
    if not node:
        return ""

    heading = (node.get("heading") or node.get("source_heading") or "").strip()
    if heading:
        return heading

    title = (node.get("source_title") or node.get("title") or "").strip()
    structure = (node.get("structure") or "").strip()

    if structure and title:
        return f"{structure} {title}".strip()

    return title


def is_numbered_heading_text(heading):
    """
    True for headings like:
    1 General disclosures
    1.a Governance
    2.b Impact, risk and opportunity management
    2.b.1 Climate change mitigation policies
    """
    return bool(
        re.match(
            r"^\s*\d+(?:\.[A-Za-z0-9]+)*\.?\s+",
            heading or ""
        )
    )


def looks_like_header_prefix(prefix):
    """
    General check: text before a heading on the same line may be a page number
    or running header. Reject prose references like:
        "... presented in section 2.a Strategy"
    """
    if not prefix:
        return True

    prefix = prefix.strip()

    if not prefix:
        return True

    # Page number only
    if re.fullmatch(r"\d{1,5}", prefix):
        return True

    # "Page 123"
    if re.fullmatch(r"page\s+\d{1,5}", prefix, flags=re.I):
        return True

    # Header-like prefix: short, no sentence punctuation, often uppercase/title-like.
    if len(prefix) <= 140 and not re.search(r"[.;:!?]\s", prefix):
        words = re.findall(r"[A-Za-z]+", prefix)

        if not words:
            return True

        lowercase_words = sum(1 for w in words if w.islower())

        # Running headers usually are mostly uppercase/title case,
        # not normal prose.
        if lowercase_words <= max(1, len(words) // 4):
            return True

    return False


def find_heading_position(text, heading, start_phrase=None, min_pos=0, window=3000, lookback=250):
    """
    General heading finder.

    It finds heading occurrences in normalized text, then rejects occurrences
    that are clearly inside prose/cross-references.

    It allows a small lookback before min_pos because PDF text sometimes places
    page headers/section headings slightly before the computed min_pos.
    """
    if not text or not heading:
        return -1

    if min_pos < 0:
        min_pos = 0

    effective_min = max(0, min_pos - lookback)

    heading_key = normalize_for_match(heading)
    phrase_key = normalize_for_match(start_phrase) if start_phrase else ""

    if not heading_key:
        return -1

    search_text = text[effective_min:]
    text_key, text_map = normalize_with_map(search_text)

    if not text_key or not text_map:
        return -1

    candidates = []

    pos_key = text_key.find(heading_key)

    while pos_key != -1:
        original_pos = effective_min + text_map[pos_key]

        # If match is far before min_pos, skip it.
        # Small lookback is allowed; big backwards jumps are not.
        if original_pos < min_pos and (min_pos - original_pos) > lookback:
            pos_key = text_key.find(heading_key, pos_key + 1)
            continue

        line_start, line_end, line = get_line_span_at_pos(text, original_pos)
        before_on_line = text[line_start:original_pos].strip()

        # Main safety check: reject prose references.
        if not looks_like_header_prefix(before_on_line):
            pos_key = text_key.find(heading_key, pos_key + 1)
            continue

        score = 50

        # Prefer heading + start_phrase if start_phrase exists.
        if phrase_key:
            after_text = text[original_pos:original_pos + window]
            after_key = normalize_for_match(after_text)

            if phrase_key in after_key:
                score += 100

        candidates.append({
            "pos": original_pos,
            "score": score,
            "before_on_line": before_on_line,
        })

        pos_key = text_key.find(heading_key, pos_key + 1)

    if not candidates:
        return -1

    candidates.sort(key=lambda x: (-x["score"], x["pos"]))
    return candidates[0]["pos"]


def accept_heading_match(text, heading, heading_pos, phrase_pos, window=1500):
    """
    Decide whether a heading match is safe.

    Numbered headings are trusted.
    Unnumbered headings are trusted only if:
    - they look like a standalone heading line, or
    - they are close to the start_phrase.
    """
    if heading_pos == -1:
        return False

    if is_numbered_heading_text(heading):
        return True

    _, _, line = get_line_span_at_pos(text, heading_pos)
    heading_score = line_heading_score(line, heading)

    close_to_phrase = (
        phrase_pos != -1
        and abs(phrase_pos - heading_pos) <= window
    )

    return heading_score >= 45 or close_to_phrase

def phrases_match(a, b):
    """
    True when two start phrases are effectively the same.
    Uses your normalized matching so punctuation/spacing does not matter.
    """
    a_key = normalize_for_match(a)
    b_key = normalize_for_match(b)

    if not a_key or not b_key:
        return False

    return a_key == b_key

def find_node_start(text, node, window=1500, min_pos=0):
    heading = get_node_heading_for_match(node)

    fallback_title = (
        node.get("source_title")
        or node.get("title")
        or ""
    )

    if node.get("is_intro_node"):
        fallback_title = (
            node.get("source_title")
            or re.sub(r"\s*-\s*Introduction\s*$", "", node.get("title", ""), flags=re.I)
        )

    start_phrase = node.get("start_phrase", "")

    search_text = text[min_pos:]

    heading_pos = find_heading_position(
        text=text,
        heading=heading,
        start_phrase=start_phrase,
        min_pos=min_pos,
        window=2500,
    )

    heading_key = normalize_for_match(heading)
    title_key = normalize_for_match(fallback_title)

    if (
        heading_pos != -1
        and heading_key
        and title_key
        and heading_key != title_key
    ):
        return heading_pos

    phrase_pos = find_phrase_position(search_text, start_phrase)
    phrase_pos = phrase_pos + min_pos if phrase_pos != -1 else -1

    raw_candidates = collect_title_candidates(search_text, fallback_title)

    scored = []

    for cand in raw_candidates:
        absolute_pos = cand["pos"] + min_pos

        score = score_title_candidate(
            text=text,
            candidate_pos=absolute_pos,
            title=fallback_title,
            phrase_pos=phrase_pos,
            window=window,
        )

        if cand["source"] == "strict_heading":
            score += 20

        if score > 0:
            scored.append({
                "pos": absolute_pos,
                "score": score,
                "source": cand["source"],
            })

    min_score = 45 if not node.get("is_llm_split_node") else 20

    return choose_boundary_from_title_and_phrase(
        title_candidates=scored,
        phrase_pos=phrase_pos,
        before_window=window,
        after_window=250,
        min_title_score=min_score,
    )


def find_node_start_after(text, node, min_pos=50, window=1500):
    return find_node_start(
        text=text,
        node=node,
        window=window,
        min_pos=min_pos
    )

def same_or_overlapping_phrase(a, b):
    a_key = normalize_for_match(a)
    b_key = normalize_for_match(b)

    if not a_key or not b_key:
        return False

    return a_key == b_key or a_key in b_key or b_key in a_key

def find_intro_end_boundary(text, intro_node, first_child, start_pos):
    """
    Find where an intro node should stop.

    For intro nodes, do NOT blindly fall back to the child start_phrase
    if that phrase is the same as the parent start_phrase. In that case,
    there is no separate intro text.
    """
    if not text or not first_child:
        return -1

    parent_phrase = intro_node.get("start_phrase", "")
    child_phrase = first_child.get("start_phrase", "")

    child_heading = get_node_heading_for_match(first_child)

    # 1. Prefer the child heading.
    heading_pos = find_heading_position(
        text=text,
        heading=child_heading,
        start_phrase=child_phrase,
        min_pos=max(0, start_pos + 1),
        window=3000,
    )

    if heading_pos != -1 and heading_pos > start_pos:
        return heading_pos

    # 2. Only use child start_phrase if it is not the same as the intro start phrase.
    if child_phrase and not same_or_overlapping_phrase(parent_phrase, child_phrase):
        phrase_pos = find_phrase_position(text[max(0, start_pos + 1):], child_phrase)

        if phrase_pos != -1:
            return max(0, start_pos + 1) + phrase_pos

    # 3. If child phrase equals parent phrase and child heading was not found,
    # we cannot separate intro from first child.
    return -1

def estimate_title_end_pos(text, title, start_pos):
    """
    Estimate where the matched title/heading ends.

    Uses normalized matching so headings like:
    1.b Strategy, business model and stakeholders

    still work when PDF spacing is weird.
    """
    if start_pos == -1 or not text or not title:
        return start_pos

    window = text[start_pos:start_pos + max(len(title) * 6, 600)]

    title_key = normalize_for_match(title)
    window_key, window_map = normalize_with_map(window)

    if not title_key or not window_key or not window_map:
        return start_pos + len(title)

    pos_key = window_key.find(title_key)

    if pos_key == -1:
        return start_pos + len(title)

    end_key = pos_key + len(title_key)

    if end_key >= len(window_map):
        return start_pos + len(window)

    return start_pos + window_map[end_key]


def find_intro_child_boundary(text, first_child, min_pos=0, window=1500):
    child_heading = get_node_heading_for_match(first_child)
    fallback_child_title = first_child.get("source_title") or first_child.get("title", "")
    child_phrase = first_child.get("start_phrase", "")

    search_text = text[min_pos:]

    phrase_pos = find_phrase_position(search_text, child_phrase)
    phrase_pos = phrase_pos + min_pos if phrase_pos != -1 else -1

    heading_pos = find_heading_position(
        text=text,
        heading=child_heading,
        min_pos=min_pos,
    )

    if accept_heading_match(
        text=text,
        heading=child_heading,
        heading_pos=heading_pos,
        phrase_pos=phrase_pos,
        window=window,
    ):
        return heading_pos

    raw_candidates = collect_title_candidates(search_text, fallback_child_title)

    scored = []

    for cand in raw_candidates:
        absolute_pos = cand["pos"] + min_pos

        score = score_title_candidate(
            text=text,
            candidate_pos=absolute_pos,
            title=fallback_child_title,
            phrase_pos=phrase_pos,
            window=window,
        )

        if cand["source"] == "strict_heading":
            score += 20

        if score > 0:
            scored.append({
                "pos": absolute_pos,
                "score": score,
                "source": cand["source"],
            })

    return choose_boundary_from_title_and_phrase(
        title_candidates=scored,
        phrase_pos=phrase_pos,
        before_window=window,
        after_window=250,
        min_title_score=45,
    )

def title_matches_at_position(text, pos, title):
    """
    Check whether title starts at exactly this text position.

    This catches headings glued to the previous paragraph/footer, e.g.
    '... Annual Report 2024. Climate risk identification, monitoring and reporting'
    """
    if pos < 0 or not text or not title:
        return False

    words = title_words(title)
    if not words:
        return False

    window = text[pos:pos + max(len(title) * 4, 300)]

    pattern = r"^\s*(?:\d+(\.\d+)*\.?\s*)?" + r"\s*[-–—:./]?\s*".join(
        re.escape(w) for w in words
    ) + r"\b"

    return re.search(pattern, window, flags=re.IGNORECASE) is not None

def build_candidate_page_order(start_page, next_start_page, page_count):
    """
    Search the expected next node page first.

    Old behavior searched next_start_page - 2 first, which can cause early
    false matches on the previous section pages.
    """
    pages = []

    def add_page(p):
        if p is None:
            return
        if p < 1 or p > page_count:
            return
        if p < start_page:
            return
        if p not in pages:
            pages.append(p)

    # 1. Expected page first.
    add_page(next_start_page)

    # 2. One page before, because headings can appear at the end of previous page.
    add_page(next_start_page - 1)

    # 3. One page after, because start_index can be slightly late.
    add_page(next_start_page + 1)

    # 4. Two pages before/after only as last resort.
    add_page(next_start_page - 2)
    add_page(next_start_page + 2)

    return pages

def build_boundary_page_order(start_page, expected_page, page_count):
    """
    Search expected page first, then one page before, then one page after.
    Do not search before the current section start page.
    """
    pages = []

    def add(p):
        if p is None:
            return
        if p < 1 or p > page_count:
            return
        if p < start_page:
            return
        if p not in pages:
            pages.append(p)

    add(expected_page)
    add(expected_page - 1)
    add(expected_page + 1)

    return pages

def coerce_pdf_page_to_text(page):
    """
    Convert one pdf_pages item into clean page text.

    Handles:
    - plain string
    - dict with text
    - tuple/list where first item is the page text
    """
    if page is None:
        return ""

    if isinstance(page, dict):
        return page.get("text", "") or ""

    if isinstance(page, (tuple, list)):
        if not page:
            return ""

        first = page[0]

        if isinstance(first, dict):
            return first.get("text", "") or ""

        if isinstance(first, str):
            return first

        return str(first or "")

    return str(page or "")

def get_pdf_page_text(pdf_pages, page_number):
    """
    Return clean text for one PDF page.

    Supports:
    - dict: pdf_pages[page_number] or pdf_pages[page_number - 1]
    - list: pdf_pages[page_number - 1] fallback pdf_pages[page_number]
    - tuple/list page objects
    """
    if page_number is None:
        return ""

    page = None

    if isinstance(pdf_pages, dict):
        page = pdf_pages.get(page_number)

        if page is None:
            page = pdf_pages.get(page_number - 1)

    elif isinstance(pdf_pages, list):
        index = page_number - 1

        if 0 <= index < len(pdf_pages):
            page = pdf_pages[index]
        elif 0 <= page_number < len(pdf_pages):
            page = pdf_pages[page_number]

    return coerce_pdf_page_to_text(page)

def get_pdf_page_text_by_index(pdf_pages, page_index):
    """
    Return clean page text by actual list index / dict key.
    """
    if page_index is None:
        return ""

    page = None

    if isinstance(pdf_pages, list):
        if 0 <= page_index < len(pdf_pages):
            page = pdf_pages[page_index]

    elif isinstance(pdf_pages, dict):
        page = pdf_pages.get(page_index)

    return coerce_pdf_page_to_text(page)

def get_pdf_page_text_variants(pdf_pages, page_number):
    """
    Return possible clean text variants for a logical page number.

    Handles both:
    - 1-based logical pages: pdf_pages[page_number - 1]
    - 0-based/physical indexes: pdf_pages[page_number]
    - nearby fallback: pdf_pages[page_number + 1]

    Each returned item:
    {
        "page_number": logical page number,
        "page_index": actual list index or dict key,
        "text": clean page text
    }
    """
    variants = []

    if page_number is None:
        return variants

    if isinstance(pdf_pages, dict):
        candidate_keys = [
            page_number,
            page_number - 1,
            page_number + 1,
        ]

        for key in candidate_keys:
            if key not in pdf_pages:
                continue

            text = coerce_pdf_page_to_text(pdf_pages.get(key))

            if text:
                variants.append({
                    "page_number": page_number,
                    "page_index": key,
                    "text": text,
                })

    elif isinstance(pdf_pages, list):
        candidate_indexes = [
            page_number - 1,
            page_number,
            page_number + 1,
        ]

        for index in candidate_indexes:
            if index < 0 or index >= len(pdf_pages):
                continue

            text = coerce_pdf_page_to_text(pdf_pages[index])

            if text:
                variants.append({
                    "page_number": page_number,
                    "page_index": index,
                    "text": text,
                })

    # Deduplicate identical page text.
    deduped = []
    seen = set()

    for variant in variants:
        text_key = normalize_for_match(variant.get("text", "")[:1000])

        if text_key in seen:
            continue

        seen.add(text_key)
        deduped.append(variant)

    return deduped

def slice_pages_text_with_index_offset(
    pdf_pages,
    start_page,
    start_pos,
    end_page,
    end_pos,
    page_index_offset,
):
    """
    Slice pages using the same list-index offset that found the start.

    page_index_offset:
    - -1 means pdf_pages[page_number - 1]
    -  0 means pdf_pages[page_number]
    -  1 means pdf_pages[page_number + 1]
    """
    parts = []

    for page_number in range(start_page, end_page + 1):
        if isinstance(pdf_pages, list):
            page_index = page_number + page_index_offset

            if page_index < 0 or page_index >= len(pdf_pages):
                page_text = ""
            else:
                page = pdf_pages[page_index]

                if isinstance(page, dict):
                    page_text = page.get("text", "") or ""
                else:
                    page_text = str(page or "")

        else:
            page_text = get_pdf_page_text(pdf_pages, page_number)

        if page_number == start_page and page_number == end_page:
            parts.append(page_text[start_pos:end_pos])
        elif page_number == start_page:
            parts.append(page_text[start_pos:])
        elif page_number == end_page:
            parts.append(page_text[:end_pos])
        else:
            parts.append(page_text)

    return "\n".join(parts).strip()

def intro_is_only_heading_or_title(text, parent_node=None, first_child=None):
    if not text or not text.strip():
        return True

    check_text = text.strip()

    candidates = []

    if parent_node:
        candidates.extend([
            get_node_heading_for_match(parent_node),
            parent_node.get("title", ""),
        ])

    if first_child:
        candidates.extend([
            get_node_heading_for_match(first_child),
            first_child.get("title", ""),
        ])

    text_key = normalize_for_match(check_text)

    for candidate in candidates:
        candidate_key = normalize_for_match(candidate)

        if candidate_key and text_key == candidate_key:
            return True

    return False

def find_title_or_heading_before_phrase_on_page(
    page_text,
    title,
    heading,
    phrase_pos,
    min_pos=0,
    lookback=2500,
):
    """
    Find the nearest occurrence of the node heading/title before its start phrase.

    This prevents matching an earlier occurrence of a generic word like 'Policies'
    at the top of the page.
    """
    if not page_text or phrase_pos == -1:
        return -1

    search_start = max(min_pos, phrase_pos - lookback)
    search_text = page_text[search_start:phrase_pos]

    candidates = []

    for label in [heading, title]:
        if not label:
            continue

        title_candidates = collect_title_candidates(search_text, label)

        for candidate in title_candidates:
            local_pos = candidate.get("pos", -1)

            if local_pos == -1:
                continue

            absolute_pos = search_start + local_pos

            if absolute_pos < min_pos:
                continue

            if absolute_pos >= phrase_pos:
                continue

            candidates.append(absolute_pos)

    if candidates:
        # Important: use nearest title before phrase, not first title on page.
        return max(candidates)

    return -1

def find_intro_start_on_page(page_text, intro_node, min_pos=0):
    """
    Find the start of an intro node on one page.

    General rule:
    - Prefer the intro's own start_phrase.
    - If the title appears shortly before that phrase, start at the title.
    - Do not match a title occurrence that appears earlier than min_pos.
    - Do not trust title-only matches before trying the phrase.
    """
    if not page_text or not intro_node:
        return -1

    source_heading = intro_node.get("source_heading") or intro_node.get("heading") or ""
    source_title = intro_node.get("source_title") or ""
    start_phrase = intro_node.get("start_phrase") or ""

    min_pos = max(0, min_pos)

    # 1. First find the real body anchor.
    phrase_pos = -1
    if start_phrase:
        local_phrase_pos = find_phrase_position(page_text[min_pos:], start_phrase)

        if local_phrase_pos != -1:
            phrase_pos = min_pos + local_phrase_pos

    # 2. If phrase is found, try to move back to the nearest title/heading before it.
    if phrase_pos != -1:
        title_before_phrase = find_title_or_heading_before_phrase_on_page(
            page_text=page_text,
            title=source_title,
            heading=source_heading,
            phrase_pos=phrase_pos,
            min_pos=min_pos,
            lookback=2500,
        )

        if title_before_phrase != -1:
            return title_before_phrase

        # If no reliable title is found, start at the phrase.
        return phrase_pos

    # 3. Only if phrase cannot be found, try heading/title as fallback.
    # This fallback is weaker, so it must respect min_pos.
    if source_heading:
        heading_pos = find_heading_position(
            text=page_text,
            heading=source_heading,
            start_phrase=start_phrase,
            min_pos=min_pos,
            window=3000,
        )

        if heading_pos != -1:
            return heading_pos

    if source_title:
        candidates = collect_title_candidates(page_text[min_pos:], source_title)

        positions = [
            candidate["pos"] + min_pos
            for candidate in candidates
            if candidate.get("pos", -1) >= 0
        ]

        if positions:
            return min(positions)

    return -1

def find_intro_start_in_combined_text(pdf_pages, intro_node, start_page, end_page):
    """
    Fallback start finder for intro nodes.

    Uses combined text across the intro page range when page-by-page matching fails.
    This helps when headings/start phrases are split, wrapped, or page extraction differs.
    """
    full_text = get_text_of_pdf_pages(pdf_pages, start_page, end_page)

    if not full_text:
        return "", -1

    start_pos = find_intro_start_on_page(
        page_text=full_text,
        intro_node=intro_node,
        min_pos=0,
    )

    return full_text, start_pos

def find_intro_child_boundary_on_page(page_text, intro_node, child_node, min_pos=0):
    """
    Find where intro text should stop on a single page.

    For intro nodes, the safest signal is usually:
    child start_phrase -> nearest child title before that phrase.

    This fixes cases like:
        2024 ANNUAL REPORT - Sustainability Reporting 193Policies
        The Group’s organisation model ...

    where the correct boundary is 'Policies' at pos 49,
    not the later table heading 'Policies Description...' at pos 1021.
    """
    if not page_text or not child_node:
        return -1

    min_pos = max(0, min_pos)

    child_heading = get_node_heading_for_match(child_node)
    child_title = child_node.get("source_title") or child_node.get("title") or ""
    child_phrase = child_node.get("start_phrase") or ""
    parent_phrase = intro_node.get("start_phrase") or ""

    # 1. Best path: find child phrase, then move back to nearest title before it.
    if child_phrase and not phrases_match(parent_phrase, child_phrase):
        local_phrase_pos = find_phrase_position(page_text[min_pos:], child_phrase)

        if local_phrase_pos != -1:
            phrase_pos = min_pos + local_phrase_pos

            title_before_phrase = find_loose_title_before_phrase(
                text=page_text,
                title=child_title,
                phrase_pos=phrase_pos,
                min_pos=min_pos,
                lookback=1500,
            )

            if title_before_phrase != -1:
                return title_before_phrase

            return phrase_pos

    # 2. Then try full heading.
    heading_pos = find_heading_position(
        text=page_text,
        heading=child_heading,
        start_phrase=child_phrase,
        min_pos=min_pos,
        window=3000,
    )

    if heading_pos != -1:
        return heading_pos

    # 3. Then strict title heading.
    if child_title:
        title_positions = find_title_heading_candidates(
            page_text[min_pos:],
            child_title,
        )

        title_positions = [
            pos + min_pos
            for pos in title_positions
            if pos + min_pos >= min_pos
        ]

        if title_positions:
            return min(title_positions)

    return -1

def find_intro_child_boundary_in_combined_text(full_text, intro_node, child_node, start_pos):
    """
    Fallback boundary finder for intro nodes in combined text.

    Same logic as page-aware boundary:
    child start_phrase -> nearest child title before phrase.
    """
    if not full_text or not child_node or start_pos == -1:
        return -1

    min_pos = max(0, start_pos + 1)

    child_heading = get_node_heading_for_match(child_node)
    child_title = child_node.get("source_title") or child_node.get("title") or ""
    child_phrase = child_node.get("start_phrase") or ""
    parent_phrase = intro_node.get("start_phrase") or ""

    # 1. Best path: find child phrase, then move back to nearest title before it.
    if child_phrase and not phrases_match(parent_phrase, child_phrase):
        local_phrase_pos = find_phrase_position(full_text[min_pos:], child_phrase)

        if local_phrase_pos != -1:
            phrase_pos = min_pos + local_phrase_pos

            title_before_phrase = find_loose_title_before_phrase(
                text=full_text,
                title=child_title,
                phrase_pos=phrase_pos,
                min_pos=min_pos,
                lookback=1500,
            )

            if title_before_phrase != -1:
                return title_before_phrase

            return phrase_pos

    # 2. Then try full heading.
    heading_pos = find_heading_position(
        text=full_text,
        heading=child_heading,
        start_phrase=child_phrase,
        min_pos=min_pos,
        window=3000,
    )

    if heading_pos != -1:
        return heading_pos

    # 3. Then strict title heading.
    if child_title:
        title_positions = find_title_heading_candidates(
            full_text[min_pos:],
            child_title,
        )

        title_positions = [
            pos + min_pos
            for pos in title_positions
            if pos + min_pos >= min_pos
        ]

        if title_positions:
            return min(title_positions)

    return -1

def find_loose_title_before_phrase(text, title, phrase_pos, min_pos=0, lookback=1500):
    """
    Find a title occurrence before a known phrase position.

    This is used when the child heading is glued to a page header, e.g.
    '2024 ANNUAL REPORT - Sustainability Reporting 193Policies'.

    It is only used when the child phrase is already found, so it is safer
    than title-only matching.
    """
    if not text or not title or phrase_pos == -1:
        return -1

    min_pos = max(0, min_pos)
    search_start = max(min_pos, phrase_pos - lookback)
    search_text = text[search_start:phrase_pos]

    positions = find_normalized_occurrences(search_text, title)

    if not positions:
        return -1

    absolute_positions = [
        search_start + pos
        for pos in positions
        if search_start + pos >= min_pos
    ]

    if not absolute_positions:
        return -1

    # Use the closest title before the phrase.
    return max(absolute_positions)

def slice_pages_text(pdf_pages, start_page, start_pos, end_page, end_pos):
    """
    Slice text across pages using page-relative positions.
    """
    parts = []

    for page_number in range(start_page, end_page + 1):
        page_text = get_pdf_page_text(pdf_pages, page_number)

        if page_number == start_page and page_number == end_page:
            parts.append(page_text[start_pos:end_pos])
        elif page_number == start_page:
            parts.append(page_text[start_pos:])
        elif page_number == end_page:
            parts.append(page_text[:end_pos])
        else:
            parts.append(page_text)

    return "\n".join(parts).strip()

def slice_physical_pages_text(pdf_pages, start_index, start_pos, end_index, end_pos):
    """
    Slice text across actual physical page indexes.
    """
    parts = []

    for page_index in range(start_index, end_index + 1):
        page_text = get_pdf_page_text_by_index(pdf_pages, page_index)

        if page_index == start_index and page_index == end_index:
            parts.append(page_text[start_pos:end_pos])
        elif page_index == start_index:
            parts.append(page_text[start_pos:])
        elif page_index == end_index:
            parts.append(page_text[:end_pos])
        else:
            parts.append(page_text)

    return "\n".join(parts).strip()

def collect_node_start_candidates_on_pages(
    page_indexes,
    pdf_pages,
    node,
    found_start_page_index=None,
    found_start_pos=0,
):
    """
    Collect possible starts for node on the given physical page indexes.

    Returns candidates like:
    {
        "page_index": int,
        "pos": int,
        "score": float,
        "phrase_pos": int,
        "heading_pos": int,
        "context": str,
    }
    """
    candidates = []

    if not node:
        return candidates

    heading = get_node_heading_for_match(node)

    fallback_title = (
        node.get("source_title")
        or node.get("title")
        or ""
    )

    if node.get("is_intro_node"):
        fallback_title = (
            node.get("source_title")
            or re.sub(
                r"\s*-\s*Introduction\s*$",
                "",
                node.get("title", ""),
                flags=re.I,
            )
        )

    start_phrase = node.get("start_phrase", "")

    for page_index in page_indexes:
        page_text = get_pdf_page_text_by_index(pdf_pages, page_index)

        if not page_text:
            continue

        min_pos = (
            found_start_pos + 1
            if found_start_page_index is not None and page_index == found_start_page_index
            else 0
        )

        search_text = page_text[min_pos:]

        phrase_pos = find_phrase_position(search_text, start_phrase)
        phrase_pos = phrase_pos + min_pos if phrase_pos != -1 else -1

        heading_pos = find_heading_position(
            text=page_text,
            heading=heading,
            start_phrase=start_phrase,
            min_pos=min_pos,
            window=2500,
        )

        if heading_pos != -1:
            score = 50

            if phrase_pos != -1 and heading_pos <= phrase_pos:
                distance = phrase_pos - heading_pos
                if distance <= 1500:
                    score += 100 - min(80, distance / 20)

            candidates.append({
                "page_index": page_index,
                "pos": heading_pos,
                "score": score,
                "phrase_pos": phrase_pos,
                "heading_pos": heading_pos,
                "source": "heading",
                "context": page_text[max(0, heading_pos - 120): heading_pos + 250],
            })

        raw_candidates = collect_title_candidates(search_text, fallback_title)

        for cand in raw_candidates:
            absolute_pos = cand["pos"] + min_pos

            score = score_title_candidate(
                text=page_text,
                candidate_pos=absolute_pos,
                title=fallback_title,
                phrase_pos=phrase_pos,
                window=1500,
            )

            if cand["source"] == "strict_heading":
                score += 20

            if score <= 0:
                continue

            candidates.append({
                "page_index": page_index,
                "pos": absolute_pos,
                "score": score,
                "phrase_pos": phrase_pos,
                "heading_pos": heading_pos,
                "source": cand["source"],
                "context": page_text[max(0, absolute_pos - 120): absolute_pos + 250],
            })

        # Also add phrase itself as fallback candidate.
        if phrase_pos != -1:
            candidates.append({
                "page_index": page_index,
                "pos": phrase_pos,
                "score": 60,
                "phrase_pos": phrase_pos,
                "heading_pos": heading_pos,
                "source": "start_phrase",
                "context": page_text[max(0, phrase_pos - 120): phrase_pos + 250],
            })

    # Deduplicate same page/position.
    deduped = {}
    for cand in candidates:
        key = (cand["page_index"], cand["pos"])
        if key not in deduped or cand["score"] > deduped[key]["score"]:
            deduped[key] = cand

    return list(deduped.values())

def choose_best_node_start_candidate(candidates, preferred_page_indexes=None):
    """
    Choose best candidate.

    Preference:
    1. candidates on preferred/correct page indexes
    2. candidates where phrase_pos is found
    3. higher score
    4. earlier position on page
    """
    if not candidates:
        return None

    preferred_page_indexes = preferred_page_indexes or []

    def candidate_sort_key(c):
        on_preferred_page = c["page_index"] in preferred_page_indexes
        has_phrase = c.get("phrase_pos", -1) != -1

        return (
            1 if on_preferred_page else 0,
            1 if has_phrase else 0,
            c.get("score", 0),
            -c.get("pos", 0),
        )

    return max(candidates, key=candidate_sort_key)

def extract_intro_text_page_aware(current_node, next_node, pdf_pages):
    """
    Page-aware extraction for generated intro nodes.

    Intro = text between parent start and first real child start.

    Logic:
    1. Find intro start using the parent intro node.
    2. Find intro end using the actual child node start.
    3. Slice physical pages from intro start to child start.

    Important:
    - The end search is based on next_node.start_index.
    - It does not use a fixed +4 physical-page window.
    """
    start_page = current_node.get("start_index")
    end_page = current_node.get("end_index")

    if start_page is None or end_page is None:
        return "", -1

    if next_node and next_node.get("start_index") is not None:
        end_page = max(end_page, next_node.get("start_index"))

    # ------------------------------------------------------------
    # 1. Find intro start.
    # ------------------------------------------------------------
    found_start_logical_page = None
    found_start_page_index = None
    found_start_pos = -1

    for page_number in range(start_page, end_page + 1):
        variants = get_pdf_page_text_variants(pdf_pages, page_number)

        for variant in variants:
            page_text = variant["text"]

            start_pos = find_intro_start_on_page(
                page_text=page_text,
                intro_node=current_node,
                min_pos=0,
            )

            if start_pos != -1:
                found_start_logical_page = page_number
                found_start_page_index = variant["page_index"]
                found_start_pos = start_pos
                break

        if found_start_page_index is not None:
            break

    if found_start_page_index is None:
        return "", -1

    # ------------------------------------------------------------
    # 2. Build candidate physical pages for child start.
    # ------------------------------------------------------------
    if isinstance(pdf_pages, list):
        max_page_index = len(pdf_pages) - 1
    else:
        numeric_keys = [k for k in pdf_pages.keys() if isinstance(k, int)]
        max_page_index = max(numeric_keys) if numeric_keys else found_start_page_index

    expected_child_page = None

    if next_node and next_node.get("start_index") is not None:
        expected_child_page = next_node.get("start_index")
    else:
        expected_child_page = end_page

    candidate_end_page_indexes = []

    def add_candidate_page_index(page_index):
        if page_index is None:
            return

        if page_index < found_start_page_index:
            return

        if page_index < 0 or page_index > max_page_index:
            return

        if page_index not in candidate_end_page_indexes:
            candidate_end_page_indexes.append(page_index)

    # Only include the intro start page if the next node is expected to start
    # on the same logical page as the intro.
    if next_node and next_node.get("start_index") == current_node.get("start_index"):
        add_candidate_page_index(found_start_page_index)

    # Use the same logical-page variant mapping used elsewhere.
    if expected_child_page is not None:
        for variant in get_pdf_page_text_variants(pdf_pages, expected_child_page):
            add_candidate_page_index(variant.get("page_index"))

        # Add nearby physical candidates around the expected child page.
        for offset in [-3, -2, -1, 0, 1, 2, 3]:
            if isinstance(pdf_pages, list):
                add_candidate_page_index(expected_child_page - 1 + offset)
                add_candidate_page_index(expected_child_page + offset)
            else:
                add_candidate_page_index(expected_child_page + offset)

    candidate_end_page_indexes = sorted(candidate_end_page_indexes)

    progress(
        f"[intro pages] node={current_node.get('node_id')} "
        f"next={next_node.get('node_id') if next_node else None} "
        f"expected_child_page={expected_child_page} "
        f"found_start_page_index={found_start_page_index} "
        f"candidate_pages={candidate_end_page_indexes}"
    )

    # ------------------------------------------------------------
    # 3. Find child start on candidate physical pages.
    #    Search expected/correct child page first.
    #    If not found there, collect candidates from nearby pages.
    # ------------------------------------------------------------
    found_end_page_index = None
    found_end_pos = -1

    expected_page_indexes = []

    if expected_child_page is not None:
        for variant in get_pdf_page_text_variants(pdf_pages, expected_child_page):
            page_index = variant.get("page_index")

            if page_index is None:
                continue

            if page_index < found_start_page_index:
                continue

            if page_index < 0 or page_index > max_page_index:
                continue

            if page_index not in expected_page_indexes:
                expected_page_indexes.append(page_index)

    # Prefer the physical page that corresponds directly to the logical next-node page.
    # For list-backed pdf_pages, this is usually expected_child_page - 1.
    if isinstance(pdf_pages, list) and expected_child_page is not None:
        direct_index = expected_child_page - 1

        if (
            0 <= direct_index <= max_page_index
            and direct_index >= found_start_page_index
            and direct_index in expected_page_indexes
        ):
            expected_page_indexes.remove(direct_index)
            expected_page_indexes.insert(0, direct_index)

    # 3a. First search only the expected/correct page candidates.
    expected_candidates = collect_node_start_candidates_on_pages(
        page_indexes=expected_page_indexes,
        pdf_pages=pdf_pages,
        node=next_node,
        found_start_page_index=found_start_page_index,
        found_start_pos=found_start_pos,
    )

    best_expected = choose_best_node_start_candidate(
        candidates=expected_candidates,
        preferred_page_indexes=expected_page_indexes,
    )

    if best_expected is not None:
        found_end_page_index = best_expected["page_index"]
        found_end_pos = best_expected["pos"]
    else:
        # 3b. Fallback: collect candidates from all nearby candidate pages.
        fallback_page_indexes = [
            page_index
            for page_index in candidate_end_page_indexes
            if page_index not in expected_page_indexes
        ]

        fallback_candidates = collect_node_start_candidates_on_pages(
            page_indexes=fallback_page_indexes,
            pdf_pages=pdf_pages,
            node=next_node,
            found_start_page_index=found_start_page_index,
            found_start_pos=found_start_pos,
        )

        best_fallback = choose_best_node_start_candidate(
            candidates=fallback_candidates,
            preferred_page_indexes=[],
        )

        if best_fallback is not None:
            found_end_page_index = best_fallback["page_index"]
            found_end_pos = best_fallback["pos"]

    if found_end_page_index is None:
        return "", -1

    if found_end_page_index == found_start_page_index and found_end_pos <= found_start_pos:
        return "", -1

    progress(
        f"[intro end] node={current_node.get('node_id')} "
        f"end_page_index={found_end_page_index} "
        f"end_pos={found_end_pos}"
    )
        
    # ------------------------------------------------------------
    # 4. Slice from intro start to child start.
    # ------------------------------------------------------------
    text = slice_physical_pages_text(
        pdf_pages=pdf_pages,
        start_index=found_start_page_index,
        start_pos=found_start_pos,
        end_index=found_end_page_index,
        end_pos=found_end_pos,
    )

    return text, found_start_pos

def extract_section_text(current_node, next_node, pdf_pages, min_start_pos=0):
    progress(
        f"[extract] node={current_node.get('node_id')} "
        f"title={current_node.get('title')!r} "
        f"intro={current_node.get('is_intro_node', False)} "
        f"pages={current_node.get('start_index')}–{current_node.get('end_index')} "
        f"next={next_node.get('node_id') if next_node else None}"
    )
    
    if current_node.get("is_intro_node"):
        return extract_intro_text_page_aware(
            current_node=current_node,
            next_node=next_node,
            pdf_pages=pdf_pages,
        )

    start_page = current_node.get("start_index")
    end_page = current_node.get("end_index")

    if not start_page or not end_page:
        return "", -1

    if next_node and next_node.get("start_index"):
        end_page = max(end_page, next_node["start_index"])

    full_text = get_text_of_pdf_pages(pdf_pages, start_page, end_page)

    if current_node.get("is_intro_node"):
        start_pos = find_intro_start_in_full_text(
            text=full_text,
            intro_node=current_node,
            min_pos=0,
        )
    else:
        start_pos = find_node_start(
            full_text,
            current_node,
            min_pos=min_start_pos,
        )

    if start_pos == -1:
        start_pos = 0

    end_pos = len(full_text)
    boundary_found = False

    # ------------------------------------------------------------
    # Intro node handling:
    # start is found normally, but end must be the first real child.
    # ------------------------------------------------------------
    if current_node.get("is_intro_node") and next_node:
        intro_end_pos = find_node_start(
            text=full_text,
            node=next_node,
            min_pos=start_pos + 1,
        )

        if intro_end_pos == -1 or intro_end_pos <= start_pos:
            return "", -1

        return full_text[start_pos:intro_end_pos].strip(), start_pos
    
    # ------------------------------------------------------------
    # Normal node handling: find next boundary
    # ------------------------------------------------------------
    if next_node:
        next_start_page = next_node.get("start_index")

        if next_start_page:
            page_count = len(pdf_pages)

            candidate_pages = build_boundary_page_order(
                start_page=start_page,
                expected_page=next_start_page,
                page_count=page_count,
            )

            next_pos_on_page = -1
            matched_page = None

            for candidate_page in candidate_pages:
                candidate_text = get_text_of_pdf_pages(
                    pdf_pages,
                    candidate_page,
                    candidate_page
                )

                if current_node.get("is_intro_node"):
                    candidate_pos = find_intro_end_boundary(
                        text=candidate_text,
                        intro_node=current_node,
                        first_child=next_node,
                        start_pos=start_pos if candidate_page == start_page else 0,
                    )
                else:
                    if candidate_page == start_page and next_start_page == start_page:
                        current_heading = get_node_heading_for_match(current_node)

                        candidate_min_pos = estimate_title_end_pos(
                            text=candidate_text,
                            title=current_heading,
                            start_pos=start_pos,
                        )
                    else:
                        candidate_min_pos = 0

                    candidate_pos = find_node_start(
                        candidate_text,
                        next_node,
                        min_pos=candidate_min_pos,
                    )

                if candidate_pos is not None and candidate_pos != -1:
                    next_pos_on_page = candidate_pos
                    matched_page = candidate_page
                    break

            if matched_page is not None:
                prefix_text = ""
                if matched_page > start_page:
                    prefix_text = get_text_of_pdf_pages(
                        pdf_pages,
                        start_page,
                        matched_page - 1
                    )

                end_pos = len(prefix_text) + next_pos_on_page
                boundary_found = True

        else:
            current_heading = get_node_heading_for_match(current_node)

            current_title_end = estimate_title_end_pos(
                text=full_text,
                title=current_heading,
                start_pos=start_pos,
            )

            next_pos = find_node_start(
                full_text,
                next_node,
                min_pos=current_title_end
            )

            if next_pos != -1:
                end_pos = next_pos
                boundary_found = True

    if current_node.get("is_intro_node") and next_node and not boundary_found:
        return "", -1

    if end_pos <= start_pos:
        if current_node.get("is_intro_node"):
            return "", -1
        end_pos = len(full_text)

    section_text = full_text[start_pos:end_pos].strip()

    return section_text, start_pos

def dedupe_unstructured_tree_preserve_hierarchy(nodes, verbose=False, ancestor_keys=None):
    """
    Preserve unstructured hierarchy, but remove recursive repeated wrappers.

    This handles Belfius-like trees where the LLM repeats:

        EU Taxonomy
          Belfius as credit institution
            EU Taxonomy
              Belfius as credit institution
                ...

    We do NOT flatten the whole tree.
    We only lift children of repeated ancestor nodes.
    """
    if ancestor_keys is None:
        ancestor_keys = set()

    result = []
    sibling_keys = set()

    for node in nodes or []:
        if not isinstance(node, dict):
            continue

        node = dict(node)
        children = node.get("nodes") or []

        # Never apply this to structured nodes.
        if node_has_structure(node):
            if children:
                node["nodes"] = dedupe_unstructured_tree_preserve_hierarchy(
                    children,
                    verbose=verbose,
                    ancestor_keys=set(),
                )

            result.append(node)
            continue

        key = unstructured_repeat_key(node)

        # First clean children.
        next_ancestor_keys = set(ancestor_keys)
        if key:
            next_ancestor_keys.add(key)

        cleaned_children = dedupe_unstructured_tree_preserve_hierarchy(
            children,
            verbose=verbose,
            ancestor_keys=next_ancestor_keys,
        )

        if cleaned_children:
            node["nodes"] = cleaned_children
        else:
            node.pop("nodes", None)

        # Case 1:
        # This node repeats an ancestor. Remove this wrapper and lift its children.
        if key and key in ancestor_keys:
            if verbose:
                print(
                    "LIFT repeated ancestor wrapper:",
                    node.get("node_id"),
                    repr(node.get("title")),
                )

            result.extend(cleaned_children)
            continue

        # Case 2:
        # Duplicate sibling with same title/start phrase. Keep first, merge children.
        if key and key in sibling_keys:
            if verbose:
                print(
                    "SKIP duplicate unstructured sibling:",
                    node.get("node_id"),
                    repr(node.get("title")),
                )

            # If it has useful children, lift them instead of losing them.
            result.extend(cleaned_children)
            continue

        if key:
            sibling_keys.add(key)

        result.append(node)

    return result

def strip_leading_section_number(title):
    """
    Remove section numbers from unstructured titles/headings.

    Examples:
        5.2. EU Taxonomy – mandatory disclosures
        -> EU Taxonomy – mandatory disclosures

        5.2.1. Belfius as a credit institution
        -> Belfius as a credit institution
    """
    title = title or ""

    return re.sub(
        r"^\s*\d+(?:\.\d+)*\.?\s+",
        "",
        title,
    ).strip()

def unstructured_repeat_key(node):
    """
    Conservative key for detecting repeated unstructured section wrappers.

    Important:
    - Strips leading numbering from titles.
    - Uses title + start_phrase when possible.
    - Ignores intro nodes.
    """
    if node.get("is_intro_node"):
        return None

    title = (
        node.get("source_title")
        or node.get("title")
        or ""
    )

    title = strip_leading_section_number(title)

    start_phrase = node.get("start_phrase") or ""

    title_key = normalize_for_match(title)
    phrase_key = normalize_for_match(start_phrase)

    if not title_key:
        return None

    if phrase_key:
        return (title_key, phrase_key)

    return (title_key, None)


def node_has_structure(node):
    return bool((node.get("structure") or "").strip())

def structured_sibling_key(node):
    """
    Only structured nodes use structure number as duplicate key.
    Unstructured nodes are handled separately.
    """
    structure = str(node.get("structure") or "").strip()

    if structure:
        return ("structure", structure)

    return None


def merge_structured_duplicate_sibling_keep_first(existing, duplicate, verbose=False):
    """
    Keep the first structured sibling node.
    Merge only missing metadata and merge children recursively.
    Do not rebuild hierarchy.
    """

    for key in [
        "summary",
        "start_phrase",
        "start_index",
        "end_index",
        "heading",
        "title",
    ]:
        if not existing.get(key) and duplicate.get(key):
            existing[key] = duplicate[key]

    existing_children = existing.get("nodes") or []
    duplicate_children = duplicate.get("nodes") or []

    if duplicate_children:
        existing["nodes"] = dedupe_structured_siblings_only(
            existing_children + duplicate_children,
            verbose=verbose,
        )

    if verbose:
        print(
            "MERGE structured duplicate sibling:",
            duplicate.get("node_id"),
            duplicate.get("structure"),
            repr(duplicate.get("title")),
            "into",
            existing.get("node_id"),
            repr(existing.get("title")),
        )

    return existing


def dedupe_structured_siblings_only(nodes, verbose=False):
    """
    Deduplicate only structured sibling nodes.

    This does not:
    - flatten the tree
    - rebuild the tree
    - move nodes between parents
    - create synthetic parents
    """

    result = []
    seen = {}

    for node in nodes or []:
        if not isinstance(node, dict):
            continue

        node = dict(node)

        if node.get("nodes"):
            node["nodes"] = dedupe_structured_siblings_only(
                node.get("nodes") or [],
                verbose=verbose,
            )

        key = structured_sibling_key(node)

        if key and key in seen:
            existing = seen[key]

            merge_structured_duplicate_sibling_keep_first(
                existing,
                node,
                verbose=verbose,
            )

            continue

        result.append(node)

        if key:
            seen[key] = node

    return result

def post_process_structure_duplicates(structure, verbose=False):
    """
    Stable cleanup version.

    1. Clean unstructured repeated wrappers.
    2. Merge structured duplicate siblings only.

    This intentionally does not repair misnested structured nodes.
    """

    if verbose:
        print("POSTPROCESS MODE: stable unstructured + structured sibling dedupe")

    structure = dedupe_unstructured_tree_preserve_hierarchy(
        structure,
        verbose=verbose,
    )

    structure = dedupe_structured_siblings_only(
        structure,
        verbose=verbose,
    )

    return structure

def chunk_dedupe_key(row):
    """
    Decide whether two chunk rows represent the same chunk.

    Prefer structure + normalized title + start phrase.
    This avoids duplicate chunks while keeping the tree unchanged.
    """
    structure = str(row.get("structure") or "").strip()
    title = normalize_for_match(row.get("title") or "")
    start_phrase = normalize_for_match(row.get("start_phrase") or "")

    if structure and title and start_phrase:
        return ("structure_title_phrase", structure, title, start_phrase)

    if structure and title:
        return ("structure_title", structure, title)

    text = row.get("text") or ""
    text_key = normalize_for_match(text[:1000])

    if text_key:
        return ("text_start", text_key)

    return None

def dedupe_chunk_rows_by_pointer(rows, verbose=False):
    """
    Keep all rows, but point duplicate rows to the same canonical chunk.

    This preserves the structure, because no nodes are moved or deleted.
    """

    seen = {}
    output = []

    for row in rows:
        row = dict(row)

        key = chunk_dedupe_key(row)

        if key and key in seen:
            canonical = seen[key]

            row["canonical_chunk_id"] = canonical["canonical_chunk_id"]
            row["duplicate_of_chunk_id"] = canonical["chunk_id"]
            row["is_duplicate_chunk"] = True

            # Important: reuse the canonical chunk id for retrieval/evaluation
            row["retrieval_chunk_id"] = canonical["canonical_chunk_id"]

            if verbose:
                print(
                    "DUPLICATE CHUNK POINTER:",
                    row.get("chunk_id"),
                    "->",
                    canonical["canonical_chunk_id"],
                    repr(row.get("title")),
                )

        else:
            row["canonical_chunk_id"] = row.get("chunk_id")
            row["duplicate_of_chunk_id"] = None
            row["is_duplicate_chunk"] = False
            row["retrieval_chunk_id"] = row.get("chunk_id")

            if key:
                seen[key] = row

        output.append(row)

    return output

def add_intro_leaf_nodes(nodes):
    """
    Insert intro candidate nodes into the actual tree.

    Parent:
        7.8 ESG1...
            7.8.1 Companies excluded...

    becomes:
        7.8 ESG1...
            7.8_intro ESG1... - Introduction
            7.8.1 Companies excluded...

    The intro node is only a candidate here.
    build_leaf_text_rows() later decides whether to keep or remove it.
    """
    result = []

    for node in nodes:
        children = node.get("nodes") or []

        if node.get("is_intro_node"):
            result.append(node)
            continue
        
        if not children:
            continue

        # Avoid adding duplicate intro nodes if this function runs twice.
        already_has_intro = any(
            child.get("is_intro_node")
            and child.get("parent_node_id") == node.get("node_id")
            for child in children
        )

        if not already_has_intro:
            first_real_child = children[0]

            parent_start = node.get("start_index")
            first_child_start = first_real_child.get("start_index")

            if parent_start is not None and first_child_start is not None:

                if parent_start is not None and first_child_start is not None:
                    intro_id = f"{node.get('node_id')}_intro"
                    clean_title = strip_leading_section_number(node.get("title") or "")
                    clean_heading = strip_leading_section_number(node.get("heading") or node.get("title") or "")

                    intro_node = {
                        "title": f"{clean_title} - Introduction",
                        "source_title": clean_title,
                        "source_heading": clean_heading,
                        "structure": node.get("structure"),
                        "heading": node.get("heading") or get_node_heading_for_match(node),
                        "start_phrase": node.get("start_phrase"),
                        "summary": "",
                        "node_id": intro_id,
                        "chunk_id": intro_id,
                        "parent_node_id": node.get("node_id"),
                        "start_index": parent_start,
                        "end_index": first_child_start,
                        "is_intro_node": True,
                        "_parent_node": node,
                        "_first_child_node": first_real_child,
                    }


                    node["nodes"] = [intro_node] + children


        # Recurse into real children only, not generated intro nodes.
        for child in list(node.get("nodes") or []):
            if not child.get("is_intro_node"):
                add_intro_leaf_nodes([child])

    return nodes

def remove_skipped_intro_nodes(nodes):
    """
    Remove intro candidates marked with _skip_intro_node from the actual tree.
    """
    for node in nodes:
        children = node.get("nodes") or []

        if not children:
            continue

        kept_children = []

        for child in children:
            if child.get("is_intro_node") and child.get("_skip_intro_node"):
                continue

            kept_children.append(child)

        node["nodes"] = kept_children

        for child in node["nodes"]:
            if not child.get("is_intro_node"):
                remove_skipped_intro_nodes([child])

    return nodes

def remove_heading_like_text(text, heading_or_title):
    """
    Remove one heading/title occurrence from text using normalized matching.
    """
    if not text or not heading_or_title:
        return text or ""

    heading_key = normalize_for_match(heading_or_title)
    text_key, text_map = normalize_with_map(text)

    if not heading_key or not text_key or not text_map:
        return text

    pos_key = text_key.find(heading_key)

    if pos_key == -1:
        return text

    start = text_map[pos_key]
    end_key = pos_key + len(heading_key)
    end = text_map[end_key] if end_key < len(text_map) else len(text)

    return (text[:start] + " " + text[end:]).strip()

def has_real_intro_text(text, parent_node=None, first_child=None, min_words=6):
    """
    Keep intro only if it contains real prose after removing parent/child headings.

    Rejects:
    - empty text
    - only parent title/heading
    - only child title/heading
    - only page numbers/header noise
    """
    if not text or not text.strip():
        return False

    check_text = text.strip()

    parent_heading = get_node_heading_for_match(parent_node) if parent_node else ""
    parent_title = parent_node.get("title", "") if parent_node else ""

    child_heading = get_node_heading_for_match(first_child) if first_child else ""
    child_title = first_child.get("title", "") if first_child else ""

    # Remove parent and child headings/titles from validation text.
    for heading_or_title in [
        parent_heading,
        parent_title,
        child_heading,
        child_title,
    ]:
        check_text = remove_heading_like_text(check_text, heading_or_title)

    useful_lines = []

    for line in check_text.splitlines():
        line = line.strip()

        if not line:
            continue

        # Page number only.
        if re.fullmatch(r"\d{1,5}", line):
            continue

        # Common report header/footer noise.
        if "ING Group Annual" in line:
            continue

        if "Contents ING at a glance" in line:
            continue

        if line in {">", "|"}:
            continue

        words = re.findall(r"[A-Za-z]{2,}", line)

        if not words:
            continue

        # Reject all-uppercase short heading-like lines.
        lowercase_words = sum(1 for w in words if w.islower())

        if len(line) < 120 and lowercase_words == 0:
            continue

        useful_lines.append(line)

    check_text = " ".join(useful_lines).strip()

    words = re.findall(r"[A-Za-z]{2,}", check_text)

    if len(words) < min_words:
        return False

    return True

def get_next_non_skipped_leaf(leaf_nodes, index):
    for j in range(index + 1, len(leaf_nodes)):
        candidate = leaf_nodes[j]
        if candidate.get("_skip_intro_node"):
            continue
        return candidate
    return None

def normalize_text_key(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def get_node_text(node):
    return (
        node.get("text")
        or node.get("page_content")
        or node.get("summary")
        or ""
    )


def get_token_count_for_query_flag(node):
    token_count = (
        node.get("text_token_count")
        or node.get("token_count")
        or None
    )

    if token_count is not None:
        try:
            return int(token_count)
        except Exception:
            pass

    return len(str(get_node_text(node)).split())


def is_intro_chunk(node):
    chunk_id = str(node.get("chunk_id", "")).lower()
    node_id = str(node.get("node_id", "")).lower()
    title = str(node.get("title", "")).lower()

    return (
        bool(node.get("is_intro_node", False))
        or "intro" in chunk_id
        or "intro" in node_id
        or title.endswith("- introduction")
    )


def is_appendix_like(node):
    title = normalize_text_key(node.get("title", ""))
    heading = normalize_text_key(node.get("heading", ""))

    appendix_terms = [
        "appendix",
        "annex",
        "glossary",
        "assurance report",
        "limited assurance",
        "independent auditor",
        "table of contents",
        "contents",
        "summary",
        "summaries",
    ]

    combined = f"{title} {heading}"

    return any(term in combined for term in appendix_terms)


def table_likeness_score(text):
    text = str(text or "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return 0

    table_like_lines = 0

    for line in lines:
        digit_count = sum(ch.isdigit() for ch in line)
        percent_count = line.count("%")
        word_count = len(line.split())

        starts_with_row_number = bool(re.match(r"^\s*\d{1,3}\s+", line))

        # Header rows like: a b c d e f g h i j
        alphabet_header = bool(
            re.match(
                r"^\s*(?:[a-z]{1,2}\s+){5,}[a-z]{1,2}\s*$",
                line.strip(),
                flags=re.I,
            )
        )

        # Rows full of table placeholders: - - - - -
        many_dashes = len(re.findall(r"(?<!\w)-(?!\w)", line)) >= 4

        # Rows with many numeric cells.
        many_numbers = digit_count >= 8

        # Percent-heavy table rows.
        many_percentages = percent_count >= 3

        # Wide column spacing, but only count it if numbers/percentages are present.
        column_spacing = bool(re.search(r"\S\s{3,}\S", line)) and (
            digit_count >= 2 or percent_count >= 1
        )

        # Numbered financial table row.
        numbered_numeric_row = (
            starts_with_row_number
            and digit_count >= 3
            and word_count <= 25
        )

        if (
            alphabet_header
            or many_dashes
            or many_numbers
            or many_percentages
            or column_spacing
            or numbered_numeric_row
        ):
            table_like_lines += 1

    return table_like_lines / len(lines)


def get_row_text(row):
    return row.get("text") or row.get("page_content") or ""

def is_table_or_template_title(row):
    title = normalize_text_key(row.get("title", ""))
    heading = normalize_text_key(row.get("heading", ""))

    words = set((title + " " + heading).split())

    return "table" in words or "template" in words

def is_large_table_chunk(row):
    text = str(get_row_text(row) or "")
    token_count = get_token_count_for_query_flag(row)
    score = table_likeness_score(text)

    return (
        is_table_or_template_title(row)
        or (token_count > 1500 and score > 0.18)
        or (token_count > 3000 and score > 0.12)
    )

def query_count_for_chunk(row):
    token_count = get_token_count_for_query_flag(row)
    text = str(get_node_text(row) or "").strip()

    if len(text) < 300:
        return 0

    if token_count < 100:
        return 0

    if token_count < 200:
        return 1

    if token_count < 300:
        return 2

    if token_count < 400:
        return 3

    if token_count < 500:
        return 4
    
    if token_count < 600:
        return 5
    
    if token_count < 700:
        return 6
    
    if token_count < 800:
        return 7
    
    if token_count < 900:
        return 8
    
    if token_count < 1000:
        return 9
    
    return 10

def should_generate_queries(row):
    """
    Decide whether a final chunk row should be used for query generation.

    Returns:
        (True, "") if queries should be generated
        (False, reason) otherwise
    """

    if is_intro_chunk(row):
        return False, "intro"

    if is_appendix_like(row):
        return False, "appendix_like"

    if is_large_table_chunk(row):
        return False, "table_like"

    num_queries = query_count_for_chunk(row)

    if num_queries == 0:
        return False, "too_short"

    return True, ""

def collect_appendix_descendant_node_ids(nodes, inherited_appendix=False):
    """
    Return node_ids for all nodes that are inside an appendix-like branch.

    This does not modify the structure.
    """
    appendix_node_ids = set()

    for node in nodes or []:
        node_id = node.get("node_id")

        current_is_appendix = inherited_appendix or is_appendix_like(node)

        if current_is_appendix and node_id:
            appendix_node_ids.add(node_id)

        child_ids = collect_appendix_descendant_node_ids(
            node.get("nodes") or [],
            inherited_appendix=current_is_appendix
        )

        appendix_node_ids.update(child_ids)

    return appendix_node_ids

def add_query_flags_to_rows(rows, structure):
    """
    Add query-generation flags to chunk rows only.
    Does not modify the PageIndex structure.
    """
    appendix_node_ids = collect_appendix_descendant_node_ids(structure)

    for row in rows:
        node_id = row.get("node_id")

        if node_id in appendix_node_ids:
            row["generate_queries"] = False
            row["skip_query_reason"] = "appendix_like"
            row["num_queries"] = 0
            continue

        generate, reason = should_generate_queries(row)

        row["generate_queries"] = generate
        row["skip_query_reason"] = reason
        row["num_queries"] = query_count_for_chunk(row) if generate else 0

    return rows

def build_leaf_text_rows(structure, pdf_pages, model=None, max_tokens=32768):
    structure = post_process_structure_duplicates(
        structure,
        verbose=True,
    )

    print("\nROOT STRUCTURES AFTER POSTPROCESS:")
    for n in structure:
        print(n.get("node_id"), n.get("structure"), repr(n.get("title")), "children:", len(n.get("nodes") or []))

    structure = add_intro_leaf_nodes(structure)
    leaf_nodes = get_leaf_nodes_in_order(structure)

    rows = []
    last_title_end_pos_by_page = {}

    for index, node in enumerate(leaf_nodes):
        if node.get("is_intro_node"):
            # Parent intro must stop at the parent’s first real child,
            # even if that first child is not a leaf.
            next_node = node.get("_first_child_node")
        else:
            # Normal leaf nodes stop at the next actual leaf/chunk candidate.
            next_node = leaf_nodes[index + 1] if index + 1 < len(leaf_nodes) else None

        node_id = node.get("node_id")
        start_page = node.get("start_index")

        min_start_pos = 0

        if not node.get("is_intro_node"):
            if start_page in last_title_end_pos_by_page:
                min_start_pos = last_title_end_pos_by_page[start_page]

        text, found_start_pos = extract_section_text(
            current_node=node,
            next_node=next_node,
            pdf_pages=pdf_pages,
            min_start_pos=min_start_pos,
        )
        
        # ------------------------------------------------------------
        # Intro nodes are only kept if they contain actual content.
        # If not, mark them and remove them from the structure later.
        # ------------------------------------------------------------
        if node.get("is_intro_node"):
            parent_node = node.get("_parent_node")
            first_child_node = node.get("_first_child_node")

            if not parent_node or not first_child_node:
                node["_skip_intro_node"] = True
                continue

            if intro_is_only_heading_or_title(
                text,
                parent_node=parent_node,
                first_child=first_child_node,
            ):
                node["_skip_intro_node"] = True
                continue

            accepted_intro = has_real_intro_text(
                text,
                parent_node=parent_node,
                first_child=first_child_node,
                min_words=6,
            )


            if not accepted_intro:
                node["_skip_intro_node"] = True
                continue


        # Skip empty failed extractions for all nodes.
        if not text or not text.strip():
            if node.get("is_intro_node"):
                node["_skip_intro_node"] = True
            continue

        # ------------------------------------------------------------
        # Update same-page min_start_pos only for real non-intro nodes.
        # This prevents later chunks on the same page from starting before
        # a heading that was already consumed.
        # ------------------------------------------------------------
        if (
            found_start_pos != -1
            and not node.get("is_intro_node")
            and start_page is not None
        ):
            node_title = get_node_heading_for_match(node)

            page_text = get_text_of_pdf_pages(
                pdf_pages,
                start_page,
                start_page,
            )

            title_end_pos = estimate_title_end_pos(
                text=page_text,
                title=node_title,
                start_pos=found_start_pos,
            )

            if not node.get("is_intro_node"):
                last_title_end_pos_by_page[start_page] = max(
                    last_title_end_pos_by_page.get(start_page, 0),
                    found_start_pos,
                    title_end_pos,
                )

        original_token_count = count_tokens(text, model=model)

        chunks = split_until_within_limit(
            title=node.get("title"),
            text=text,
            max_tokens=max_tokens,
            model=model,
        )

        # ------------------------------------------------------------
        # Normal one-chunk node
        # ------------------------------------------------------------
        if len(chunks) == 1:
            chunk = chunks[0]
            token_count = count_tokens(chunk["text"], model=model)

            if node.get("is_intro_node"):
                summary = generate_node_summary(
                   title=chunk["title"],
                   text=chunk["text"],
                   model=model,
                )
            else:
                summary = node.get("summary") or generate_node_summary(
                   title=chunk["title"],
                   text=chunk["text"],
                   model=model,
                )

            rows.append({
                "chunk_id": node_id,
                "node_id": node_id,
                "parent_node_id": node.get("parent_node_id", ""),
                "structure": node.get("structure", ""),
                "title": chunk["title"],
                "heading": node.get("heading", ""),
                "start_phrase": node.get("start_phrase", ""),
                "summary": summary,
                "start_index": node.get("start_index"),
                "end_index": node.get("end_index"),
                "token_count": token_count,
                "text": chunk["text"],
                "is_intro_node": node.get("is_intro_node", False),
                "skip_intro_node": node.get("_skip_intro_node", False), 
            })

            node["chunk_id"] = node_id
            node["text_token_count"] = token_count
            node["summary"] = summary
            node.pop("text", None)

        # ------------------------------------------------------------
        # Node too large: split into child chunks
        # ------------------------------------------------------------
        else:
            node["nodes"] = []
            node["text_token_count"] = original_token_count
            node.pop("chunk_id", None)
            node.pop("text", None)

            for i, chunk in enumerate(chunks, start=1):
                child_id = f"{node_id}_{i:03d}"
                child_token_count = count_tokens(chunk["text"], model=model)

                child_title = chunk["title"]

                child_summary = generate_node_summary(
                    title=child_title,
                    text=chunk["text"],
                    model=model,
                )

                child_node = {
                    "chunk_id": child_id,
                    "node_id": child_id,
                    "parent_node_id": node_id,
                    "structure": node.get("structure", ""),
                    "title": child_title,
                    "heading": node.get("heading", ""),
                    "start_phrase": node.get("start_phrase", ""),
                    "summary": summary,
                    "start_index": node.get("start_index"),
                    "end_index": node.get("end_index"),
                    "token_count": child_token_count,
                    "text": chunk["text"],
                    "is_intro_node": False,
                    "is_llm_split_node": True,
                    "skip_intro_node": False,
                }

                node["nodes"].append(child_node)

                rows.append({
                    "chunk_id": child_id,
                    "node_id": child_id,
                    "parent_node_id": node_id,
                    "structure": node.get("structure", ""),
                    "title": child_title,
                    "heading": node.get("heading", ""),
                    "start_phrase": node.get("start_phrase", ""),
                    "summary": summary,
                    "start_index": node.get("start_index"),
                    "end_index": node.get("end_index"),
                    "token_count": child_token_count,
                    "text": chunk["text"],
                    "is_intro_node": False,
                    "is_llm_split_node": True,
                    "skip_intro_node": False,
                })

            node["summary"] = generate_parent_summary_from_children(
                title=node.get("title"),
                child_nodes=node["nodes"],
                model=model,
            )

    remove_skipped_intro_nodes(structure)

    rows = [
        row for row in rows
        if not row.get("skip_intro_node", False)
    ]

    rows = remove_next_heading_from_previous_rows(
        rows,
        model=model,
    )

    rows = dedupe_chunk_rows_by_pointer(
        rows,
        verbose=True,
    )

    rows = add_query_flags_to_rows(
        rows,
        structure,
    )

    return rows, structure

def get_canonical_retrieval_rows(rows):
    canonical = []

    for row in rows:
        if not row.get("is_duplicate_chunk", False):
            canonical.append(row)

    return canonical

def save_leaf_text_jsonl(rows, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
