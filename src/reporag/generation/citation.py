"""Line-level citation extraction (Issue 25).

Parses an LLM answer for ``[file_path:start_line-end_line]`` citation
markers (the format :data:`~reporag.generation.prompt_builder.CITATION_FORMAT`
instructs the model to use), validates each one against the code context the
model was actually shown, and reports how much of the answer is backed by a
citation at all.

Why
---
A citation the model invented -- a file that was never retrieved, or a line
range wider than what was actually shown -- is indistinguishable from a real
one just by looking at it; it has to be checked against the context. Three
things matter here:

* **Extraction** must be exact about the format
  (:data:`~reporag.generation.prompt_builder.CITATION_FORMAT`) so a citation
  in a slightly different shape (a missing dash, a stray space) is not
  silently dropped and mistaken for "the model didn't cite this claim".
* **Validation** must allow a citation to be *narrower* than what was shown.
  The prompt's own citation rule
  (:data:`~reporag.generation.prompt_builder._CITATION_RULES`) tells the
  model to "cite the narrowest range that supports the claim" -- a context
  block for lines 20-31 and a citation of lines 22-24 is the model doing
  exactly what was asked, not a hallucination. Validation therefore checks
  *containment* within a shown range, not exact equality.
* **Coverage** is a heuristic, not a hard measurement -- "claim" is not a
  formally defined unit of text. This module's definition (see
  :func:`compute_citation_coverage`) is stated plainly so a caller can judge
  whether it fits their use, rather than presenting a precise-looking number
  for an inherently fuzzy quantity.

Design
------
Three layers, each independently testable and usable on its own:

* :func:`extract_citations` -- pure parsing, no context needed.
* :func:`validate_citations` -- pure validation against a context string,
  reusing :func:`~reporag.generation.prompt_builder.extract_file_index`'s
  same header format so a citation is checked against literally the same
  parse the prompt's own "FILES IN CONTEXT" whitelist was built from.
* :func:`analyze_citations` -- the end-to-end convenience combining both,
  plus coverage, returning the single :class:`CitationReport` a caller
  actually wants.

Accepts either a raw context string or a
:class:`~reporag.generation.prompt_builder.BuiltPrompt` (reading its
``sections["context"]``) wherever context is needed, matching this
codebase's now-established pattern (see ``PromptBuilder.build_prompt``) of
not making a caller unpack a structured object just to pass a string in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reporag.generation.prompt_builder import BuiltPrompt

__all__ = [
    "Citation",
    "CitationReport",
    "analyze_citations",
    "compute_citation_coverage",
    "extract_citations",
    "validate_citations",
]

# Matches this codebase's exact citation format:
# [file_path:start_line-end_line], including the "?" placeholder
# ContextAssembler emits for an unknown line number, since a model copying a
# header verbatim (as the prompt's citation rule instructs) could
# legitimately reproduce a "?" bound too. File paths are assumed not to
# contain ':' or brackets -- true of every real filesystem path this
# pipeline indexes -- so the boundaries are unambiguous without lazy
# matching.
_CITATION_RE = re.compile(
    r"\[(?P<file_path>[^\[\]:]+):(?P<start>\d+|\?)-(?P<end>\d+|\?)\]"
)

# Matches the same "## path (lines a-b)" header ContextAssembler emits and
# prompt_builder.extract_file_index parses, but keeps the *body* of each
# block too (not just the header), since a valid citation's `snippet` is
# read out of that body.
_CONTEXT_BLOCK_RE = re.compile(
    r"^##\s+(?P<file_path>.+?)\s+\(lines\s+(?P<start>\d+|\?)-(?P<end>\d+|\?)\)\s*$"
    r"\n```[^\n]*\n(?P<body>.*?)\n```",
    re.MULTILINE | re.DOTALL,
)

# A citation-marker-only fragment (after citations are stripped, nothing
# but whitespace/punctuation remains) is folded back into the claim it
# follows when splitting into claims for coverage -- see
# `_split_into_candidate_claims`.

# A candidate claim shorter than this many characters (after stripping
# citation markers and whitespace) is treated as structural noise -- a bare
# list number, a lone heading, a leftover fragment from splitting -- rather
# than a claim that could reasonably carry its own citation. Tuned to
# exclude things like "1." or "Summary:" while keeping short factual
# sentences like "It returns None." (this heuristic's limitations are
# documented on `compute_citation_coverage`).
# A newline immediately followed by a bullet marker (-, *, or the Unicode bullet char) or a
# numbered-list marker ("1.", "2.", ...) is treated as a claim boundary in
# `_split_into_candidate_claims`, in addition to sentence-ending
# punctuation. Without this, a bullet or numbered list with no terminal
# punctuation between items ("- item one\n- item two") is read as a
# single run-on claim, so one cited bullet among several uncited ones
# would falsely report full coverage for the whole list.
_LIST_ITEM_BOUNDARY_RE = re.compile(r"\n(?=\s*(?:[-*\u2022]|\d+\.)\s)")

_MIN_CLAIM_LENGTH = 12


@dataclass
class Citation:
    """One citation marker extracted from an answer.

    Attributes:
        file_path: The cited file path, exactly as written in the marker.
        start_line: The cited start line, or ``None`` for a ``?``
            placeholder.
        end_line: The cited end line, or ``None`` for a ``?`` placeholder.
        snippet: The actual source text for ``start_line-end_line``, read
            from the matching context block. Empty until validated, and
            empty after validation too when the citation was invalid (there
            is no source to show for a range that was never retrieved).
        valid: ``True`` when the cited file and line range are contained
            within a block actually shown in the context, ``False`` when
            they are not (a hallucinated file, an out-of-range line number,
            or a reversed range), ``None`` before :func:`validate_citations`
            has run.
        raw: The exact matched text, e.g. ``"[src/a.py:10-20]"`` -- useful
            for locating the citation back in the original answer text.
    """

    file_path: str
    start_line: int | None
    end_line: int | None
    snippet: str = ""
    valid: bool | None = None
    raw: str = ""


@dataclass
class CitationReport:
    """The end-to-end result of :func:`analyze_citations`.

    Attributes:
        citations: Every citation found, in order of appearance, each
            already validated (``.valid`` is never ``None`` here).
        coverage: :func:`compute_citation_coverage`'s result for the same
            answer text -- the fraction of substantive claims that carry at
            least one citation, valid or not (an invalid citation is still
            evidence the model *tried* to ground the claim; whether the
            grounding held up is a separate question, answered by
            ``valid_count``/``invalid_count``).
        valid_count: Number of citations that passed validation.
        invalid_count: Number of citations that failed validation --
            flagged as likely hallucinated.
    """

    citations: list[Citation] = field(default_factory=list)
    coverage: float = 0.0
    valid_count: int = 0
    invalid_count: int = 0

    @property
    def all_valid(self) -> bool:
        """``True`` when every extracted citation passed validation.

        Vacuously ``True`` when there are no citations at all -- an answer
        with zero citations has zero invalid ones, but that is a coverage
        problem (see :attr:`coverage`), not a validity problem, and the two
        are deliberately kept separate rather than conflated into one flag.
        """
        return self.invalid_count == 0


# ---------------------------------------------------------------------------
# Context parsing (validation-side)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ContextBlock:
    """One parsed ``## file (lines a-b)`` block from an assembled context."""

    file_path: str
    start_line: int | None
    end_line: int | None
    lines: tuple[str, ...]


def _parse_int_or_none(value: str) -> int | None:
    return None if value == "?" else int(value)


def _parse_context_blocks(context: str) -> list[_ContextBlock]:
    """Parse every ``## file (lines a-b)`` block, header and body, from *context*."""
    blocks = []
    for match in _CONTEXT_BLOCK_RE.finditer(context or ""):
        blocks.append(
            _ContextBlock(
                file_path=match["file_path"],
                start_line=_parse_int_or_none(match["start"]),
                end_line=_parse_int_or_none(match["end"]),
                lines=tuple(match["body"].split("\n")),
            )
        )
    return blocks


def _resolve_context_text(context: str | BuiltPrompt) -> str:
    """Accept a raw context string or a `BuiltPrompt`, returning the context text."""
    if isinstance(context, str):
        return context
    sections = getattr(context, "sections", None)
    if sections and "context" in sections:
        return sections["context"]
    # Fall back to the full user message, which contains the fenced context
    # (inside its own "=== CODE CONTEXT ===" fence) even without `.sections`.
    return getattr(context, "user", "") or ""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_citations(text: str) -> list[Citation]:
    """Extract every citation marker from *text*, unvalidated.

    Pure parsing -- no context is consulted, so every :class:`Citation`
    returned has ``valid=None``. Use :func:`validate_citations` (or
    :func:`analyze_citations` for both steps at once) to populate it.

    Args:
        text: The LLM's answer text.

    Returns:
        A list of :class:`Citation`, in order of first appearance.
        Duplicate markers (the same file and range cited twice) are kept as
        separate entries -- each occurrence is a real claim being backed,
        even if it repeats an earlier one.
    """
    citations = []
    for match in _CITATION_RE.finditer(text or ""):
        citations.append(
            Citation(
                file_path=match["file_path"],
                start_line=_parse_int_or_none(match["start"]),
                end_line=_parse_int_or_none(match["end"]),
                raw=match.group(0),
            )
        )
    return citations


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _is_contained(citation: Citation, block: _ContextBlock) -> bool:
    """Whether *citation*'s range fits within *block*'s shown range.

    A ``?`` bound on either side only matches a block with the identical
    ``?`` bound on that side -- there is no numeric containment check
    possible for an unknown line number, so it falls back to exact
    agreement, which is what a model faithfully copying a ``(lines ?-?)``
    header verbatim would produce.
    """
    if citation.start_line is None or citation.end_line is None:
        return (
            citation.start_line == block.start_line
            and citation.end_line == block.end_line
        )
    if block.start_line is None or block.end_line is None:
        return False
    if citation.start_line > citation.end_line:
        return False  # a reversed range is never valid, regardless of block
    return (
        block.start_line <= citation.start_line <= citation.end_line <= block.end_line
    )


def _snippet_for(citation: Citation, block: _ContextBlock) -> str:
    """Read the source lines *citation* covers out of *block*'s body."""
    if (
        citation.start_line is None
        or citation.end_line is None
        or block.start_line is None
    ):
        return "\n".join(block.lines)
    offset_start = citation.start_line - block.start_line
    offset_end = citation.end_line - block.start_line
    return "\n".join(block.lines[offset_start : offset_end + 1])


def validate_citations(
    citations: list[Citation], context: str | BuiltPrompt
) -> list[Citation]:
    """Validate *citations* against *context*, filling in `.valid`/`.snippet`.

    A citation is valid when its file path matches a block actually shown
    in *context* **and** its line range is contained within (not
    necessarily equal to) that block's shown range -- see the module
    docstring for why exact equality would be too strict. When a file was
    split across multiple retrieved blocks, a citation is checked against
    all of them; it only needs to fit inside one.

    Args:
        citations: Citations to validate, typically from
            :func:`extract_citations`. Not mutated -- a new list of
            :class:`Citation` is returned.
        context: The code-context string the model was shown, or the
            :class:`~reporag.generation.prompt_builder.BuiltPrompt` it came
            from.

    Returns:
        A new list of :class:`Citation`, same order, each with `.valid` set
        and `.snippet` populated for valid citations (empty for invalid
        ones -- there is no source to show for a range that was never
        retrieved).
    """
    context_text = _resolve_context_text(context)
    blocks_by_file: dict[str, list[_ContextBlock]] = {}
    for block in _parse_context_blocks(context_text):
        blocks_by_file.setdefault(block.file_path, []).append(block)

    validated = []
    for citation in citations:
        matching_block = next(
            (
                b
                for b in blocks_by_file.get(citation.file_path, [])
                if _is_contained(citation, b)
            ),
            None,
        )
        if matching_block is None:
            validated.append(
                Citation(
                    file_path=citation.file_path,
                    start_line=citation.start_line,
                    end_line=citation.end_line,
                    snippet="",
                    valid=False,
                    raw=citation.raw,
                )
            )
        else:
            validated.append(
                Citation(
                    file_path=citation.file_path,
                    start_line=citation.start_line,
                    end_line=citation.end_line,
                    snippet=_snippet_for(citation, matching_block),
                    valid=True,
                    raw=citation.raw,
                )
            )
    return validated


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def _split_into_candidate_claims(text: str) -> list[str]:
    """Split *text* into naive sentence-level claim candidates.

    Two split points are used, applied in order: a newline that starts a
    new bullet or numbered-list item (:data:`_LIST_ITEM_BOUNDARY_RE` --
    without this, a list with no terminal punctuation between items reads
    as one run-on claim, badly distorting coverage for exactly the bulleted
    "one short section per component" shape the exploratory template's own
    answer-shape instructions ask for), and sentence-ending punctuation
    followed by whitespace (skipping a decimal-like ``3.5``). A
    citation-only trailing fragment is then merged back into the claim it
    follows -- so ``"It does X. [a.py:1-5]."`` becomes one claim, not a
    claim plus a separate fragment that would otherwise look like its own,
    uncited "claim". A fragment is citation-only when stripping every
    citation marker out of it leaves fewer than :data:`_MIN_CLAIM_LENGTH`
    characters.

    This is a line-count-of-code-simple heuristic, not real sentence
    segmentation -- see :func:`compute_citation_coverage` for its
    documented limitations.
    """
    if not text or not text.strip():
        return []

    raw_sentences: list[str] = []
    for list_segment in _LIST_ITEM_BOUNDARY_RE.split(text.strip()):
        raw_sentences.extend(
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+(?!\d)", list_segment.strip())
            if s.strip()
        )

    merged: list[str] = []
    for sentence in raw_sentences:
        stripped = _CITATION_RE.sub("", sentence).strip()
        if merged and len(stripped) < _MIN_CLAIM_LENGTH:
            merged[-1] = f"{merged[-1]} {sentence}"
        else:
            merged.append(sentence)
    return merged


def compute_citation_coverage(text: str) -> float:
    """Estimate what fraction of *text*'s claims carry a citation.

    **This is a heuristic, not an exact measurement** -- "claim" has no
    formal definition here. The approximation used: split *text* into
    naive sentences (see :func:`_split_into_candidate_claims`), discard
    ones shorter than :data:`_MIN_CLAIM_LENGTH` characters after stripping
    citation markers (structural noise -- a bare list number, a short
    heading -- rather than a claim), and report the fraction of the
    remainder that contains at least one citation marker. It will
    undercount answers that front-load one citation covering several
    following sentences, and overcount answers that cite trivially on
    every short sentence; it is meant as a rough signal for the target in
    Issue 25's acceptance criteria (aiming for roughly 90% coverage on
    average across many answers), not a per-answer pass/fail gate.

    Args:
        text: The LLM's answer text.

    Returns:
        A float in ``[0.0, 1.0]``. ``1.0`` (vacuously) when there are no
        substantive claims to cover at all.
    """
    claims = []
    for candidate in _split_into_candidate_claims(text):
        stripped = _CITATION_RE.sub("", candidate).strip()
        if len(stripped) >= _MIN_CLAIM_LENGTH:
            claims.append(candidate)

    if not claims:
        return 1.0

    cited = sum(1 for c in claims if _CITATION_RE.search(c))
    return cited / len(claims)


# ---------------------------------------------------------------------------
# End-to-end convenience
# ---------------------------------------------------------------------------


def analyze_citations(answer_text: str, context: str | BuiltPrompt) -> CitationReport:
    """Extract, validate, and score every citation in *answer_text* in one call.

    Args:
        answer_text: The LLM's answer text.
        context: The code-context string the model was shown, or the
            :class:`~reporag.generation.prompt_builder.BuiltPrompt` it came
            from.

    Returns:
        A :class:`CitationReport`.
    """
    citations = validate_citations(extract_citations(answer_text), context)
    valid_count = sum(1 for c in citations if c.valid)
    return CitationReport(
        citations=citations,
        coverage=compute_citation_coverage(answer_text),
        valid_count=valid_count,
        invalid_count=len(citations) - valid_count,
    )
