"""Line-level citation extraction and validation (Issue 25).

Parses ``[file_path:start_line-end_line]`` markers out of an LLM answer,
resolves each one against the code that was actually retrieved, and reports
which claims in the answer are backed by a citation.

Why
---
A cited answer is only worth more than an uncited one if the citations are
real.  An LLM that invents ``[src/auth/service.py:88-107]`` for a file that
was never retrieved produces an answer that *looks* grounded and is not --
the exact failure Issue 35 measures as hallucination.  So every marker is
checked against the retrieved context, and the ones that cannot be backed
are flagged rather than dropped: a caller that cannot see the invalid
citations cannot tell a grounded answer from a confident guess.

The second half of the module answers the issue's coverage criterion (">= 90%
of claims have at least one citation").  That needs a definition of "claim",
so the answer is split into sentence-level claims -- ignoring fenced code,
headings, and marker-only fragments -- and each claim is matched to the
citations that support it.

Design
------
The layout mirrors :mod:`reporag.generation.prompt_builder` and
:mod:`reporag.agent.planner`, so the generation package reads consistently:

* **Pure and offline** -- no LLM, no network, no I/O.  Everything here is a
  free function or a small value object, so each behaviour is testable
  without constructing a generator.
* **One parser, one instruction** -- :data:`CITATION_FORMAT` is imported
  from the prompt builder rather than restated, so the shape this module
  parses is by construction the shape the prompt asked for.  If Issue 24
  changes the instruction, this module cannot silently drift from it.
* **Validate against what was sent, not what was retrieved** -- the prompt
  builder may drop chunks to fit the context window.  A citation to a chunk
  that was retrieved but trimmed out of the prompt is still a hallucination
  from the model's point of view, so :meth:`ContextIndex.from_context`
  builds the index from the assembled context block itself.
  :func:`build_context_index` accepts either form.
* **Tolerant where the model is sloppy, strict where it matters** -- a bare
  ``[auth.py:90]`` is resolved to the one retrieved path that ends in
  ``auth.py`` and marked as resolved; two candidate paths make it
  ``ambiguous-file`` instead of a coin flip.  Line ranges are never
  widened: a range that runs past the retrieved lines is invalid even
  though the file is right.

Citation statuses
-----------------
``verified``
    The file was retrieved and the cited lines lie inside the retrieved
    line ranges for that file.
``unverified``
    No context was supplied (or the file was retrieved with unknown line
    bounds, as documentation chunks are), so the marker could not be
    checked.  Counted as valid -- an unproven citation is not a proven
    fake.
``unknown-file``
    No retrieved chunk comes from that file.
``ambiguous-file``
    A partial path (``auth.py``) matches more than one retrieved file.
``line-range-outside-context``
    The file is right but the cited lines fall outside every retrieved
    range for it.
``inverted-range``
    ``start_line > end_line``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from reporag.generation.prompt_builder import CITATION_FORMAT

logger = logging.getLogger(__name__)

__all__ = [
    "CITATION_FORMAT",
    "CITATION_MARKER_RE",
    "MIN_CITATION_COVERAGE",
    "Citation",
    "CitationMarker",
    "CitationReport",
    "CitationStatus",
    "Claim",
    "ContextIndex",
    "SourceSpan",
    "build_context_index",
    "extract_citations",
    "find_citation_markers",
    "mask_code_blocks",
    "split_claims",
]


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

MIN_CITATION_COVERAGE = 0.9
"""Issue 25's target: at least 90% of claims carry a valid citation."""

CitationStatus = Literal[
    "verified",
    "unverified",
    "unknown-file",
    "ambiguous-file",
    "line-range-outside-context",
    "inverted-range",
]
"""Outcome of checking one marker against the retrieved context."""

_VALID_STATUSES: frozenset[str] = frozenset({"verified", "unverified"})

# Matches ``[path:12-34]``, ``[path:12]``, and the ``L``-prefixed spelling
# GitHub uses (``[path:L12-L34]``).  Whitespace around the separators is
# tolerated because models add it.
#
# The path may not contain brackets or colons, and must contain a ``/`` or a
# ``.`` -- that one requirement is what keeps ordinary prose in brackets
# ("[Note: 3]", "[Step: 2]") from being read as a citation, without needing
# a list of known file extensions.
CITATION_MARKER_RE = re.compile(
    r"\[\s*(?P<path>[^\[\]:\n]*[./][^\[\]:\n]*?)\s*:\s*"
    r"L?(?P<start>\d+)"
    r"(?:\s*-\s*L?(?P<end>\d+))?\s*\]"
)

# Sentence terminator followed by the start of something that looks like a
# new sentence.  Requiring an upper-case letter, a digit, or an opening
# bracket after the space is what keeps "defined in auth.py and called by"
# from splitting mid-sentence, while "... in auth.py. The route then ..."
# still splits.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])["\')\]]*\s+(?=[A-Z0-9("\[`])')

# Abbreviations that end in a period without ending a sentence.
_ABBREVIATIONS: frozenset[str] = frozenset(
    {"e.g.", "i.e.", "etc.", "vs.", "cf.", "approx.", "fig.", "no.", "ref."}
)

_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_BLOCKQUOTE_RE = re.compile(r"^\s*>+\s?")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s")
_HORIZONTAL_RULE_RE = re.compile(r"^\s*(?:[-*_]\s*){3,}$")
_FENCE_RE = re.compile(r"^\s*(?:`{3,}|~{3,})")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

# Matches the headers ContextAssembler emits: ``## path/to/file.py (lines 5-6)``.
# ``?`` is accepted for a bound because the assembler renders it for chunks
# whose line numbers are unknown (documentation, README sections).
_CONTEXT_HEADER_RE = re.compile(
    r"^##[ \t]+(?P<path>.+?)[ \t]+\(lines[ \t]+(?P<start>\d+|\?)-(?P<end>\d+|\?)\)[ \t]*$",
    re.MULTILINE,
)

# A claim needs at least this many words to be worth scoring.  Below it the
# fragment is a heading, a label, or a bare citation marker -- none of which
# are assertions about the code.
_MIN_CLAIM_WORDS = 3


# ---------------------------------------------------------------------------
# Retrieved context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpan:
    """One retrieved region of a file that a citation may legitimately name.

    Attributes:
        file_path: The path as it appears in the retrieved context.
        start_line: First line of the region, or ``None`` when the chunk
            carries no line bounds (documentation chunks).
        end_line: Last line of the region, inclusive, or ``None``.
        text: The chunk body, used to cut the snippet for a citation.
    """

    file_path: str
    start_line: int | None = None
    end_line: int | None = None
    text: str = ""

    @property
    def has_line_bounds(self) -> bool:
        """``True`` when both bounds are known, so the span can be checked."""
        return self.start_line is not None and self.end_line is not None


def _normalize_path(path: str) -> str:
    """Normalize a file path for comparison, without resolving it on disk."""
    cleaned = (path or "").strip().strip("`\"'")
    cleaned = cleaned.replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    while "//" in cleaned:
        cleaned = cleaned.replace("//", "/")
    return cleaned.strip("/")


def _merge_ranges(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping and adjacent ``(start, end)`` ranges.

    Merging matters for validation: a file retrieved as lines 1-10 and 11-20
    genuinely contains lines 5-15, even though no single chunk does.
    """
    if not ranges:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


class ContextIndex:
    """The code that was put in front of the model, indexed for lookup.

    Answers the three questions validation needs: was this file retrieved,
    were these lines retrieved, and what code is at those lines.

    Args:
        spans: The retrieved regions.  Order is preserved per file.
    """

    def __init__(self, spans: Iterable[SourceSpan] = ()) -> None:
        self._spans: dict[str, list[SourceSpan]] = {}
        self._merged: dict[str, list[tuple[int, int]]] | None = None
        for span in spans:
            self.add(span)

    # -- construction ---------------------------------------------------

    def add(self, span: SourceSpan) -> None:
        """Add one retrieved region to the index."""
        path = _normalize_path(span.file_path)
        if not path:
            return
        if path != span.file_path:
            span = SourceSpan(path, span.start_line, span.end_line, span.text)
        self._spans.setdefault(path, []).append(span)
        self._merged = None

    @classmethod
    def from_results(cls, results: Iterable[Any]) -> ContextIndex:
        """Build an index from retrieval results.

        Accepts :class:`~reporag.retrieval.vector_search.RetrievalResult`
        objects, mappings, :class:`SourceSpan` objects, or anything
        duck-typed with ``file_path`` / ``chunk_text``.  Items that carry no
        file path are skipped with a debug log rather than raising -- a
        malformed chunk should cost one citation's verification, not the
        whole answer.
        """
        index = cls()
        for item in results:
            span = _coerce_span(item)
            if span is None:
                logger.debug("ContextIndex: skipping unusable chunk %r", item)
                continue
            index.add(span)
        return index

    @classmethod
    def from_context(cls, context: str) -> ContextIndex:
        """Build an index from an assembled context block.

        Parses the ``## path (lines a-b)`` headers and fenced bodies that
        :class:`~reporag.generation.context_assembler.ContextAssembler`
        emits.  This is the form to prefer when a prompt has already been
        built: it indexes exactly the chunks that survived the prompt
        builder's context-window trimming, so a citation to a retrieved but
        dropped chunk is correctly reported as unbacked.
        """
        index = cls()
        text = context or ""
        matches = list(_CONTEXT_HEADER_RE.finditer(text))
        for position, match in enumerate(matches):
            body_start = match.end()
            body_end = (
                matches[position + 1].start()
                if position + 1 < len(matches)
                else len(text)
            )
            index.add(
                SourceSpan(
                    file_path=match.group("path"),
                    start_line=_optional_int(match.group("start")),
                    end_line=_optional_int(match.group("end")),
                    text=_strip_code_fence(text[body_start:body_end]),
                )
            )
        return index

    # -- introspection --------------------------------------------------

    @property
    def files(self) -> tuple[str, ...]:
        """Every retrieved file path, in first-seen order."""
        return tuple(self._spans)

    @property
    def is_empty(self) -> bool:
        """``True`` when nothing was indexed, so nothing can be verified."""
        return not self._spans

    def spans_for(self, file_path: str) -> tuple[SourceSpan, ...]:
        """Every retrieved region of *file_path*."""
        return tuple(self._spans.get(_normalize_path(file_path), ()))

    def covered_ranges(self, file_path: str) -> list[tuple[int, int]]:
        """The merged line ranges retrieved for *file_path*."""
        if self._merged is None:
            self._merged = {
                path: _merge_ranges(
                    [
                        (span.start_line, span.end_line)
                        for span in spans
                        if span.has_line_bounds
                    ]
                )
                for path, spans in self._spans.items()
            }
        return self._merged.get(_normalize_path(file_path), [])

    # -- resolution and validation --------------------------------------

    def resolve_path(self, path: str) -> tuple[str | None, CitationStatus]:
        """Resolve a cited path against the retrieved files.

        Tries, in order: the path as written, a unique retrieved path ending
        in it (so ``auth/service.py`` finds ``src/app/auth/service.py``), and
        a unique retrieved path with the same file name.  Each fallback is
        only taken when exactly one candidate matches; more than one is
        ``ambiguous-file`` rather than a guess.

        Returns:
            ``(resolved_path, "verified")`` on success, otherwise
            ``(None, "unknown-file")`` or ``(None, "ambiguous-file")``.
        """
        wanted = _normalize_path(path)
        if not wanted:
            return None, "unknown-file"
        if wanted in self._spans:
            return wanted, "verified"

        suffix_matches = [
            candidate
            for candidate in self._spans
            if candidate.endswith("/" + wanted) or wanted.endswith("/" + candidate)
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0], "verified"
        if len(suffix_matches) > 1:
            return None, "ambiguous-file"

        name = wanted.rsplit("/", 1)[-1]
        name_matches = [
            candidate
            for candidate in self._spans
            if candidate.rsplit("/", 1)[-1] == name
        ]
        if len(name_matches) == 1:
            return name_matches[0], "verified"
        if len(name_matches) > 1:
            return None, "ambiguous-file"
        return None, "unknown-file"

    def snippet(self, file_path: str, start_line: int, end_line: int) -> str:
        """Return the retrieved code for ``file_path`` lines *start*-*end*.

        The slice is taken from the span that contains the range, using the
        span's own start line as the offset.  When the chunk body does not
        line up with its declared bounds (a chunk stored with a header line,
        say) the whole span text is returned rather than a wrong slice --
        showing too much beats showing the wrong lines.
        """
        for span in self.spans_for(file_path):
            if not span.has_line_bounds or not span.text:
                continue
            if start_line < span.start_line or end_line > span.end_line:
                continue
            lines = span.text.split("\n")
            declared = span.end_line - span.start_line + 1
            if len(lines) != declared:
                return span.text
            first = start_line - span.start_line
            return "\n".join(lines[first : first + (end_line - start_line + 1)])
        return ""

    def validate(self, marker: CitationMarker) -> Citation:
        """Check one parsed marker and return the resulting :class:`Citation`."""
        if marker.start_line > marker.end_line:
            return marker.to_citation(
                file_path=_normalize_path(marker.raw_path),
                status="inverted-range",
            )

        if self.is_empty:
            return marker.to_citation(
                file_path=_normalize_path(marker.raw_path),
                status="unverified",
            )

        resolved, status = self.resolve_path(marker.raw_path)
        if resolved is None:
            return marker.to_citation(
                file_path=_normalize_path(marker.raw_path), status=status
            )

        ranges = self.covered_ranges(resolved)
        if not ranges:
            # The file was retrieved, but with no usable line bounds (a
            # documentation chunk).  The path checks out; the lines cannot
            # be checked either way.
            return marker.to_citation(file_path=resolved, status="unverified")

        contained = any(
            start <= marker.start_line and marker.end_line <= end
            for start, end in ranges
        )
        if not contained:
            return marker.to_citation(
                file_path=resolved, status="line-range-outside-context"
            )

        return marker.to_citation(
            file_path=resolved,
            status="verified",
            snippet=self.snippet(resolved, marker.start_line, marker.end_line),
        )

    def __repr__(self) -> str:
        return (
            f"ContextIndex(files={len(self._spans)}, "
            f"spans={sum(len(s) for s in self._spans.values())})"
        )


def _optional_int(raw: str) -> int | None:
    """Parse a line bound that the assembler may have rendered as ``?``."""
    return int(raw) if raw.isdigit() else None


def _strip_code_fence(body: str) -> str:
    """Return the code inside a fenced block, without the fences."""
    lines = body.strip("\n").split("\n")
    if lines and _FENCE_RE.match(lines[0]):
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and _FENCE_RE.match(lines[-1]):
        lines.pop()
    return "\n".join(lines)


def _coerce_span(item: Any) -> SourceSpan | None:
    """Coerce one chunk-ish object into a :class:`SourceSpan`.

    Accepts a :class:`SourceSpan`, a mapping, or any object with the
    attributes a ``RetrievalResult`` has.  Returns ``None`` when no file
    path can be found, which is the one thing an index entry cannot do
    without.
    """
    if isinstance(item, SourceSpan):
        return item

    if isinstance(item, Mapping):
        getter = item.get
    else:

        def getter(key: str, default: Any = None) -> Any:
            return getattr(item, key, default)

    path = getter("file_path") or getter("path") or getter("file")
    if not isinstance(path, str) or not path.strip():
        return None

    text = (
        getter("chunk_text")
        or getter("text")
        or getter("snippet")
        or getter("code")
        or ""
    )
    return SourceSpan(
        file_path=path,
        start_line=_as_optional_int(getter("start_line", getter("start"))),
        end_line=_as_optional_int(getter("end_line", getter("end"))),
        text=text if isinstance(text, str) else "",
    )


def _as_optional_int(value: Any) -> int | None:
    """Coerce a line bound to ``int``, tolerating strings and ``None``."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def build_context_index(chunks: Any) -> ContextIndex:
    """Build a :class:`ContextIndex` from whatever the caller has.

    Accepts an existing index, an assembled context string, a built prompt
    (its context section is used), a sequence of retrieval results, spans or
    mappings, or ``None``.  Anything unrecognized yields an empty index and
    a warning: validation is then skipped, and every citation is reported as
    ``unverified`` rather than falsely flagged.
    """
    if chunks is None:
        return ContextIndex()
    if isinstance(chunks, ContextIndex):
        return chunks
    if isinstance(chunks, str):
        return ContextIndex.from_context(chunks)

    sections = getattr(chunks, "sections", None)
    if isinstance(sections, Mapping):
        return ContextIndex.from_context(str(sections.get("context", "")))

    if isinstance(chunks, Mapping):
        return ContextIndex.from_results([chunks])
    if isinstance(chunks, Sequence):
        return ContextIndex.from_results(chunks)
    if isinstance(chunks, Iterable):
        return ContextIndex.from_results(list(chunks))

    logger.warning(
        "build_context_index: cannot index %s; citations will be unverified.",
        type(chunks).__name__,
    )
    return ContextIndex()


# ---------------------------------------------------------------------------
# Markers and citations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CitationMarker:
    """One ``[path:a-b]`` marker as it was written in the answer.

    Attributes:
        raw_path: The path exactly as the model wrote it.
        start_line: First cited line.
        end_line: Last cited line, inclusive.  Equal to *start_line* for a
            single-line ``[path:12]`` marker.
        marker: The matched text, useful for rewriting the answer.
        start_offset: Character offset of the marker in the answer.
        end_offset: Character offset just past the marker.
    """

    raw_path: str
    start_line: int
    end_line: int
    marker: str
    start_offset: int
    end_offset: int

    def to_citation(
        self,
        *,
        file_path: str,
        status: CitationStatus,
        snippet: str = "",
    ) -> Citation:
        """Build the :class:`Citation` this marker resolved to."""
        return Citation(
            file_path=file_path,
            start_line=self.start_line,
            end_line=self.end_line,
            snippet=snippet,
            status=status,
            marker=self.marker,
            raw_path=self.raw_path,
            start_offset=self.start_offset,
            end_offset=self.end_offset,
        )


@dataclass(frozen=True)
class Citation:
    """A citation parsed from an answer and checked against the context.

    Attributes:
        file_path: The resolved repository path.  Equal to the model's own
            spelling unless a partial path had to be resolved.
        start_line: First cited line.
        end_line: Last cited line, inclusive.
        snippet: The retrieved code at those lines; empty when the citation
            could not be verified.
        status: Why this citation is (in)valid -- see the module docstring.
        marker: The exact marker text found in the answer.
        raw_path: The path as the model wrote it, before resolution.
        start_offset: Character offset of the marker in the answer, so a UI
            can turn it into a link without re-parsing (Issue 32).
        end_offset: Character offset just past the marker.
    """

    file_path: str
    start_line: int
    end_line: int
    snippet: str = ""
    status: CitationStatus = "verified"
    marker: str = ""
    raw_path: str = ""
    start_offset: int = -1
    end_offset: int = -1

    @property
    def valid(self) -> bool:
        """``True`` unless the citation was proven wrong against the context."""
        return self.status in _VALID_STATUSES

    @property
    def verified(self) -> bool:
        """``True`` only when the file and the line range were both checked."""
        return self.status == "verified"

    @property
    def was_resolved(self) -> bool:
        """``True`` when the model's path had to be resolved to a real one."""
        return bool(self.raw_path) and _normalize_path(self.raw_path) != self.file_path

    @property
    def line_count(self) -> int:
        """Number of lines the citation covers."""
        return max(0, self.end_line - self.start_line + 1)

    def as_marker(self) -> str:
        """Render this citation in the canonical citation format."""
        return f"[{self.file_path}:{self.start_line}-{self.end_line}]"

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready view, for the Issue 26 query endpoint."""
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "snippet": self.snippet,
            "valid": self.valid,
            "status": self.status,
            "marker": self.marker,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
        }

    def __str__(self) -> str:
        return self.as_marker()


def mask_code_blocks(text: str) -> str:
    """Blank out fenced code blocks, preserving every character offset.

    Citations are parsed from the masked text so that a ``[path:1-2]``
    *inside* a code sample -- the model showing what a citation looks like,
    or quoting a log line -- is not counted as a citation of the repository.
    Blanking rather than deleting keeps every offset in the masked text
    equal to its offset in the original.
    """
    if not text:
        return ""
    masked: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        is_fence = bool(_FENCE_RE.match(line))
        if is_fence or in_fence:
            masked.append(" " * len(line))
        else:
            masked.append(line)
        if is_fence:
            in_fence = not in_fence
    return "\n".join(masked)


def find_citation_markers(text: str) -> list[CitationMarker]:
    """Find every citation marker in *text*, in order of appearance.

    Fenced code blocks are ignored (see :func:`mask_code_blocks`).  A marker
    with a single line number (``[path:12]``) is read as the one-line range
    ``12-12``.
    """
    markers: list[CitationMarker] = []
    for match in CITATION_MARKER_RE.finditer(mask_code_blocks(text)):
        start_line = int(match.group("start"))
        end_raw = match.group("end")
        markers.append(
            CitationMarker(
                raw_path=match.group("path").strip(),
                start_line=start_line,
                end_line=int(end_raw) if end_raw else start_line,
                marker=match.group(0),
                start_offset=match.start(),
                end_offset=match.end(),
            )
        )
    return markers


# ---------------------------------------------------------------------------
# Claims and coverage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    """One assertion in the answer, and the citations that back it.

    Attributes:
        text: The sentence, with list markers and blockquote prefixes
            stripped.
        start_offset: Character offset of the claim in the answer.
        end_offset: Character offset just past the claim.
        citations: The citations attached to this claim.
    """

    text: str
    start_offset: int
    end_offset: int
    citations: tuple[Citation, ...] = ()

    @property
    def supported(self) -> bool:
        """``True`` when at least one attached citation is valid."""
        return any(citation.valid for citation in self.citations)

    @property
    def has_citation(self) -> bool:
        """``True`` when the claim carries any marker, valid or not."""
        return bool(self.citations)


def _iter_line_spans(text: str) -> Iterator[tuple[str, int]]:
    """Yield each line of *text* with its absolute start offset."""
    offset = 0
    for line in text.split("\n"):
        yield line, offset
        offset += len(line) + 1


def _ends_with_abbreviation(fragment: str) -> bool:
    """``True`` when *fragment* ends in a known non-terminal abbreviation."""
    tail = fragment.rstrip().rsplit(" ", 1)[-1].lower()
    return tail in _ABBREVIATIONS


def _split_sentences(line: str, base_offset: int) -> list[tuple[str, int, int]]:
    """Split one line into ``(text, start_offset, end_offset)`` sentences."""
    pieces: list[tuple[str, int, int]] = []
    cursor = 0
    for match in _SENTENCE_SPLIT_RE.finditer(line):
        candidate = line[cursor : match.start()]
        if _ends_with_abbreviation(candidate):
            continue
        pieces.append((candidate, base_offset + cursor, base_offset + match.start()))
        cursor = match.end()
    pieces.append((line[cursor:], base_offset + cursor, base_offset + len(line)))
    return pieces


def _is_claim(text: str) -> bool:
    """``True`` when a fragment asserts something worth citing.

    Fragments below :data:`_MIN_CLAIM_WORDS` words are headings, labels, or
    bare citation markers -- scoring them would drag coverage down for
    formatting rather than for missing evidence.
    """
    stripped = CITATION_MARKER_RE.sub(" ", text)
    return len(_WORD_RE.findall(stripped)) >= _MIN_CLAIM_WORDS


def split_claims(text: str) -> list[Claim]:
    """Split an answer into the claims that citation coverage is scored over.

    Fenced code blocks, headings, horizontal rules, and fragments too short
    to assert anything are excluded.  List items are separate claims, since
    a bulleted walkthrough makes one assertion per bullet.  Offsets are
    absolute in *text*, so a caller can highlight an uncited claim in place.
    """
    if not text:
        return []
    masked = mask_code_blocks(text)
    claims: list[Claim] = []
    for line, line_offset in _iter_line_spans(masked):
        if (
            not line.strip()
            or _HEADING_RE.match(line)
            or _HORIZONTAL_RULE_RE.match(line)
        ):
            continue

        content_start = 0
        blockquote = _BLOCKQUOTE_RE.match(line)
        if blockquote:
            content_start = blockquote.end()
        marker = _LIST_MARKER_RE.match(line[content_start:])
        if marker:
            content_start += marker.end()

        body = line[content_start:]
        for sentence, start, _end in _split_sentences(
            body, line_offset + content_start
        ):
            stripped = sentence.strip()
            if not stripped or not _is_claim(stripped):
                continue
            lead = len(sentence) - len(sentence.lstrip())
            claims.append(
                Claim(
                    text=stripped,
                    start_offset=start + lead,
                    end_offset=start + lead + len(stripped),
                )
            )
    return claims


def _attach_citations(
    claims: Sequence[Claim], citations: Sequence[Citation]
) -> tuple[Claim, ...]:
    """Attach each citation to the claim it supports.

    A citation belongs to the last claim that starts at or before it.  That
    covers both spellings models use -- the marker inside the sentence, and
    the marker trailing after the full stop -- without needing to guess
    which one this model prefers.  A citation before the first claim is
    attached to the first claim.
    """
    if not claims:
        return ()
    attached: dict[int, list[Citation]] = {}
    for citation in citations:
        chosen = 0
        for position, claim in enumerate(claims):
            if claim.start_offset <= citation.start_offset:
                chosen = position
            else:
                break
        attached.setdefault(chosen, []).append(citation)
    return tuple(
        Claim(
            text=claim.text,
            start_offset=claim.start_offset,
            end_offset=claim.end_offset,
            citations=tuple(attached.get(position, ())),
        )
        for position, claim in enumerate(claims)
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CitationReport:
    """Everything known about the citations in one answer.

    Attributes:
        citations: Every marker found, in order, valid or not.
        claims: The answer's claims, each carrying its citations.
        context_available: ``False`` when no retrieved context was supplied,
            in which case every citation is ``unverified``.
        metadata: Free-form extras (indexed file count, and so on).
    """

    citations: tuple[Citation, ...] = ()
    claims: tuple[Claim, ...] = ()
    context_available: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def valid_citations(self) -> tuple[Citation, ...]:
        """Citations that were not proven wrong against the context."""
        return tuple(citation for citation in self.citations if citation.valid)

    @property
    def invalid_citations(self) -> tuple[Citation, ...]:
        """Citations flagged as unbacked -- the hallucinated ones."""
        return tuple(citation for citation in self.citations if not citation.valid)

    @property
    def unique_citations(self) -> tuple[Citation, ...]:
        """Valid citations deduplicated by ``(file_path, start, end)``."""
        seen: set[tuple[str, int, int]] = set()
        unique: list[Citation] = []
        for citation in self.valid_citations:
            key = (citation.file_path, citation.start_line, citation.end_line)
            if key not in seen:
                seen.add(key)
                unique.append(citation)
        return tuple(unique)

    @property
    def cited_files(self) -> tuple[str, ...]:
        """Distinct files backed by a valid citation, in first-cited order."""
        ordered: list[str] = []
        for citation in self.valid_citations:
            if citation.file_path not in ordered:
                ordered.append(citation.file_path)
        return tuple(ordered)

    @property
    def coverage(self) -> float:
        """Share of claims carrying at least one valid citation, in ``[0, 1]``.

        An answer with no claims scores ``1.0``: there is nothing to
        support, so there is nothing missing.
        """
        if not self.claims:
            return 1.0
        supported = sum(1 for claim in self.claims if claim.supported)
        return supported / len(self.claims)

    @property
    def uncited_claims(self) -> tuple[Claim, ...]:
        """Claims with no valid citation -- where to look when coverage drops."""
        return tuple(claim for claim in self.claims if not claim.supported)

    @property
    def meets_coverage_target(self) -> bool:
        """``True`` when coverage reaches :data:`MIN_CITATION_COVERAGE`."""
        return self.coverage >= MIN_CITATION_COVERAGE

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready view, for the Issue 26 query endpoint."""
        return {
            "citations": [citation.to_dict() for citation in self.unique_citations],
            "invalid_citations": [
                citation.to_dict() for citation in self.invalid_citations
            ],
            "citation_coverage": self.coverage,
            "claim_count": len(self.claims),
            "context_available": self.context_available,
        }

    def __repr__(self) -> str:
        return (
            f"CitationReport(citations={len(self.citations)}, "
            f"valid={len(self.valid_citations)}, claims={len(self.claims)}, "
            f"coverage={self.coverage:.2f})"
        )


def extract_citations(text: str, chunks: Any = None) -> CitationReport:
    """Extract, validate, and score the citations in an LLM answer.

    Args:
        text: The answer text as the model produced it.
        chunks: The retrieved context to validate against -- retrieval
            results, an assembled context string, a
            :class:`~reporag.generation.prompt_builder.BuiltPrompt`, a
            prepared :class:`ContextIndex`, or ``None`` to skip validation.

    Returns:
        A :class:`CitationReport` with every marker found, the answer's
        claims with their supporting citations attached, and the coverage
        ratio.
    """
    index = build_context_index(chunks)
    citations = tuple(index.validate(marker) for marker in find_citation_markers(text))
    claims = _attach_citations(split_claims(text), citations)
    return CitationReport(
        citations=citations,
        claims=claims,
        context_available=not index.is_empty,
        metadata={"indexed_files": len(index.files)},
    )
