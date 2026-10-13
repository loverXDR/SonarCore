"""Word (.docx) rendering of summaries and meeting minutes"""

import io
import re
from datetime import datetime
from typing import Dict, List, NamedTuple, Optional

from docx import Document as DocxDocument
from docx.document import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph

from Core.Schemas import MeetingMinutes, SummaryOptions, SummaryStyle

_FONT_NAME = "Calibri"
_FONT_SIZE = Pt(11)
_ACCENT_COLOR = RGBColor(0x1F, 0x3A, 0x5F)
_HEADER_FILL = "D9E2F3"
_MUTED_COLOR = RGBColor(0x6B, 0x6B, 0x6B)

# "[SPEAKER_00]: text" lines produced by the pipeline when diarization is on
_SPEAKER_LINE = re.compile(r"^\[(?P<speaker>[^\]]+)\]:\s*(?P<text>.*)$")
_HEADING_LINE = re.compile(r"^(?P<level>#{1,6})\s+(?P<text>.+)$")
_BULLET_LINE = re.compile(r"^\s*[-*•]\s+(?P<text>.+)$")
_NUMBERED_LINE = re.compile(r"^\s*(?P<num>\d+)[.)]\s+(?P<text>.+)$")
_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")
_BOLD = re.compile(r"\*\*(.+?)\*\*")


class _Labels(NamedTuple):
    """Localized document captions"""

    minutes_title: str
    summary_title: str
    styles: Dict[SummaryStyle, str]
    date: str
    session: str
    participants_count: str
    summary: str
    participants: str
    speaker: str
    name: str
    role: str
    agenda: str
    discussion: str
    decisions: str
    action_items: str
    number: str
    task: str
    owner: str
    deadline: str
    open_questions: str
    transcript: str
    not_specified: str
    page: str
    of: str


_RU = _Labels(
    minutes_title="Протокол встречи",
    summary_title="Саммари",
    styles={
        SummaryStyle.BRIEF: "Краткое саммари",
        SummaryStyle.DETAILED: "Подробный обзор",
        SummaryStyle.BULLET_POINTS: "Ключевые тезисы",
        SummaryStyle.MEETING_MINUTES: "Протокол встречи",
    },
    date="Дата",
    session="Сессия",
    participants_count="Участников",
    summary="Краткое содержание",
    participants="Участники",
    speaker="Метка",
    name="Имя",
    role="Роль",
    agenda="Повестка",
    discussion="Ход обсуждения",
    decisions="Принятые решения",
    action_items="Поручения",
    number="№",
    task="Задача",
    owner="Ответственный",
    deadline="Срок",
    open_questions="Открытые вопросы",
    transcript="Приложение. Стенограмма",
    not_specified="не указано",
    page="Стр.",
    of="из",
)

_EN = _Labels(
    minutes_title="Meeting minutes",
    summary_title="Summary",
    styles={
        SummaryStyle.BRIEF: "Brief summary",
        SummaryStyle.DETAILED: "Detailed review",
        SummaryStyle.BULLET_POINTS: "Key points",
        SummaryStyle.MEETING_MINUTES: "Meeting minutes",
    },
    date="Date",
    session="Session",
    participants_count="Participants",
    summary="Summary",
    participants="Participants",
    speaker="Label",
    name="Name",
    role="Role",
    agenda="Agenda",
    discussion="Discussion",
    decisions="Decisions",
    action_items="Action items",
    number="#",
    task="Task",
    owner="Owner",
    deadline="Deadline",
    open_questions="Open questions",
    transcript="Appendix. Transcript",
    not_specified="not specified",
    page="Page",
    of="of",
)

_LABELS: Dict[str, _Labels] = {"ru": _RU, "en": _EN}


def _get_labels(language: str) -> _Labels:
    """Return document captions for a language, falling back to English"""
    return _LABELS.get(language.lower(), _EN)


def _new_document() -> Document:
    """Create a document with base fonts, margins and heading colors"""
    doc = DocxDocument()

    section = doc.sections[0]
    section.top_margin = section.bottom_margin = Cm(2)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(1.5)

    normal = doc.styles["Normal"]
    normal.font.name = _FONT_NAME
    normal.font.size = _FONT_SIZE
    # East Asian font slot keeps Cyrillic from falling back to another font
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), _FONT_NAME)
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.line_spacing = 1.15

    for level, size in ((1, 16), (2, 13), (3, 12)):
        style = doc.styles[f"Heading {level}"]
        style.font.name = _FONT_NAME
        style.font.size = Pt(size)
        style.font.color.rgb = _ACCENT_COLOR
        style.paragraph_format.space_before = Pt(14 if level == 1 else 10)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.keep_with_next = True

    return doc


def _add_field(paragraph: Paragraph, instruction: str) -> None:
    """Insert a Word field (PAGE, NUMPAGES, ...) into the paragraph"""
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, end])


def _add_footer(doc: Document, labels: _Labels, caption: str) -> None:
    """Add a footer with the document caption and page numbers"""
    footer = doc.sections[0].footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.style.font.size = Pt(9)
    footer.add_run(f"{caption} · {labels.page} ")
    _add_field(footer, "PAGE")
    footer.add_run(f" {labels.of} ")
    _add_field(footer, "NUMPAGES")
    for run in footer.runs:
        run.font.color.rgb = _MUTED_COLOR


def _add_title(doc: Document, title: str, subtitle: Optional[str] = None) -> None:
    """Add the document title with an optional subtitle"""
    heading = doc.add_heading(title, level=0)
    for run in heading.runs:
        run.font.color.rgb = _ACCENT_COLOR
    if subtitle:
        paragraph = doc.add_paragraph()
        run = paragraph.add_run(subtitle)
        run.font.size = Pt(14)
        run.bold = True


def _add_inline(paragraph: Paragraph, text: str) -> None:
    """Add text to the paragraph turning **bold** markup into bold runs"""
    for idx, part in enumerate(_BOLD.split(text)):
        if part:
            paragraph.add_run(part.replace("`", "")).bold = idx % 2 == 1


def _add_placeholder(doc: Document, labels: _Labels) -> None:
    """Add a muted 'not specified' line for an empty section"""
    run = doc.add_paragraph().add_run(labels.not_specified.capitalize())
    run.italic = True
    run.font.color.rgb = _MUTED_COLOR


def _add_numbered(doc: Document, items: List[str]) -> None:
    """Add a numbered list that always starts from 1

    Numbers are written explicitly: Word's built-in 'List Number' style
    continues numbering across separate lists in one document.
    """
    for idx, item in enumerate(items, start=1):
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Cm(0.75)
        paragraph.paragraph_format.first_line_indent = Cm(-0.75)
        paragraph.add_run(f"{idx}.\t")
        _add_inline(paragraph, item)


def _add_bullets(doc: Document, items: List[str]) -> None:
    """Add a bulleted list"""
    for item in items:
        _add_inline(doc.add_paragraph(style="List Bullet"), item)


def _shade_cell(cell, fill: str) -> None:
    """Fill a table cell background"""
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shading)


def _add_table(
    doc: Document,
    header: List[str],
    rows: List[List[str]],
    widths: Optional[List[float]] = None,
) -> Table:
    """Add a grid table with a shaded header row repeated on every page

    Args:
        doc (Document): Target document
        header (List[str]): Header captions
        rows (List[List[str]]): Table rows
        widths (List[float], optional): Column widths in cm

    Returns:
        Table: Created table
    """
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    header_row = table.rows[0]
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    header_row._tr.get_or_add_trPr().append(repeat)
    for cell, caption in zip(header_row.cells, header):
        cell.text = ""
        cell.paragraphs[0].add_run(caption).bold = True
        _shade_cell(cell, _HEADER_FILL)

    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = ""
            _add_inline(cell.paragraphs[0], value)

    if widths:
        table.autofit = False
        for row in table.rows:
            for cell, width in zip(row.cells, widths):
                cell.width = Cm(width)

    doc.add_paragraph()
    return table


def _add_transcript(
    doc: Document,
    transcript: str,
    speaker_names: Dict[str, str],
) -> None:
    """Add the transcript with bold speaker names

    Args:
        doc (Document): Target document
        transcript (str): Session text, '[SPEAKER_XX]: text' lines when diarized
        speaker_names (Dict[str, str]): Speaker label -> display name
    """
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        paragraph = doc.add_paragraph()
        match = _SPEAKER_LINE.match(line)
        if match:
            speaker = match.group("speaker")
            paragraph.add_run(f"{speaker_names.get(speaker, speaker)}: ").bold = True
            line = match.group("text")
        paragraph.add_run(line)


def _to_bytes(doc: Document) -> bytes:
    """Serialize the document to .docx bytes"""
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def build_minutes_docx(
    minutes: MeetingMinutes,
    language: str,
    created_at: datetime,
    session_id: str,
    transcript: Optional[str] = None,
) -> bytes:
    """Render structured meeting minutes as a Word document

    Args:
        minutes (MeetingMinutes): Extracted minutes
        language (str): ISO 639-1 language code of the captions
        created_at (datetime): Meeting (session) date
        session_id (str): Session identifier
        transcript (str, optional): Session text added as an appendix

    Returns:
        bytes: .docx file content
    """
    labels = _get_labels(language)
    doc = _new_document()
    _add_footer(doc, labels, labels.minutes_title)
    _add_title(doc, labels.minutes_title, minutes.title)

    info = doc.add_table(rows=0, cols=2)
    info.autofit = False
    for caption, value in (
        (labels.date, created_at.strftime("%d.%m.%Y %H:%M")),
        (labels.participants_count, str(len(minutes.participants))),
        (labels.session, session_id),
    ):
        cells = info.add_row().cells
        cells[0].text = ""
        cells[0].paragraphs[0].add_run(caption).bold = True
        cells[1].text = value
        cells[0].width, cells[1].width = Cm(4), Cm(13)
    doc.add_paragraph()

    doc.add_heading(labels.summary, level=1)
    _add_inline(doc.add_paragraph(), minutes.summary)

    speaker_names = {}
    doc.add_heading(labels.participants, level=1)
    if minutes.participants:
        rows = []
        for participant in minutes.participants:
            if participant.name:
                speaker_names[participant.speaker] = (
                    f"{participant.name} ({participant.speaker})"
                )
            rows.append([
                participant.speaker,
                participant.name or "—",
                participant.role or "—",
            ])
        _add_table(doc, [labels.speaker, labels.name, labels.role], rows, [4, 6, 7])
    else:
        _add_placeholder(doc, labels)

    doc.add_heading(labels.agenda, level=1)
    if minutes.agenda:
        _add_numbered(doc, [item.title for item in minutes.agenda])
    else:
        _add_placeholder(doc, labels)

    if minutes.agenda:
        doc.add_heading(labels.discussion, level=1)
        for idx, item in enumerate(minutes.agenda, start=1):
            doc.add_heading(f"{idx}. {item.title}", level=2)
            _add_inline(doc.add_paragraph(), item.discussion)
            _add_bullets(doc, item.key_points)

    doc.add_heading(labels.decisions, level=1)
    if minutes.decisions:
        _add_numbered(doc, minutes.decisions)
    else:
        _add_placeholder(doc, labels)

    doc.add_heading(labels.action_items, level=1)
    if minutes.action_items:
        rows = [
            [
                str(idx),
                action.task,
                action.owner or labels.not_specified,
                action.deadline or labels.not_specified,
            ]
            for idx, action in enumerate(minutes.action_items, start=1)
        ]
        _add_table(
            doc,
            [labels.number, labels.task, labels.owner, labels.deadline],
            rows,
            [1, 9, 4, 3],
        )
    else:
        _add_placeholder(doc, labels)

    doc.add_heading(labels.open_questions, level=1)
    if minutes.open_questions:
        _add_bullets(doc, minutes.open_questions)
    else:
        _add_placeholder(doc, labels)

    if transcript and transcript.strip():
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        doc.add_heading(labels.transcript, level=1)
        _add_transcript(doc, transcript, speaker_names)

    return _to_bytes(doc)


def _split_table_row(line: str) -> List[str]:
    """Split a Markdown table row into cell texts"""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def build_summary_docx(
    summary: str,
    options: SummaryOptions,
    created_at: datetime,
) -> bytes:
    """Render a Markdown summary as a Word document

    Supports headings, bullet and numbered lists, tables and **bold** text.

    Args:
        summary (str): Generated summary in Markdown
        options (SummaryOptions): Options the summary was generated with
        created_at (datetime): Session creation date

    Returns:
        bytes: .docx file content
    """
    labels = _get_labels(options.language)
    caption = labels.styles[options.style]
    doc = _new_document()
    _add_footer(doc, labels, caption)
    _add_title(doc, caption)

    date = doc.add_paragraph().add_run(
        f"{labels.date}: {created_at.strftime('%d.%m.%Y %H:%M')}"
    )
    date.font.color.rgb = _MUTED_COLOR

    lines = summary.strip().splitlines()
    numbered: List[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].rstrip()
        number_match = _NUMBERED_LINE.match(line)
        if numbered and not number_match:
            _add_numbered(doc, numbered)
            numbered = []

        if not line.strip():
            idx += 1
            continue

        if line.lstrip().startswith("|"):
            rows = []
            while idx < len(lines) and lines[idx].lstrip().startswith("|"):
                if not _TABLE_SEPARATOR.match(lines[idx].strip()):
                    rows.append(_split_table_row(lines[idx]))
                idx += 1
            width = max(len(row) for row in rows)
            rows = [row + [""] * (width - len(row)) for row in rows]
            _add_table(doc, rows[0], rows[1:])
            continue

        heading_match = _HEADING_LINE.match(line)
        bullet_match = _BULLET_LINE.match(line)
        if heading_match:
            # Summary prompts use '###' for top-level sections
            level = 1 if len(heading_match.group("level")) <= 3 else 2
            doc.add_heading(_BOLD.sub(r"\1", heading_match.group("text")), level=level)
        elif number_match:
            numbered.append(number_match.group("text"))
        elif bullet_match:
            _add_inline(doc.add_paragraph(style="List Bullet"), bullet_match.group("text"))
        else:
            _add_inline(doc.add_paragraph(), line.strip())
        idx += 1

    if numbered:
        _add_numbered(doc, numbered)

    return _to_bytes(doc)
