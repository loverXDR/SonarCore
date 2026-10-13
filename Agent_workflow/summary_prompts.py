"""Localized prompt catalog for parametrized document summarization"""

from typing import Dict, List, NamedTuple

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from Core.Schemas import SummaryOptions, SummarySize, SummaryStyle


class _PromptSet(NamedTuple):
    """Prompt texts for a single summary language"""

    material_query: str
    base_rules: str
    styles: Dict[SummaryStyle, str]
    sizes: Dict[SummarySize, str]
    language_rule: str
    material_header: str
    comment_header: str


_RU = _PromptSet(
    material_query=(
        "Изложи подробно и полностью всё содержание документа: все обсуждаемые темы, "
        "факты, цифры, даты, имена и спикеров, аргументы, принятые решения, "
        "договорённости и поставленные задачи. Ничего не опускай и не сокращай. "
        "Пиши на языке оригинала документа."
    ),
    base_rules=(
        "Общие правила:\n"
        "• Используй ТОЛЬКО факты из предоставленного материала. Ничего не выдумывай и не додумывай.\n"
        "• Сохраняй точные имена, цифры, даты и термины.\n"
        "• Не пиши вступлений и заключений вроде «Вот саммари» или «Надеюсь, это поможет» — сразу результат.\n"
        "• Если для обязательного раздела нет данных, пиши «не указано».\n"
        "• Используй Markdown-разметку там, где это предусмотрено форматом."
    ),
    styles={
        SummaryStyle.BRIEF: (
            "Ты — опытный редактор, который делает ёмкие выжимки.\n"
            "Задача: передать суть документа так, чтобы читатель за минуту понял, о чём он.\n"
            "Формат: связный текст без списков и заголовков. "
            "Начни с главной темы, затем ключевые моменты, закончи основным выводом или итогом."
        ),
        SummaryStyle.DETAILED: (
            "Ты — аналитик, готовящий подробный аналитический обзор документа.\n"
            "Задача: раскрыть все значимые темы с фактами, аргументами и цифрами.\n"
            "Формат:\n"
            "• Разбей текст на разделы с заголовками `### <Тема>` в порядке их появления в документе.\n"
            "• В каждом разделе — суть темы, ключевые факты, позиции участников и цифры.\n"
            "• В конце — раздел `### Выводы` с главными итогами документа."
        ),
        SummaryStyle.BULLET_POINTS: (
            "Ты — специалист по структурированию информации.\n"
            "Задача: свести содержание документа к чётким тезисам.\n"
            "Формат:\n"
            "• Сгруппируй тезисы по темам, каждая тема — подзаголовок `**<Тема>**`.\n"
            "• Под темой — маркированный список `- `; один пункт = одна мысль, не длиннее 1–2 строк.\n"
            "• Внутри темы располагай пункты от самого важного к менее важному."
        ),
        SummaryStyle.MEETING_MINUTES: (
            "Ты — секретарь, составляющий официальный протокол встречи.\n"
            "Задача: зафиксировать ход встречи, решения и поручения.\n"
            "Формат — строго следующие разделы в указанном порядке:\n"
            "### Тема встречи\n"
            "### Участники — по меткам спикеров вида `[SPEAKER_XX]` или по упомянутым именам\n"
            "### Повестка — нумерованный список обсуждаемых вопросов\n"
            "### Ход обсуждения — по каждому вопросу повестки: кто что предлагал, ключевые аргументы\n"
            "### Принятые решения — нумерованный список\n"
            "### Задачи — таблица `| Задача | Ответственный | Срок |`\n"
            "### Открытые вопросы — что осталось нерешённым"
        ),
    },
    sizes={
        SummarySize.SHORT: (
            "Объём: кратко, до ~100 слов. Только самое главное. "
            "Для протокола — минимально заполненные разделы, «Ход обсуждения» в 1–2 предложения."
        ),
        SummarySize.MEDIUM: (
            "Объём: умеренно, ~150–300 слов. Основные темы и ключевые детали."
        ),
        SummarySize.LONG: (
            "Объём: развёрнуто, ~400–800 слов. Раскрой все темы документа с деталями и цифрами. "
            "Для протокола — подробный «Ход обсуждения» по каждому вопросу."
        ),
    },
    language_rule="Пиши итоговый текст строго на русском языке.",
    material_header="Материал документа:",
    comment_header=(
        "Дополнительные пожелания пользователя (учитывай их, если они не противоречат "
        "заданному формату и фактам из материала):"
    ),
)

_EN = _PromptSet(
    material_query=(
        "Present the full content of the document in detail: all topics discussed, "
        "facts, figures, dates, names and speakers, arguments, decisions made, "
        "agreements and assigned tasks. Do not omit or shorten anything. "
        "Write in the original language of the document."
    ),
    base_rules=(
        "General rules:\n"
        "• Use ONLY facts from the provided material. Do not invent or assume anything.\n"
        "• Preserve exact names, figures, dates and terms.\n"
        "• Do not add intros or outros like \"Here is the summary\" or \"Hope this helps\" — output the result directly.\n"
        "• If a required section has no data, write \"not specified\".\n"
        "• Use Markdown where the format calls for it."
    ),
    styles={
        SummaryStyle.BRIEF: (
            "You are an experienced editor who writes concise digests.\n"
            "Task: convey the essence of the document so a reader understands it in one minute.\n"
            "Format: coherent prose with no lists or headings. "
            "Start with the main topic, then the key points, and finish with the main conclusion or outcome."
        ),
        SummaryStyle.DETAILED: (
            "You are an analyst preparing a detailed analytical review of the document.\n"
            "Task: cover every significant topic with facts, arguments and figures.\n"
            "Format:\n"
            "• Split the text into sections with `### <Topic>` headings in the order they appear in the document.\n"
            "• In each section: the gist of the topic, key facts, participants' positions and figures.\n"
            "• Finish with a `### Conclusions` section summarizing the main outcomes."
        ),
        SummaryStyle.BULLET_POINTS: (
            "You are an information structuring specialist.\n"
            "Task: reduce the document content to clear key points.\n"
            "Format:\n"
            "• Group points by topic, each topic is a `**<Topic>**` subheading.\n"
            "• Under each topic, a `- ` bullet list; one bullet = one idea, no longer than 1–2 lines.\n"
            "• Within a topic, order bullets from most to least important."
        ),
        SummaryStyle.MEETING_MINUTES: (
            "You are a secretary drafting official meeting minutes.\n"
            "Task: record the course of the meeting, decisions and action items.\n"
            "Format — exactly the following sections in this order:\n"
            "### Meeting topic\n"
            "### Participants — from speaker labels like `[SPEAKER_XX]` or mentioned names\n"
            "### Agenda — numbered list of discussed items\n"
            "### Discussion — for each agenda item: who proposed what, key arguments\n"
            "### Decisions — numbered list\n"
            "### Action items — table `| Task | Owner | Deadline |`\n"
            "### Open questions — what remains unresolved"
        ),
    },
    sizes={
        SummarySize.SHORT: (
            "Length: short, up to ~100 words. Only the essentials. "
            "For minutes — minimally filled sections, \"Discussion\" in 1–2 sentences."
        ),
        SummarySize.MEDIUM: (
            "Length: moderate, ~150–300 words. Main topics and key details."
        ),
        SummarySize.LONG: (
            "Length: extensive, ~400–800 words. Cover every topic with details and figures. "
            "For minutes — a detailed \"Discussion\" for each agenda item."
        ),
    },
    language_rule="Write the final text strictly in the language with ISO 639-1 code '{language}'.",
    material_header="Document material:",
    comment_header=(
        "Additional user instructions (follow them unless they conflict with "
        "the required format or the facts in the material):"
    ),
)

_PROMPTS: Dict[str, _PromptSet] = {
    "ru": _RU,
    "en": _EN,
}

_DEFAULT_LANGUAGE = "en"


def _get_prompt_set(language: str) -> _PromptSet:
    """Return the prompt set for a language, falling back to the default one

    Args:
        language (str): ISO 639-1 language code

    Returns:
        _PromptSet: Localized prompt texts
    """
    return _PROMPTS.get(language.lower(), _PROMPTS[_DEFAULT_LANGUAGE])


def get_material_query(language: str) -> str:
    """Build the query used to extract full document material from the SummaryIndex

    Args:
        language (str): ISO 639-1 language code of the summary

    Returns:
        str: Material extraction query
    """
    return _get_prompt_set(language).material_query


def build_summary_messages(options: SummaryOptions, material: str) -> List[AnyMessage]:
    """Build LLM messages that compose the final summary from extracted material

    Args:
        options (SummaryOptions): Summary language, style, size and comment
        material (str): Full document material extracted from the SummaryIndex

    Returns:
        List[AnyMessage]: System and human messages for the composing LLM call
    """
    prompts = _get_prompt_set(options.language)

    system = "\n\n".join([
        prompts.styles[options.style],
        prompts.base_rules,
        prompts.sizes[options.size],
        prompts.language_rule.format(language=options.language),
    ])

    human = f"{prompts.material_header}\n<material>\n{material}\n</material>"
    if options.comment and options.comment.strip():
        human += f"\n\n{prompts.comment_header}\n<comment>\n{options.comment.strip()}\n</comment>"

    return [SystemMessage(content=system), HumanMessage(content=human)]
