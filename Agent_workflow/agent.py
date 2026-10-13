"""LangGraph multi-agent orchestration for RAG-based document chat"""

import operator
from typing import List, Annotated, Literal, Optional

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from Core.Schemas import (
    AgentConfig,
    AgentMessage,
    AgentResponse,
    MeetingMinutes,
    SummaryOptions,
)
from .summary_prompts import (
    build_minutes_messages,
    build_summary_messages,
    get_material_query,
)

# Rough chars-per-token ratio used to decide whether a transcript fits
# into half of the LLM context window (the rest is left for the prompt and output)
_CHARS_PER_TOKEN = 3


def _content_to_text(raw) -> str:
    """Normalize message content to plain text

    Some LLM backends return content as a list of blocks instead of a plain string.

    Args:
        raw: Message content (str or list of blocks)

    Returns:
        str: Plain text content
    """
    if isinstance(raw, str):
        return raw
    return " ".join(
        b.get("text", "") if isinstance(b, dict) else str(b) for b in raw
    )


class AgentState(TypedDict):
    """State for the multi-agent graph"""

    messages: Annotated[List[AnyMessage], operator.add]
    next: str


class Route(BaseModel):
    """Routing decision for the supervisor"""

    next: Literal["qa", "summarize", "chat", "FINISH"] = Field(
        ...,
        description=(
            "The next agent to route to. "
            "Use 'qa' for specific questions about the document content. "
            "Use 'summarize' for summaries, key points, or meeting minutes. "
            "Use 'chat' for greetings, small talk, questions about the conversation history, "
            "or anything that does not require searching the document. "
            "Use 'FINISH' ONLY when the user explicitly ends the conversation (e.g. 'goodbye', 'bye')."
        ),
    )


class SonarAgent:
    """Multi-Agent orchestrator built on LangGraph StateGraph

    Uses a Supervisor to route between:
    - QA Agent (for answering questions)
    - Summarize Agent (for meeting minutes and summaries)
    """

    def __init__(
        self,
        config: AgentConfig,
        search_tool,
        summarize_tool,
    ) -> None:
        self.config = config
        self.search_tool = search_tool
        self.summarize_tool = summarize_tool
        self._history: List[AnyMessage] = []
        self._llm = self._build_llm()
        self._graph = self._build_graph()

    def _build_llm(self) -> ChatOpenAI:
        """Build base LLM"""
        return ChatOpenAI(
            model=self.config.llm.model,
            openai_api_base=self.config.llm.api_base,
            openai_api_key=self.config.llm.api_key,
        )

    def _build_graph(self) -> StateGraph:
        """Build and compile the multi-agent graph"""

        # 1. Supervisor Node
        def supervisor_node(state: AgentState):
            supervisor_prompt = (
                "Ты — маршрутизатор. Все запросы связаны с загруженным документом/аудиозаписью.\n"
                "Агенты-исполнители:\n"
                "1) 'qa' — вопросы по содержанию документа, уточнения, поиск информации в тексте. "
                "Используй по умолчанию для любого вопроса о документе.\n"
                "2) 'summarize' — создание саммари, конспектов, списков ключевых тем, протоколов.\n"
                "3) 'chat' — ТОЛЬКО: приветствия ('привет', 'добрый день') "
                "и вопросы о ходе нашего диалога ('о чём мы говорили?', 'что ты уже ответил?').\n"
                "Текст документа уже загружен. "
                "'FINISH' — только если пользователь явно прощается ('пока', 'до свидания', 'спасибо, всё')."
            )
            router = self._llm.with_structured_output(Route)
            messages = [SystemMessage(content=supervisor_prompt)] + state["messages"]
            decision = router.invoke(messages)
            return {"next": decision.next}

        # 2. QA Node
        qa_llm = self._llm.bind_tools([self.search_tool])
        def qa_node(state: AgentState):
            system = SystemMessage(
                content="Ты — QA-агент. Текст документа уже загружен в базу данных.\n"
                "Твоя задача — отвечать на вопросы пользователя по документу.\n"
                "Тебе НЕ НУЖНО просить пользователя предоставить текст. СРАЗУ используй инструмент поиска `search_documents`, чтобы найти ответ в базе данных на вопрос пользователя."
            )
            messages = [system] + state["messages"]
            response = qa_llm.invoke(messages)

            # If tool called, execute and return as QA answer
            if response.tool_calls:
                results = [response]
                for tc in response.tool_calls:
                    output = self.search_tool.invoke(tc["args"])
                    results.append(
                        ToolMessage(content=str(output), tool_call_id=tc["id"])
                    )
                # LLM processes tool output to final answer
                final = self._llm.invoke([system] + state["messages"] + results)
                return {"messages": [final]}

            return {"messages": [response]}

        # 3. Summarize Node
        sum_llm = self._llm.bind_tools([self.summarize_tool])
        def summarize_node(state: AgentState):
            system = SystemMessage(
                content="Ты — Summarize-агент. Текст документа уже загружен в базу данных.\n"
                "Твоя задача — составлять конспекты и выжимки из документа.\n"
                "Тебе НЕ НУЖНО просить пользователя предоставить текст. СРАЗУ используй "
                "инструмент `summarize_documents`, чтобы получить содержание из базы данных. "
                "Используй запрос вида 'summary' или суть того, что просит пользователь."
            )
            messages = [system] + state["messages"]
            response = sum_llm.invoke(messages)

            if response.tool_calls:
                results = [response]
                for tc in response.tool_calls:
                    output = self.summarize_tool.invoke(tc["args"])
                    results.append(
                        ToolMessage(content=str(output), tool_call_id=tc["id"])
                    )
                final = self._llm.invoke([system] + state["messages"] + results)
                return {"messages": [final]}

            return {"messages": [response]}

        # 4. Chat Node — greetings and meta-questions about the conversation only.
        # Must stay grounded to the loaded document context.
        def chat_node(state: AgentState):
            system = SystemMessage(
                content="Ты — SonarAI, ассистент для работы с загруженным документом или аудиозаписью.\n"
                "Твоя единственная специализация — помогать пользователю разобраться в загруженном тексте.\n"
                "Правила:\n"
                "• Если пользователь здоровается — поздоровайся и предложи задать вопрос по документу.\n"
                "• Если спрашивает о нашем диалоге — кратко перескажи, что уже обсуждалось.\n"
                "• Если пишет что-то не связанное с документом — вежливо верни разговор к тексту.\n"
                "• НЕ отвечай на вопросы, не касающиеся загруженного документа.\n"
                "• НЕ выдумывай информацию вне контекста документа.\n"
                "Отвечай на языке пользователя."
            )
            response = self._llm.invoke([system] + state["messages"])
            return {"messages": [response]}

        # Graph Construction
        builder = StateGraph(AgentState)
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("qa", qa_node)
        builder.add_node("summarize", summarize_node)
        builder.add_node("chat", chat_node)

        builder.add_edge(START, "supervisor")

        builder.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {
                "qa": "qa",
                "summarize": "summarize",
                "chat": "chat",
                "FINISH": END,
            },
        )

        builder.add_edge("qa", END)
        builder.add_edge("summarize", END)
        builder.add_edge("chat", END)

        return builder.compile()

    def chat(self, message: str) -> AgentResponse:
        """Send a message and get agent response

        Args:
            message (str): User message text

        Returns:
            AgentResponse: Agent answer with sources
        """
        self._history.append(HumanMessage(content=message))

        # Invoke the graph
        result = self._graph.invoke(
            {"messages": list(self._history), "next": ""}
        )

        new_messages = result["messages"]
        self._history = new_messages

        # Extract the answer from the last message produced by a worker node.
        # When supervisor routes to FINISH with no worker invoked, the last message
        # is the HumanMessage the user just sent — fall back to a closing line.
        last_msg = new_messages[-1] if new_messages else None
        if last_msg is None or isinstance(last_msg, HumanMessage):
            answer = "До свидания! Обращайтесь, если понадоблюсь."
        else:
            answer = _content_to_text(getattr(last_msg, "content", ""))

        # Build history filtering out internal tool messages
        history = []
        for msg in new_messages:
            if isinstance(msg, ToolMessage):
                continue
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                continue
            content = _content_to_text(getattr(msg, "content", ""))
            if content:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                history.append(AgentMessage(role=role, content=content))

        return AgentResponse(
            answer=answer,
            sources=[],
            history=history,
        )

    def summarize(self, options: SummaryOptions) -> AgentResponse:
        """Generate a document summary with explicit parameters

        Two-stage generation bypassing the supervisor:
        1. Extract full document material via the summarize tool (SummaryIndex).
        2. Compose the final summary with a localized style-specific prompt.
        Conversation history is left untouched.

        Args:
            options (SummaryOptions): Summary language, style, size and comment

        Returns:
            AgentResponse: Generated summary
        """
        material = self.summarize_tool.invoke(
            {"query": get_material_query(options.language)}
        )
        final = self._llm.invoke(build_summary_messages(options, str(material)))
        answer = _content_to_text(getattr(final, "content", ""))
        return AgentResponse(answer=answer, sources=[], history=[])

    def meeting_minutes(
        self,
        options: SummaryOptions,
        transcript: Optional[str] = None,
    ) -> MeetingMinutes:
        """Extract structured meeting minutes

        The full transcript is used when it fits into the context window, so
        speaker labels and the order of the discussion are preserved. Otherwise
        the material is extracted via the summarize tool (SummaryIndex).

        Args:
            options (SummaryOptions): Minutes language, size and comment
            transcript (str, optional): Source transcript of the session

        Returns:
            MeetingMinutes: Structured minutes
        """
        max_chars = self.config.llm.context_window * _CHARS_PER_TOKEN // 2
        if transcript and len(transcript) <= max_chars:
            material = transcript
        else:
            material = str(self.summarize_tool.invoke(
                {"query": get_material_query(options.language)}
            ))
        extractor = self._llm.with_structured_output(MeetingMinutes)
        return extractor.invoke(build_minutes_messages(options, material))

    def reset(self) -> None:
        """Reset conversation history"""
        self._history = []
