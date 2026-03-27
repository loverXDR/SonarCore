"""LangGraph multi-agent orchestration for RAG-based document chat"""

import operator
from typing import List, Annotated, Literal

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

from Core.Schemas import AgentConfig, AgentMessage, AgentResponse


class AgentState(TypedDict):
    """State for the multi-agent graph"""

    messages: Annotated[List[AnyMessage], operator.add]
    next: str


class Route(BaseModel):
    """Routing decision for the supervisor"""

    next: Literal["qa", "summarize", "FINISH"] = Field(
        ...,
        description="The next agent to route to, or FINISH if the user request is complete or cannot be handled.",
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
                "Ты — маршрутизатор (Supervisor). У тебя есть два агента-исполнителя:\n"
                "1) 'qa' — для ответа на конкретные вопросы по тексту/записи.\n"
                "2) 'summarize' — для создания саммари, конспектов, выделения главных тем.\n"
                "Проанализируй запрос пользователя (последнее сообщение) и реши, кому передать задачу.\n"
                "Если пользователь просто здоровается или диалог завершен, верни 'FINISH'."
            )
            # Use structured output for guaranteed routing
            router = self._llm.with_structured_output(Route)
            messages = [SystemMessage(content=supervisor_prompt)] + state["messages"]
            decision = router.invoke(messages)
            return {"next": decision.next}

        # 2. QA Node
        qa_llm = self._llm.bind_tools([self.search_tool])
        def qa_node(state: AgentState):
            system = SystemMessage(
                content="Ты — QA-агент. Твоя задача — "
                "отвечать на вопросы пользователя по документу, обязательно используя инструмент поиска."
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
                content="Ты — Summarize-агент. Твоя задача — "
                "составлять конспекты и выжимки из документа, обязательно используя инструмент суммаризации."
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

        # Graph Construction
        builder = StateGraph(AgentState)
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("qa", qa_node)
        builder.add_node("summarize", summarize_node)

        builder.add_edge(START, "supervisor")

        # Routing from supervisor
        builder.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {
                "qa": "qa",
                "summarize": "summarize",
                "FINISH": END,
            },
        )

        # After worker finishes, go back to END (simple star topology for 1-turn interactions)
        builder.add_edge("qa", END)
        builder.add_edge("summarize", END)

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

        # The last message is the final answer from QA or Summarize node
        assistant_msg = new_messages[-1]
        
        # If supervisor routed to FINISH directly without producing a new AIMessage
        if isinstance(assistant_msg, HumanMessage):
             answer = "Что я могу еще для вас сделать?"
        else:
             answer = assistant_msg.content

        # Since we squashed tool execution in the node, 
        # let's just use the history to build AgentMessages
        history = []
        for msg in new_messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, ToolMessage):
                continue
            elif hasattr(msg, "tool_calls") and msg.tool_calls:
                continue
            else:
                role = "assistant"
            history.append(
                AgentMessage(role=role, content=msg.content)
            )

        return AgentResponse(
            answer=answer,
            sources=[],
            history=history,
        )

    def reset(self) -> None:
        """Reset conversation history"""
        self._history = []
