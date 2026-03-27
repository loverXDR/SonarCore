"""LangGraph agent for RAG-based document chat"""

import operator
from typing import List, Annotated

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


class SonarAgentState(TypedDict):
    """State for the agent graph"""

    messages: Annotated[List[AnyMessage], operator.add]


class SonarAgent:
    """Chat agent built on LangGraph StateGraph

    Uses ReAct pattern: llm_call -> tool_node -> llm_call
    until the LLM produces a final answer without tool calls.
    """

    def __init__(
        self,
        config: AgentConfig,
        tools: list,
    ) -> None:
        self.config = config
        self.tools = tools
        self._history: List[AnyMessage] = []
        self._llm = self._build_llm()
        self._graph = self._build_graph()

    def _build_llm(self) -> ChatOpenAI:
        """Build LLM with tools bound"""
        llm = ChatOpenAI(
            model=self.config.llm.model,
            openai_api_base=self.config.llm.api_base,
            openai_api_key=self.config.llm.api_key,
        )
        if self.tools:
            llm = llm.bind_tools(self.tools)
        return llm

    def _build_graph(self) -> StateGraph:
        """Build and compile the agent graph

        Returns:
            Compiled LangGraph StateGraph
        """
        tools_by_name = {t.name: t for t in self.tools}

        def llm_call(state: dict):
            system = SystemMessage(
                content=self.config.system_prompt,
            )
            response = self._llm.invoke(
                [system] + state["messages"],
            )
            return {"messages": [response]}

        def tool_node(state: dict):
            results = []
            last = state["messages"][-1]
            for tc in last.tool_calls:
                tool_fn = tools_by_name[tc["name"]]
                output = tool_fn.invoke(tc["args"])
                results.append(
                    ToolMessage(
                        content=str(output),
                        tool_call_id=tc["id"],
                    )
                )
            return {"messages": results}

        def should_continue(state: SonarAgentState):
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tool_node"
            return END

        builder = StateGraph(SonarAgentState)
        builder.add_node("llm_call", llm_call)
        builder.add_node("tool_node", tool_node)
        builder.add_edge(START, "llm_call")
        builder.add_conditional_edges(
            "llm_call",
            should_continue,
            ["tool_node", END],
        )
        builder.add_edge("tool_node", "llm_call")

        return builder.compile()

    def chat(self, message: str) -> AgentResponse:
        """Send a message and get agent response

        Args:
            message (str): User message text

        Returns:
            AgentResponse: Agent answer with sources
        """
        self._history.append(HumanMessage(content=message))

        result = self._graph.invoke(
            {"messages": list(self._history)},
        )

        new_messages = result["messages"]
        self._history = new_messages

        assistant_msg = new_messages[-1]
        answer = assistant_msg.content

        sources = []
        for msg in new_messages:
            if isinstance(msg, ToolMessage):
                sources.append({"tool_output": msg.content})

        history = []
        for msg in new_messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, ToolMessage):
                continue
            else:
                role = "assistant"
            history.append(
                AgentMessage(role=role, content=msg.content)
            )

        return AgentResponse(
            answer=answer,
            sources=sources,
            history=history,
        )

    def reset(self) -> None:
        """Reset conversation history"""
        self._history = []
