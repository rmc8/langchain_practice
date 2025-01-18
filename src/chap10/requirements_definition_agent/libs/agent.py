from typing import Any, Dict, Optional

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from .state import InterviewState
from .data import Personas, InterviewResult, EvaluationResult
from .component import (
    PersonaGenerator,
    InterviewConductor,
    InformationEvaluator,
    RequirementsDocumentGenerator,
)


class DocumentationAgent:
    GENERATE_PERSONAS = "generate_personas"
    CONDUCT_INTERVIEWS = "conduct_interviews"
    EVALUATE_INFORMATION = "evaluate_information"
    GENERATE_REQUIREMENTS = "generate_requirements"

    def __init__(self, llm: ChatOllama, k: Optional[int] = None):
        self.persona_generator = PersonaGenerator(llm, k)
        self.interview_conductor = InterviewConductor(llm)
        self.information_evaluator = InformationEvaluator(llm)
        self.requirements_generator = RequirementsDocumentGenerator(llm)

    def _create_graph(self) -> StateGraph:
        # Create a new graph with an initial node representing the start of the workflow
        workflow = StateGraph(InterviewState)
        # Add nodes to the graph
        workflow.add_node(self.GENERATE_PERSONAS, self._generate_personas)
        workflow.add_node(self.CONDUCT_INTERVIEWS, self._conduct_interviews)
        workflow.add_node(self.EVALUATE_INFORMATION, self._evaluate_information)
        workflow.add_node(self.GENERATE_REQUIREMENTS, self._generate_requirements)
        # Set the entry point of the workflow
        workflow.set_entry_point(self.GENERATE_PERSONAS)
        # Add edges between nodes to define the workflow flow
        workflow.add_edge(self.GENERATE_PERSONAS, self.CONDUCT_INTERVIEWS)
        workflow.add_edge(self.CONDUCT_INTERVIEWS, self.EVALUATE_INFORMATION)
        # Add conditional edges to handle different scenarios based on the current state of the workflow
        workflow.add_conditional_edges(
            self.EVALUATE_INFORMATION,
            lambda state: not state.is_information_sufficient and state.iteration < 5,
            {True: self.GENERATE_PERSONAS, False: self.GENERATE_REQUIREMENTS},
        )
        workflow.add_edge(self.GENERATE_REQUIREMENTS, END)
        # Compile the workflow to generate a runnable graph
        return workflow.compile()

    def _generate_personas(self, state: InterviewState) -> Dict[str, Any]:
        new_personas: Personas = self.persona_generator.run(state.user_request)
        return {
            "personas": new_personas.personas,
            "iteration": state.iteration + 1,
        }

    def _conduct_interviews(self, state: InterviewState) -> Dict[str, Any]:
        new_interviews: InterviewResult = self.interview_conductor.run(
            state.user_request,
            state.personas[-5:],
        )
        return {"interviews": new_interviews.interviews}

    def _evaluate_information(self, state: InterviewState) -> Dict[str, Any]:
        evaluation_result: EvaluationResult = self.information_evaluator.run(
            state.user_request,
            state.interviews,
        )
        return {
            "is_information_sufficient": evaluation_result.is_sufficient,
            "evaluation_reason": evaluation_result.reason,
        }

    def _generate_requirements(self, state: InterviewState) -> Dict[str, Any]:
        requirements_doc: str = self.requirements_generator.run(
            state.user_request,
            state.interviews,
        )
        return {"requirements_doc": requirements_doc}

    def run(self, user_request: str) -> str:
        initial_state = InterviewState(user_request=user_request)
        graph = self._create_graph()
        final_state = graph.invoke(initial_state)
        return final_state["requirements_doc"]
