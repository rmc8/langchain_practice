from typing import List

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .data import EvaluationResult, Persona, Personas, Interview, InterviewResult


class PersonaGenerator:
    def __init__(self, llm: ChatOllama, k: int = 5):
        self.llm = llm.with_structured_output(Personas)
        self.k = k

    def run(self, user_request: str) -> Personas:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはユーザーインタビュー向けに多様なペルソナを作成する専門家です。",
                ),
                (
                    "user",
                    "\n".join(
                        [
                            f"以下のユーザーリクエストに関するインタビュー用に、{self.k}人の多様なペルソナを作成してください。\n\n",
                            "ユーザーリクエスト： {user_request}\n\n",
                            "各ペルソナには名前と簡単な背景を含めてください。年齢、性別、職業、技術的専門知識において多様性を確保してください。",
                        ]
                    ),
                ),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke({"user_request": user_request})


class InterviewConductor:
    def __init__(self, llm: ChatOllama):
        self.llm = llm

    def run(self, user_request: str, personas: List[Persona]) -> InterviewResult:
        questions = self._generate_questions(
            user_request=user_request,
            personas=personas,
        )
        answers = self._generate_answers(personas=personas, questions=questions)
        interviews = self._create_interviews(
            personas=personas,
            questions=questions,
            answers=answers,
        )
        return InterviewResult(interviews=interviews)

    def _generate_questions(
        self, user_request: str, personas: List[Persona]
    ) -> List[str]:
        question_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはユーザーの要求に基づいて適切な質問を生成する専門家です。",
                ),
                (
                    "user",
                    "\n".join(
                        [
                            "以下のペルソナに関連するユーザーのリクエストについて、1つの質問を生成してください。\n\n",
                            "ユーザーリクエスト： {user_request}\n",
                            "ペルソナ: {persona_name} - {persona_background}\n\n",
                            "質問は具体的で、このペルソナの支店から重要な情報を引き出すように設計してください。",
                        ]
                    ),
                ),
            ]
        )
        question_chain = question_prompt | self.llm | StrOutputParser()
        question_queries = [
            {
                "user_request": user_request,
                "persona_name": persona.name,
                "persona_background": persona.background,
            }
            for persona in personas
        ]
        return question_chain.batch(question_queries)

    def _generate_answers(
        self, personas: List[Persona], questions: List[str]
    ) -> List[str]:
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは以下のペルソナとして回答しています： {persona_name} - {persona_background}",
                ),
                ("human", "質問: {question}"),
            ]
        )
        answer_chain = answer_prompt | self.llm | StrOutputParser()
        answer_queries = [
            {
                "persona_name": persona.name,
                "persona_background": persona.background,
                "question": question,
            }
            for persona, question in zip(personas, questions)
        ]
        return answer_chain.batch(answer_queries)

    def _create_interviews(
        self, personas: List[Persona], questions: List[str], answers: List[str]
    ) -> List[Interview]:
        return [
            Interview(persona=persona, question=question, answer=answer)
            for persona, question, answer in zip(personas, questions, answers)
        ]


class InformationEvaluator:
    def __init__(self, llm: ChatOllama):
        self.llm = llm.with_structured_output(EvaluationResult)

    def run(self, user_request: str, interviews: List[Interview]) -> EvaluationResult:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは包括的な要件文書を作成するための情報の十分性を評価する専門家です。",
                ),
                (
                    "human",
                    "\n".join(
                        [
                            "以下のユーザーリクエストとインタビュー結果に基づいて、包括的な要件文書を作成するのに十分な情報が集まったかどうかを判断してください。\n\n",
                            "ユーザーリクエスト: {user_request}\n\n",
                            "インタビュー結果:\n{interview_results}\n\n",
                        ],
                    ),
                ),
            ],
        )
        chain = prompt | self.llm
        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    "\n".join(
                        [
                            f"ペルソナ: {i.persona.name} - {i.persona.background}",
                            f"質問: {i.question}",
                            f"回答: {i.answer}",
                        ]
                    )
                    for i in interviews
                ),
            }
        )


class RequirementsDocumentGenerator:
    def __init__(self, llm: ChatOllama):
        self.llm = llm

    def run(self, user_request: str, interviews: List[Interview]) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは収集した情報に基づいて要件文書を作成する専門家です。",
                ),
                (
                    "human",
                    "\n".join(
                        [
                            "以下のユーザーリクエストと複数のペルソナからインタビュー結果に基づいて、要件文書を作成してください。\n\n",
                            "ユーザーリクエスト: {user_request}\n\n",
                            "インタビュー結果：\n{interview_results}\n"
                            "要件定義文書には以下のセクションを含めてください：",
                            "1. プロジェクト概要",
                            "2. 主要機能",
                            "3. 非機能要件",
                            "4. 制約条件",
                            "5. ターゲットユーザー",
                            "6. 優先順位",
                            "7. リスク評価とリスク軽減策",
                            "出力は必ず日本語でお願いします。\n\n要件文書:",
                        ]
                    ),
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"ペルソナ: {i.persona.name} - {i.persona.background}\n"
                    f"質問: {i.question}\n回答: {i.answer}\n"
                    for i in interviews
                ),
            }
        )
