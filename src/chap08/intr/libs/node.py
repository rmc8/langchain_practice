from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

from .model import get_model
from .state import State, Judgement
from .role import ROLES


def selection_node(state: State) -> Dict[str, Any]:
    query = state.query
    role_options = "\n".join(
        [f"{k}. {v['name']}: {v['description']}" for k, v in ROLES.items()]
    )
    prompt = ChatPromptTemplate.from_template(
        """
        質問を分析し、最も適切な回答担当ロールを選択してください。
        
        選択肢：
        {role_options}
        
        解答は選択肢の番号(1,2,または3)のみを返してください。
        
        質問：{query}
        """.strip()
    )
    # NOTE:選択肢の番号のみを返すことを期待しないため、max_tokensの値を1に変更
    llm = get_model()
    chain = prompt | llm | StrOutputParser()
    role_number = chain.invoke(
        {
            "role_options": role_options,
            "query": query,
        }
    )
    try:
        selected_role = ROLES[role_number.strip()]["name"]
    except KeyError:
        DEFAULT_ROLE = "1"
        selected_role = ROLES[DEFAULT_ROLE]["name"]
    return {
        "current_role": selected_role,
    }


def answering_node(state: State) -> Dict[str, Any]:
    query = state.query
    role = state.current_role
    role_details = "\n".join([f"- {v['name']}: {v['details']}" for v in ROLES.values()])
    prompt = ChatPromptTemplate.from_template(
        """
        あなたは{role}として回答してください。以下の質問に対して、あなたの役割に基づいた適切な回答を提供してください。
        
        役割の詳細：
        {role_details}
        
        質問： {query}
        
        回答：
        """.strip()
    )
    llm = get_model()
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke(
        {
            "role": role,
            "role_details": role_details,
            "query": query,
        }
    )
    return {
        "messages": [answer],
    }


def check_node(state: State) -> Dict[str, Any]:
    query = state.query
    answer = state.messages[-1]
    prompt = ChatPromptTemplate.from_template(
        """
        以下の回答の品質をチェックし、問題がある場合は'False'、問題がない場合は'True'を回答してください。また、その判断理由も説明してください。
        
        ユーザーからの質問： {query}
        
        回答： {answer}
        """.strip()
    )
    llm = get_model()
    chain = prompt | llm | PydanticOutputParser(pydantic_object=Judgement)
    result: Judgement = chain.invoke({"query": query, "answer": answer})
    return {
        "current_judge": result.judge,
        "judgement_reason": result.reason,
    }


def check_node(state: State) -> Dict[str, Any]:
    query = state.query
    answer = state.messages[-1]
    prompt = ChatPromptTemplate.from_template(
        """
        以下の回答の品質をチェックしてください。問題がある場合は、理由とともに'False'を、問題がない場合は、理由とともに'True'をJSON形式で回答してください。

        出力形式:
        ```json
        {{
            "reason": "判定理由",
            "judge": true or false
        }}
        ```

        ユーザーからの質問： {query}

        回答： {answer}
        """.strip()
    )
    llm = get_model()
    chain = prompt | llm | PydanticOutputParser(pydantic_object=Judgement)
    result: Judgement = chain.invoke({"query": query, "answer": answer})
    return {
        "current_judge": result.judge,
        "judgement_reason": result.reason,
    }
