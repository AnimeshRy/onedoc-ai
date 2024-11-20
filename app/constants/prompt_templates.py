from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain.prompts import MessagesPlaceholder

EMBEDDING_PROMPT_TEMPLATES = {
    "default": ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are a helpful AI assistant. Use the following context to answer the question.
            If you cannot find the answer in the context, say so. Always maintain a professional and friendly tone."""
            ),
            HumanMessagePromptTemplate.from_template(
                "content='Context:\n{context}\n\nQuestion: {question}"
            ),
        ]
    ),
    "technical": ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are a technical expert. Provide detailed, technical responses using the context provided.
            Include code examples or technical specifications where relevant. If information is missing from the context,
            clearly state what additional details would be needed."""
            ),
            HumanMessagePromptTemplate.from_template(
                "content='Technical Context:\n{context}\n\nTechnical Query: {question}"
            ),
        ]
    ),
    "summary": ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""Provide a concise summary of the relevant information from the context.
            Break down complex topics into clear, easy-to-understand points. Focus on the key takeaways."""
            ),
            HumanMessagePromptTemplate.from_template(
                "content='Content to Summarize:\n{context}\n\nSummary Request: {question}"
            ),
        ]
    ),
    "analytical": ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""Conduct a detailed analysis of the information provided in the context.
            Consider multiple perspectives, identify patterns, and provide evidence-based conclusions.
            Structure your response with clear sections for background, analysis, and recommendations."""
            ),
            HumanMessagePromptTemplate.from_template(
                "content='Analysis Context:\n{context}\n\nAnalytical Question: {question}"
            ),
        ]
    ),
}

CONTEXT_SYSTEM_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
            Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood
            without the chat history. Do NOT answer the question, just reformulate if it needed and otherwise return it as is.
            """
        ),
        MessagesPlaceholder("chat_history"),
        HumanMessagePromptTemplate.from_template("input: {input}"),
    ]
)
