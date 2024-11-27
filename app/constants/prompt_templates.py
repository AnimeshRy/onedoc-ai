from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
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

CHAT_EMBEDDING_PROMPT_TEMPLATES = {
    "default": ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("""You are a helpful AI assistant. Use the following context to answer the question.
            If you cannot find the answer in the context, say so. Always maintain a professional and friendly tone.

            Do not answer questions other than the provided context:
            {context}
            """),
            MessagesPlaceholder("chat_history"),
            HumanMessagePromptTemplate.from_template("Question: {input}"),
        ]
    ),
    "technical": ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("""You are a technical expert. Provide detailed, technical responses using the context provided.
            Include code examples or technical specifications where relevant. If information is missing from the context,
            clearly state what additional details would be needed.

            Do not answer questions other than the provided context:
            {context}
            """),
            MessagesPlaceholder("chat_history"),
            HumanMessagePromptTemplate.from_template("Technical Query: {input}"),
        ]
    ),
    "summary": ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("""Provide a concise summary of the relevant information from the context.
            Break down complex topics into clear, easy-to-understand points. Focus on the key takeaways.

            Do not answer questions other than the provided context:
            {context}
            """),
            MessagesPlaceholder("chat_history"),
            HumanMessagePromptTemplate.from_template("Summary Request: {input}"),
        ]
    ),
    "analytical": ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("""Conduct a detailed analysis of the information provided in the context.
            Consider multiple perspectives, identify patterns, and provide evidence-based conclusions.
            Structure your response with clear sections for background, analysis, and recommendations.

            Do not answer questions other than the provided context:
            {context}
            """),
            MessagesPlaceholder("chat_history"),
            HumanMessagePromptTemplate.from_template("Analytical Question: {input}"),
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

SUMMARY_PROMPT_TEMPLATES = {
    "default": {
        "map_prompt": """
        The following is a set of documents:
{text}
Please identify the key points and write a succinct, bullet-point summary highlighting the main themes and important details.
Helpful Answer:
""",
        "reduce_prompt": """
        The following are summaries in bullet points:
{text}
Take these bullet points and consolidate them into a final, concise set of key points, maintaining clarity and brevity.
Helpful Answer:
""",
    },
    "executive": {
        "map_prompt": """
        The following is a set of documents:
{text}
Please summarize the key takeaways in a high-level, executive summary. Focus on major themes and important points, and keep it concise.
Helpful Answer:
""",
        "reduce_prompt": """
        The following are executive summaries:
{text}
Consolidate these summaries into a single, high-level executive summary with a focus on key takeaways and clarity.
Helpful Answer:
""",
    },
    "thematic": {
        "map_prompt": """
        The following is a set of documents:
{text}
Please identify and summarize the main themes in the documents. Group related points and focus on the core topics discussed.
Helpful Answer:
""",
        "reduce_prompt": """
        The following are thematic summaries:
{text}
Take these thematic summaries and distill them into a final summary that clearly outlines the main themes and related topics.
Helpful Answer:
""",
    },
    "narrative": {
        "map_prompt": """
        The following is a set of documents:
{text}
Please summarize the documents into a cohesive narrative, focusing on key themes and presenting the information in an engaging story-like format.
Helpful Answer:
""",
        "reduce_prompt": """
        The following are narrative summaries:
{text}
Consolidate these narrative summaries into a final cohesive story, ensuring all important points are captured clearly.
Helpful Answer:
""",
    },
    "comparative": {
        "map_prompt": """
        The following is a set of documents:
{text}
Please provide a comparative summary, highlighting the similarities and differences between the documents. Focus on contrasting themes and major points.
Helpful Answer:
""",
        "reduce_prompt": """
        The following are comparative summaries:
{text}
Take these comparative summaries and create a final consolidated summary, emphasizing the key differences and similarities.
Helpful Answer:
""",
    },
    "detailed": {
        "map_prompt": """
        The following is a set of documents:
{text}
Please create a detailed summary, breaking down each document’s main points, explaining key concepts, and including significant details.
Helpful Answer:
""",
        "reduce_prompt": """
        The following are detailed summaries:
{text}
Consolidate these detailed summaries into a single, comprehensive summary, ensuring all relevant points are included clearly.
Helpful Answer:
""",
    },
}
