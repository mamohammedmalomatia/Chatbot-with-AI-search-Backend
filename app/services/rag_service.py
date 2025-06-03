from typing import List, Dict, Any, Optional, Union
from langchain_community.vectorstores import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from neo4j import GraphDatabase
# from .neo4j_service import neo4j_service  
import os
from dotenv import load_dotenv
import logging
from azure.search.documents.aio import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class RAGService:
    def __init__(self):
        # Initialize Azure Search Client
        self.search_client = SearchClient(
            endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
            index_name=os.getenv('AZURE_SEARCH_INDEX_NAME'),
            credential=AzureKeyCredential(os.getenv('AZURE_SEARCH_API_KEY'))
        )
        
        # Initialize Azure OpenAI for chat
        self.chat_model = AzureChatOpenAI(
            deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
            openai_api_version=os.getenv('OPENAI_API_VERSION'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('OPENAI_API_KEY')
        )
        # Neo4j initialization
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI'),
            auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
        )
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
    async def _azure_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search using Azure AI Search"""
        # try:
        logging.info(f"Executing Azure Search with query: {query}")
        results = []

        # Create a temporary client
        search_client = SearchClient(
            endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
            index_name=os.getenv('AZURE_SEARCH_INDEX_NAME'),
            credential=AzureKeyCredential(os.getenv('AZURE_SEARCH_API_KEY'))
        )

        async with search_client:
            search_results = await search_client.search(
                search_text=query,
                query_type="semantic",
                top=3,
            )

            async for result in search_results:
                results.append({
                    "id": result.get('id', ''),
                    "title": result.get('title_Data_Column', ''),
                    "policy_type": result.get('policy_type', ''),
                    "department": result.get('department', ''),
                    "date": result.get('date', ''),
                    "content": result.get('chunk', ''),
                    "score": result.get('@search.score', 0.0),
                    "page_number": result.get('page_number', 0),
                    "source": result.get('source', ''),
                })

        # logging.info(f"Search results: {results}")
        return results
        # except Exception as e:
        #     logging.error(f"Error during Azure Search: {str(e)}")
        #     return []

        
    async def _neo4j_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search using Neo4j Knowledge Graph"""
        with self.neo4j_driver.session() as session:
            cypher_query = """
                CALL db.index.fulltext.queryNodes("documentIndex", $query)
                YIELD node, score
                WHERE node:Document
                RETURN 
                    node.id AS id,
                    node.title AS title,
                    node.description AS description,
                    node.source AS source,
                    node.author AS author,
                    node.department AS department,
                    node.date AS date,
                    node.content AS content,
                    score
                ORDER BY score DESC
                LIMIT 10
            """
            result = session.run(cypher_query, query=query)
            return [{
                "id": record["id"],
                "title": record["title"],
                "description": record["description"],
                "source": record["source"],
                "author": record["author"],
                "department": record["department"],
                "metadata": record["metadata"],
                "date": record["date"],
                "content": record["content"],
                "score": record["score"],
                "search_type": "neo4j"
            } for record in result]

    async def _combined_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining Azure and Neo4j results"""
        azure_results = await self._azure_search(query, filters)
        neo4j_results = await self._neo4j_search(query, filters)
        
        # Merge and deduplicate results
        combined = {}
        for result in azure_results + neo4j_results:
            doc_id = result["id"]
            if doc_id not in combined:
                combined[doc_id] = result
                combined[doc_id]["combined_score"] = result["score"]
            else:
                # Average scores for duplicates
                combined[doc_id]["combined_score"] = (
                    combined[doc_id]["combined_score"] + result["score"]
                ) / 2
        
        return sorted(
            list(combined.values()),
            key=lambda x: x["combined_score"],
            reverse=True
        )
    def _build_filter_string(self, filters: Dict[str, Any]) -> str:
        """Build Azure Search filter string"""
        filter_parts = []
        for field, value in filters.items():
            if field == "date":
                filter_parts.append(f"{field} ge {value}")
            elif isinstance(value, list):
                filter_parts.append(f"{field}/any(t: t eq '{value[0]}')")
            else:
                filter_parts.append(f"{field} eq '{value}'")
        return " and ".join(filter_parts)

    def _rewrite_query_with_history(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Rewrite ambiguous queries using chat history context via LLM."""
        if not chat_history or not isinstance(chat_history, list) or len(chat_history) == 0:
            return query
        
        try:
            prompt = (
                "You are an AI assistant. The user has asked a follow-up or ambiguous question. "
                "Rewrite the user's question to be fully self-contained and explicit and make it breif as well, using the previous chat history for context. "
                "Do not answer the question, just rewrite it.\n\n"
                f"Chat History:\n{format_chat_history(chat_history)}\n\n"
                f"User's Question: {query}\n\n"
                "Rewritten Question:"
            )
            response = self.chat_model.predict(prompt)
            rewritten = response.strip()
            # If the LLM returns an empty or too short response, fallback to original query
            if not rewritten or len(rewritten) < 5:
                return query
            return rewritten
        except Exception as e:
            logger.error(f"Error rewriting query: {str(e)}")
            return query

    async def get_chat_response(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None, search_type: str = "ai_search"):
        # Rewrite query if chat history exists
        if chat_history:
            query = self._rewrite_query_with_history(query, chat_history)
        # Retrieve docs from Azure Search
        documents = await self._azure_search(query)

        # Build the context string from document contents + metadata (limit length if needed)
        context_text = "\n\n---\n\n".join(
            [f"Content: {doc['content']}" for doc in documents]
        )

        # Prepare prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=(
                "You are an assistant at Qatar Research Development and Innovation Council(QRDI). Use the following documents to answer the question.\n\n"
                "Answer the question breifly and accurately based on the context provided.\n\n"
                "Context: {context}\n\nQuestion: {question}\nAnswer:.\n\n"
                "Chat History: {chat_history}.\n\n"
            )
        )
        
        # Optionally add conversation memory if you want
       

        # Create prompt with context and question
        prompt = prompt_template.format(context=context_text, question=query, chat_history=format_chat_history(chat_history))
        
        
        # Call LLM
        response = self.chat_model.invoke(prompt)
        
        for doc in documents:
            print(doc)

        # Format sources info
        sources = [
            {
                "title": doc.get("title"),
                "source": doc.get("source"),
                "relevance": doc.get("score", 0.0),
                "page": doc.get("page_number", 0)
            }
            for doc in documents
        ]

        return {
            "answer": response.content,
            "sources": sources,
        }


    async def _generate_follow_up_questions(self, query: str, answer: str) -> List[str]:
        """Generate follow-up questions based on the chat response."""
        
        documents = await self._azure_search(query)
        # Build the context string from document contents + metadata (limit length if needed)
        context_text = "\n\n---\n\n".join(
            [f"Content: {doc['content']}" for doc in documents]
        )
        try:
            prompt = f"""
            Based on the following question and answer, context and generate provided, generate 3 relevant follow-up questions and make it breif as well.
            Question: {query}
            Answer: {answer}
            Context: {context_text}
            Generate questions that:
            1. Seek clarification on specific points
            2. Explore related topics
            3. Request more detailed information
            
            Return only the questions, one per line.
            
            Do not return in numbered list.
            
            Do not return follow up questions that are already answered in the context.
            
            Do not return follow up questions that are not related to the context.
            """
            
            response = self.chat_model.predict(prompt)
            questions = [q.strip() for q in response.split('\n') if q.strip()]
            return questions[:3]  # Return top 3 questions
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {str(e)}")
            return []
# Create a singleton instance
rag_service = RAGService()

async def process_query(request):
    return await rag_service.process_query(
        query=request.query,
        rag_type=request.rag_type,
        stream=request.stream
    )
    
def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """Format chat history for prompt input."""
    formatted = []
    for msg in chat_history:
        if 'user' in msg:
            formatted.append(f"User: {msg['user']}")
        elif 'assistant' in msg:
            formatted.append(f"Assistant: {msg['assistant']}")
    return "\n".join(formatted)
