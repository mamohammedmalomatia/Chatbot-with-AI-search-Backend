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
            index_name="rag-1748934332165",#os.getenv('AZURE_SEARCH_INDEX_NAME'),
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

    async def get_suggestions(self, query: str) -> List[Dict[str, Any]]:
        """Get keyword suggestions for a given query."""
        try:
            search_results = await self._azure_search(query)
            seen = set()
            suggestions = []

            query_lower = query.lower()
            logging.info(f"Search results: {search_results}")
            for result in search_results:
                keywords = result.get('key_words', [])
                if not isinstance(keywords, list):
                    continue

                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    if query_lower in keyword_lower and keyword_lower not in seen:
                        seen.add(keyword_lower)
                        suggestions.append({
                            'text': keyword,
                            'type': 'keyword',
                            'score': result.get('@search.score', 0.0),
                        })

            # Sort by score descending
            suggestions.sort(key=lambda x: x['score'], reverse=True)

            return suggestions

        except Exception as e:
            logging.error(f"Failed to get suggestions: {e}")
            return []

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
            index_name="rag-1748934332165",#os.getenv('AZURE_SEARCH_INDEX_NAME'),
            credential=AzureKeyCredential(os.getenv('AZURE_SEARCH_API_KEY')),
            api_version="2025-05-01-preview"
        )

        filter_string = self._build_filter_expression(filters) if filters else None
        logging.info(f"Filter string: {filter_string}")
        async with search_client:
            search_results = await search_client.search(
                search_text=query,
                filter=filter_string if filter_string else "",
                query_type="semantic",
                top=10,
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
                    "source": result.get('source', ''),
                    "key_words": result.get('key_words', ''),
                    "page_number": result.get('page_number', 0)
                })

        # logging.info(f"Search results: {results}")
        return results
        # except Exception as e:
        #     logging.error(f"Error during Azure Search: {str(e)}")
        #     return []

    def _build_filter_expression(self, filters: Dict[str, Any]) -> str:
        """Convert filters for 'department' and 'policy_type' into OData expression"""
        expressions = []

        for field in ['department', 'policy_type']:
            values = filters.get(field)
            if values:
                if isinstance(values, list):
                    if len(values) == 1:
                        expressions.append(f"{field} eq '{values[0]}'")
                    else:
                        or_clause = " or ".join(f"{field} eq '{v}'" for v in values)
                        expressions.append(f"({or_clause})")

        return " and ".join(expressions)
        
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

    async def get_chat_response(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None, search_type: str = "ai_search"):
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
                "You are an AI assistant. Use the following documents to answer the question.\n\n"
                "{context}\n\nQuestion: {question}\nAnswer:.\n\n"
                "Chat History: {chat_history}"
            )
        )
        
        # Optionally add conversation memory if you want
       

        # Create prompt with context and question
        prompt = prompt_template.format(context=context_text, question=query, chat_history=format_chat_history(chat_history))
        
        
        # Call LLM
        response = self.chat_model.invoke(prompt)

        # Format sources info
        sources = [
            {
                "title": doc.get("title"),
                "source": doc.get("source"),
                "relevance": doc.get("score", 0.0)
            }
            for doc in documents
        ]

        return {
            "answer": response.content,
            "sources": sources,
            "follow_up_questions": self._generate_follow_up_questions(query, response.content)
        }


    def _generate_follow_up_questions(self, query: str, answer: str) -> List[str]:
        """Generate follow-up questions based on the chat response."""
        try:
            prompt = f"""
            Based on the following question and answer, generate 3 relevant follow-up questions.
            Question: {query}
            Answer: {answer}
            
            Generate questions that:
            1. Seek clarification on specific points
            2. Explore related topics
            3. Request more detailed information
            
            Return only the questions, one per line.
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
