"""
RAG (Retrieval-Augmented Generation) service for API recommendation.
Handles document embedding, retrieval, and reranking operations.
"""

import os
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import jieba

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sklearn.metrics.pairwise import cosine_similarity

from .config import Config
from .utils import rank_apis_by_frequency, validate_api_object


class RAGService:
    """
    RAG service for retrieving and ranking relevant APIs based on mashup descriptions.
    Combines vector search, BM25, and cross-encoder reranking for optimal results.
    """
    
    def __init__(self, config: Config):
        """
        Initialize RAG service with configuration.
        
        Args:
            config (Config): Configuration object containing model and retrieval parameters
        """
        self.config = config
        self.embeddings = None
        self.vector_db = None
        self.api_embed_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding models and API similarity model."""
        print("Initializing embedding models...")
        
        # Initialize HuggingFace embeddings for document retrieval
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.model.embed_model_path,
            show_progress=self.config.model.show_progress,
            model_kwargs={
                "trust_remote_code": self.config.model.trust_remote_code,
            },
        )
        
        # Initialize SentenceTransformer for API similarity computation
        # Force to use local cache only to avoid network requests
        try:
            self.api_embed_model = SentenceTransformer(
                self.config.model.api_embed_model_name,
                local_files_only=True
            )
        except Exception as e:
            print(f"Failed to load {self.config.model.api_embed_model_name} from local cache: {e}")
            print("Trying without local_files_only restriction...")
            self.api_embed_model = SentenceTransformer(
                self.config.model.api_embed_model_name
            )
        
        print("Models initialized successfully")
    
    def create_document_store(self, mashups: List[Dict[str, Any]]) -> List[Document]:
        """
        Create document store from mashup data.
        
        Args:
            mashups (List[Dict[str, Any]]): List of mashup objects
            
        Returns:
            List[Document]: List of LangChain Document objects
        """
        docs = []
        
        for mashup in mashups:
            content = mashup.get("description", "")
            doc = Document(
                page_content=content, 
                metadata={"title": mashup.get("title", "")}
            )
            docs.append(doc)
        
        print(f"Created {len(docs)} documents from mashups")
        return docs
    
    def build_vector_database(self, documents: List[Document]) -> FAISS:
        """
        Build and save FAISS vector database from documents.
        
        Args:
            documents (List[Document]): List of documents to index
            
        Returns:
            FAISS: FAISS vector database instance
        """
        print("Building vector database...")
        
        # Create FAISS database
        db = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )
        
        # Save database locally
        db.save_local(self.config.paths.vectordb_dir)
        print(f"Vector database saved to {self.config.paths.vectordb_dir}")
        
        return db
    
    def create_ensemble_retriever(self, documents: List[Document], vector_db: FAISS) -> EnsembleRetriever:
        """
        Create ensemble retriever combining BM25 and vector search.
        
        Args:
            documents (List[Document]): Documents for BM25 indexing
            vector_db (FAISS): Vector database for semantic search
            
        Returns:
            EnsembleRetriever: Combined retriever instance
        """
        print("Creating ensemble retriever...")
        
        # Create vector retriever
        vector_retriever = vector_db.as_retriever(
            search_kwargs={"k": self.config.retrieval.initial_k}
        )
        
        # Create BM25 retriever with Chinese tokenization
        bm25_retriever = BM25Retriever.from_documents(
            documents,
            k=self.config.retrieval.initial_k,
            bm25_params={
                "k1": self.config.retrieval.bm25_k1, 
                "b": self.config.retrieval.bm25_b
            },
            preprocess_func=jieba.lcut,
        )
        
        # Combine retrievers with configured weights
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever], 
            weights=[
                self.config.retrieval.bm25_weight,
                self.config.retrieval.vector_weight
            ]
        )
        
        print("Ensemble retriever created successfully")
        return ensemble_retriever
    
    def create_compression_retriever(self, base_retriever: EnsembleRetriever) -> ContextualCompressionRetriever:
        """
        Create compression retriever with cross-encoder reranking.
        
        Args:
            base_retriever (EnsembleRetriever): Base ensemble retriever
            
        Returns:
            ContextualCompressionRetriever: Compression retriever with reranking
        """
        print("Creating compression retriever with reranking...")
        
        # Initialize cross-encoder reranking model
        # Try to load from local cache first to avoid network requests
        try:
            rerank_model = HuggingFaceCrossEncoder(
                model_name=self.config.model.rerank_model_name,
                local_files_only=True
            )
        except Exception as e:
            print(f"Failed to load {self.config.model.rerank_model_name} from local cache: {e}")
            print("Trying without local_files_only restriction...")
            try:
                rerank_model = HuggingFaceCrossEncoder(
                    model_name=self.config.model.rerank_model_name
                )
            except Exception as e2:
                print(f"Failed to load rerank model completely: {e2}")
                print("Skipping reranking, using base retriever only...")
                return base_retriever
        
        # Create compressor with configured top-n
        compressor = CrossEncoderReranker(
            model=rerank_model, 
            top_n=self.config.retrieval.rerank_top_n
        )
        
        # Create compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=base_retriever
        )
        
        print("Compression retriever created successfully")
        return compression_retriever
    
    def retrieve_relevant_mashups(self, queries: List[Dict[str, Any]], 
                                retriever: ContextualCompressionRetriever) -> List[List[str]]:
        """
        Retrieve relevant mashups for given queries.
        
        Args:
            queries (List[Dict[str, Any]]): List of query objects with 'description' field
            retriever (ContextualCompressionRetriever): Retriever instance
            
        Returns:
            List[List[str]]: List of relevant mashup titles for each query
        """
        print("Retrieving relevant mashups...")
        rerank_answers = []
        
        for query in tqdm(queries, desc="Processing queries"):
            content = query.get("description", "")
            relevant_docs = retriever.invoke(content)
            
            # Extract mashup titles from retrieved documents
            relevant_mashups = [doc.metadata["title"] for doc in relevant_docs]
            rerank_answers.append(relevant_mashups)
        
        return rerank_answers
    
    def extract_apis_from_mashups(self, mashup_results: List[List[str]], 
                                 all_mashups: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Extract and rank APIs from retrieved mashups.
        
        Args:
            mashup_results (List[List[str]]): Retrieved mashup titles for each query
            all_mashups (List[Dict[str, Any]]): Complete mashup dataset
            
        Returns:
            List[List[Dict[str, Any]]]: Ranked API lists for each query
        """
        print("Extracting APIs from mashups...")
        api_results = []
        
        # Create mashup lookup dictionary for efficient access
        mashup_lookup = {mashup["title"]: mashup for mashup in all_mashups}
        
        for mashup_titles in mashup_results:
            api_doc_set = []
            
            # Extract APIs from each retrieved mashup
            for title in mashup_titles:
                mashup = mashup_lookup.get(title)
                if not mashup:
                    continue
                
                if "related_apis" in mashup and mashup["related_apis"]:
                    for api in mashup["related_apis"]:
                        if validate_api_object(api):
                            api_json = {
                                "title": api["title"],
                                "tags": api["tags"],
                            }
                            api_doc_set.append(api_json)
            
            # Rank APIs by frequency and remove duplicates
            sorted_apis = rank_apis_by_frequency(api_doc_set)
            api_results.append(sorted_apis)
        
        return api_results
    
    def rerank_apis_by_similarity(self, api_results: List[List[Dict[str, Any]]], 
                                 queries: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Rerank APIs based on tag similarity with query categories and tags.
        
        Args:
            api_results (List[List[Dict[str, Any]]]): API results from mashup retrieval
            queries (List[Dict[str, Any]]): Original query objects with categories and tags
            
        Returns:
            List[List[Dict[str, Any]]]: Reranked API results
        """
        print("Reranking APIs by similarity...")
        final_results = []
        
        for index, api_doc_set in enumerate(tqdm(api_results, desc="Reranking APIs")):
            if len(api_doc_set) == 0:
                final_results.append([])
                continue
            
            # Get query categories and tags
            query = queries[index]
            categories = query.get("categories", [])
            tags = query.get("tags", [])
            
            # Combine categories and tags for embedding
            query_text = ", ".join(categories + tags)
            
            if not query_text.strip():
                # If no categories/tags, return original ranking
                final_results.append(api_doc_set[:self.config.retrieval.final_api_limit])
                continue
            
            # Generate embedding for query
            query_embedding = self.api_embed_model.encode(query_text)
            
            # Calculate similarity scores for APIs
            similarity_scores = []
            for api_doc in api_doc_set:
                api_tags_text = ", ".join(api_doc.get("tags", []))
                
                if api_tags_text.strip():
                    api_embedding = self.api_embed_model.encode(api_tags_text)
                    similarity = cosine_similarity([query_embedding], [api_embedding])[0][0]
                else:
                    similarity = 0.0
                
                similarity_scores.append((api_doc, similarity))
            
            # Sort by similarity and combine with frequency-based ranking
            sorted_by_similarity = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            
            # Take top APIs by frequency (ordered by mashup retrieval)
            ordered_apis = api_doc_set[:self.config.retrieval.ordered_apis_limit]
            
            # Add top APIs by similarity that aren't already in ordered list
            ordered_titles = {api["title"] for api in ordered_apis}
            similarity_apis = [
                api[0] for api in sorted_by_similarity 
                if api[0]["title"] not in ordered_titles
            ][:self.config.retrieval.similarity_top_n]
            
            # Combine and limit final results
            final_api_list = ordered_apis + similarity_apis
            final_api_list = final_api_list[:self.config.retrieval.final_api_limit]
            
            final_results.append(final_api_list)
        
        return final_results
    
    def run_rag_pipeline(self, mashups: List[Dict[str, Any]], 
                        queries: List[Dict[str, Any]]) -> Tuple[List[List[Dict[str, Any]]], int]:
        """
        Run the complete RAG pipeline for API recommendation.
        
        Args:
            mashups (List[Dict[str, Any]]): Complete mashup dataset
            queries (List[Dict[str, Any]]): Query objects with description, categories, tags
            
        Returns:
            Tuple[List[List[Dict[str, Any]]], int]: 
                - Recommended APIs for each query
                - Total character count processed
        """
        print("Starting RAG pipeline...")
        
        # Step 1: Create documents and build vector database
        documents = self.create_document_store(mashups)
        vector_db = self.build_vector_database(documents)
        
        # Step 2: Create ensemble retriever with BM25 + vector search
        ensemble_retriever = self.create_ensemble_retriever(documents, vector_db)
        
        # Step 3: Add cross-encoder reranking
        compression_retriever = self.create_compression_retriever(ensemble_retriever)
        
        # Step 4: Retrieve relevant mashups
        mashup_results = self.retrieve_relevant_mashups(queries, compression_retriever)
        
        # Step 5: Extract APIs from retrieved mashups
        api_results = self.extract_apis_from_mashups(mashup_results, mashups)
        
        # Step 6: Rerank APIs by tag similarity
        final_results = self.rerank_apis_by_similarity(api_results, queries)
        
        # Calculate total characters processed (placeholder)
        total_characters = sum(len(str(result)) for result in final_results)
        
        print("RAG pipeline completed successfully")
        return final_results, total_characters
