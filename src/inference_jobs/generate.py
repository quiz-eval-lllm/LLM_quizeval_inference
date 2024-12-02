import os
import textwrap
from pathlib import Path
import asyncio
import pdfplumber
import logging
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores import Qdrant
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from llama_parse import LlamaParse
from langchain.vectorstores import utils as chromautils
from inference_jobs.generate_prompt_template import mcq_prompt_en, mcq_prompt_id, essay_prompt_en, essay_prompt_id

# Load environment variables
load_dotenv()

# Configure Groq
parser_key = os.getenv("PARSER_KEY")
groq_api = os.getenv("GROQ_API")

# Parsing PDF to text


def parse_pdf_to_text(pdf_path):
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        logging.info(f"Error parsing PDF: {e}")
        return ""

# Setup for model and retriever


def setup_model_and_retriever(parsed_text, embeddings_model_name="BAAI/bge-base-en-v1.5"):
    """
    Set up the document, splitter, retriever, and compressor for the QA chain.
    """
    try:
        # Save the processed text to a markdown file
        document_path = Path("data/parsed_document.md")
        document_path.parent.mkdir(parents=True, exist_ok=True)
        document_path.write_text(
            parsed_text, encoding='utf-8', errors='ignore')

        # Load the markdown document
        loader = UnstructuredMarkdownLoader(document_path)
        loaded_documents = loader.load()

        # Split the document into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=128)
        docs = text_splitter.split_documents(loaded_documents)
        # docs = chromautils.filter_complex_metadata(docs)

        # Use the FastEmbed model for embeddings
        embeddings = FastEmbedEmbeddings(model_name=embeddings_model_name)
        # db = Chroma.from_documents(docs, embeddings, persist_directory="./db")

        qdrant = Qdrant.from_documents(
            docs,
            embeddings,
            # location=":memory:",
            path="./db",
            collection_name="document_embeddings",
        )

        # Set up the retriever with compression
        retriever = qdrant.as_retriever(search_kwargs={"k": 5})
        compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    except Exception as e:
        logging.info(f"Error setting up model and retriever: {e}")
        return None

# Language handler for generate questions


def get_prompt_by_type_and_language(question_type: str, language: str) -> PromptTemplate:
    if question_type == "mcq":
        return mcq_prompt_en if language == "en" else mcq_prompt_id
    elif question_type == "essay":
        return essay_prompt_en if language == "en" else essay_prompt_id
    else:
        raise ValueError("Unsupported question type. Use 'mcq' or 'essay'.")

# Function to generate questions


def generate_questions(context: str, question_type: str, language: str, qa_instance):
    prompt = get_prompt_by_type_and_language(question_type, language)
    qa_instance.combine_documents_chain.llm_chain.prompt = prompt
    return qa_instance.invoke(context)

# Parsing and printing functions


def parse_mcq(response):
    result = response['result']
    lines = result.split("\n")
    soal_pg = []
    jawaban_pg = []
    current_question = []

    for line in lines:
        if line.startswith("1. Pertanyaan:") or line.startswith("1. Question:") or \
           line.startswith("2. Pertanyaan:") or line.startswith("2. Question:") or \
           line.startswith("3. Pertanyaan:") or line.startswith("3. Question:") or \
           line.startswith("4. Pertanyaan:") or line.startswith("4. Question:") or \
           line.startswith("5. Pertanyaan:") or line.startswith("5. Question:"):
            if current_question:
                soal_pg.append("\n".join(current_question))
                current_question = []
            current_question.append(line.split("Pertanyaan:")[1].strip(
            ) if "Pertanyaan:" in line else line.split("Question:")[1].strip())
        elif line.strip().startswith("A.") or line.strip().startswith("B.") or line.strip().startswith("C.") or line.strip().startswith("D."):
            current_question.append(line.strip())
        elif line.startswith("   Jawaban:") or line.startswith("   Answer:"):
            jawaban_pg.append(line.split("Jawaban:")[1].strip(
            ) if "Jawaban:" in line else line.split("Answer:")[1].strip())
    if current_question:
        soal_pg.append("\n".join(current_question))

    return soal_pg, jawaban_pg

# Parsing Essay


def parse_essay(response):
    result = response['result']
    lines = result.split("\n")
    soal_essay = []
    jawaban_essay = []
    for line in lines:
        if "Pertanyaan:" in line or "Question:" in line:
            soal_essay.append(line.split("Pertanyaan:")[1].strip(
            ) if "Pertanyaan:" in line else line.split("Question:")[1].strip())
        elif "Jawaban:" in line or "Answer:" in line:
            jawaban_essay.append(line.split("Jawaban:")[1].strip(
            ) if "Jawaban:" in line else line.split("Answer:")[1].strip())

    return soal_essay, jawaban_essay


async def generate_quiz_question(package):
    """
    Generate quiz questions based on a package containing context (PDF or text).
    """
    pdf_path = package.get("pdf_path")
    question_type = package.get("type")
    language = package.get("lang")
    prompt = package.get("prompt") or ""

    # Parse PDF as context
    parsed_text = parse_pdf_to_text(pdf_path)
    if not parsed_text:
        logging.info("Failed to parse PDF.")
        return

    # Setup for RAG
    compression_retriever = setup_model_and_retriever(parsed_text)
    if not compression_retriever:
        logging.info("Failed to setup model and retriever.")
        return

    # Context Retrival
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192",
                   groq_api_key=groq_api)
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=compression_retriever)

    # Genrating multiple choice quiz
    if question_type == 0:
        prompt = "Generate 5 Multiple Choice Question from Context" + \
            str(prompt)

        # Quiz for indonesia
        if (language == "id"):
            response_mcq = generate_questions(prompt, "mcq", "id", qa)

        # Quiz for english
        if (language == "en"):
            response_mcq = generate_questions(prompt, "mcq", "en", qa)

        questions, answers = parse_mcq(response_mcq)

        return questions, answers

    # Genrating essay quiz
    elif question_type == 1:
        prompt = "Generate 5 Essay Question from Context" + str(prompt)

        # Quiz for indonesia
        if (language == "id"):
            response_essay = generate_questions(prompt, "essay", "id", qa)

        # Quiz for english
        if (language == "en"):
            response_essay = generate_questions(prompt, "essay", "en", qa)

        questions, contexts = parse_essay(response_essay)

        return questions, contexts

    else:
        logging.info("Error occurs when generating quiz")
        return [], []
