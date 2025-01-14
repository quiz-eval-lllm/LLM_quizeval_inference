import os
import textwrap
from pathlib import Path
import asyncio
import pdfplumber
import logging
from dotenv import load_dotenv
import re

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
import shutil


# Load environment variables
load_dotenv()

# Configure Groq
parser_key = os.getenv("PARSER_KEY")
llm_api = os.getenv("LLM_KEY")

# Global Model Initialization
EMBEDDINGS_MODEL = None

# Initialize global embeddings and retriever at application start.


def initialize_global_resources(embeddings_model_name="BAAI/bge-base-en-v1.5"):
    global EMBEDDINGS_MODEL, RETRIEVER

    if EMBEDDINGS_MODEL is None:
        EMBEDDINGS_MODEL = FastEmbedEmbeddings(
            model_name=embeddings_model_name)


# Delete previous RAG


def reset_vectorstore_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
        logging.info("[2] Deleting Vectorstore")
    os.makedirs(directory_path, exist_ok=True)


# Parsing PDF to text


def parse_pdf_to_text(pdf_path):
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        logging.info("[1] Parsing PDF")
        return text
    except Exception as e:
        logging.info(f"Error parsing PDF: {e}")
        return ""

# Setup for model and retriever


def setup_model_and_retriever(parsed_text):

    try:
        vectorstore_directory = "./db"

        # Reset vectorstore directory
        reset_vectorstore_directory(vectorstore_directory)

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

        # Use the globally initialized embeddings
        logging.info("[3] Initialize Embeddings")

        qdrant = Qdrant.from_documents(
            docs,
            EMBEDDINGS_MODEL,
            # location=":memory:",
            path="./db",
            collection_name="document_embeddings",
        )

        # Set up the retriever with compression
        logging.info("[4] Managing Retriever")
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
    logging.info("[6] Response parsing")

    result = response['result']
    lines = result.split("\n")
    soal_pg = []
    options_pg = []
    jawaban_pg = []
    current_question = None
    current_options = []
    correct_option_letter = None

    for line in lines:
        line = line.strip()

        # Encode the line to UTF-8 and decode it back to ensure consistent encoding
        line = line.encode('utf-8', 'replace').decode('utf-8')

        # Identify a new question
        if re.match(r"^\d+\. (Pertanyaan|Question):", line):
            # Save the previous question and options
            if current_question and current_options and correct_option_letter:
                soal_pg.append(current_question)
                options_pg.append(current_options)
                correct_index = ord(correct_option_letter) - ord('A')
                jawaban_pg.append(current_options[correct_index])
                current_options = []
                correct_option_letter = None

            # Extract the question text
            current_question = re.split(
                r"Pertanyaan:|Question:", line, maxsplit=1)[1].strip()

        # Identify options
        elif re.match(r"^[A-D]\.", line):  # Match options like "A.", "B.", "C.", "D."
            # Remove the "A.", "B.", etc., and store only the option text
            option_text = re.split(r"^[A-D]\.\s*", line, maxsplit=1)[1].strip()
            current_options.append(option_text)

        # Identify the answer
        elif "Jawaban:" in line or "Answer:" in line:
            match = re.search(
                r"Jawaban:\s*([A-D])", line) if "Jawaban:" in line else re.search(r"Answer:\s*([A-D])", line)
            if match:
                correct_option_letter = match.group(1)
            else:
                logging.error(
                    f"Failed to parse correct option from line: {line}")
                correct_option_letter = None

    # Append the last question and options
    if current_question and current_options and correct_option_letter:
        soal_pg.append(current_question)
        options_pg.append(current_options)
        correct_index = ord(correct_option_letter) - ord('A')
        jawaban_pg.append(current_options[correct_index])

    return soal_pg, options_pg, jawaban_pg

# Parsing Essay


def parse_essay(response):
    logging.info("[6] Response parsing")

    result = response['result']
    lines = result.split("\n")
    soal_essay = []
    empty = []
    jawaban_essay = []

    for line in lines:
        # Encode the line to UTF-8 and decode it back to ensure consistent encoding
        line = line.encode('utf-8', 'replace').decode('utf-8')

        # Identify essay questions
        if "Pertanyaan:" in line or "Question:" in line:
            if "Pertanyaan:" in line:
                soal_essay.append(line.split("Pertanyaan:")[1].strip())
            elif "Question:" in line:
                soal_essay.append(line.split("Question:")[1].strip())

        # Identify essay answers
        elif "Jawaban:" in line or "Answer:" in line:
            if "Jawaban:" in line:
                jawaban_essay.append(line.split("Jawaban:")[1].strip())
            elif "Answer:" in line:
                jawaban_essay.append(line.split("Answer:")[1].strip())

    # Ensure all outputs are properly encoded in UTF-8
    soal_essay = [q.encode('utf-8', 'replace').decode('utf-8')
                  for q in soal_essay]
    jawaban_essay = [a.encode('utf-8', 'replace').decode('utf-8')
                     for a in jawaban_essay]

    return soal_essay, empty, jawaban_essay


async def generate_quiz_question(package):

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
    llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768",
                   groq_api_key=llm_api)
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=compression_retriever)

    # Genrating multiple choice quiz
    logging.info("[5] LLM Response Generating")
    if question_type == 0:
        prompt = "Generate 5 Multiple Choice Question from Context" + \
            str(prompt)

        # Quiz for indonesia
        if (language == "id"):
            response_mcq = generate_questions(prompt, "mcq", "id", qa)

        # Quiz for english
        if (language == "en"):
            response_mcq = generate_questions(prompt, "mcq", "en", qa)

        questions, options, answers = parse_mcq(response_mcq)

        return questions, options, answers

    # Genrating essay quiz
    elif question_type == 1:
        prompt = "Generate 5 Essay Question from Context" + str(prompt)

        # Quiz for indonesia
        if (language == "id"):
            response_essay = generate_questions(prompt, "essay", "id", qa)

        # Quiz for english
        if (language == "en"):
            response_essay = generate_questions(prompt, "essay", "en", qa)

        questions, options, contexts = parse_essay(response_essay)

        return questions, options, contexts

    else:
        logging.info("Error occurs when generating quiz")
        return [], []
