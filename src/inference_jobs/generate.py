import os
import textwrap
from pathlib import Path
import asyncio
import pdfplumber
import logging
import torch
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
llm_api = os.getenv("LLM_KEY")

# Check if CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your GPU setup.")

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
        embeddings = FastEmbedEmbeddings(
            model_name=embeddings_model_name, device="cuda")
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
    soal_pg = []         # Store questions
    options_pg = []      # Store options as lists
    jawaban_pg = []      # Store correct answer text
    current_question = None
    current_options = []
    correct_option_letter = None

    for line in lines:
        line = line.strip()  # Clean leading/trailing whitespaces

        # Encode the line to UTF-8 and decode it back to ensure consistent encoding
        line = line.encode('utf-8', 'replace').decode('utf-8')

        # Identify a new question
        if line.startswith("1. Pertanyaan:") or line.startswith("1. Question:") or \
           line.startswith("2. Pertanyaan:") or line.startswith("2. Question:") or \
           line.startswith("3. Pertanyaan:") or line.startswith("3. Question:") or \
           line.startswith("4. Pertanyaan:") or line.startswith("4. Question:") or \
           line.startswith("5. Pertanyaan:") or line.startswith("5. Question:"):

            # Save the previous question and options
            if current_question and current_options and correct_option_letter:
                soal_pg.append(current_question.encode(
                    'utf-8', 'replace').decode('utf-8'))
                options_pg.append(
                    [opt.encode('utf-8', 'replace').decode('utf-8') for opt in current_options])
                # Map the correct answer letter to the corresponding option text
                correct_index = ord(correct_option_letter) - ord('A')
                jawaban_pg.append(current_options[correct_index].encode(
                    'utf-8', 'replace').decode('utf-8'))
                current_options = []  # Reset options for the next question
                correct_option_letter = None  # Reset correct answer letter

            # Extract the question text
            if "Pertanyaan:" in line:
                current_question = line.split("Pertanyaan:")[1].strip()
            elif "Question:" in line:
                current_question = line.split("Question:")[1].strip()

        # Identify options
        elif line.startswith("A.") or line.startswith("B.") or line.startswith("C.") or line.startswith("D."):
            # Add the entire option line (with "A.", "B.", etc.)
            current_options.append(line.strip())

        # Identify the answer
        elif line.startswith("Jawaban:") or line.startswith("Answer:"):
            correct_option_letter = line.split("Jawaban:")[1].strip(
            ) if "Jawaban:" in line else line.split("Answer:")[1].strip()

    # Append the last question and options
    if current_question and current_options and correct_option_letter:
        soal_pg.append(current_question.encode(
            'utf-8', 'replace').decode('utf-8'))
        options_pg.append([opt.encode('utf-8', 'replace').decode('utf-8')
                          for opt in current_options])
        correct_index = ord(correct_option_letter) - ord('A')
        jawaban_pg.append(current_options[correct_index].encode(
            'utf-8', 'replace').decode('utf-8'))

    return soal_pg, options_pg, jawaban_pg


# Parsing Essay


def parse_essay(response):
    result = response['result']
    lines = result.split("\n")
    soal_essay = []      # List to store essay questions
    # Placeholder for empty answers (as per function signature)
    empty = []
    jawaban_essay = []   # List to store essay answers

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
                   groq_api_key=llm_api, device="cuda")
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=compression_retriever)

    # Genrating multiple choice quiz
    if question_type == 0:
        prompt = "Generate 5 Multiple Choice Question fropim Context" + \
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
