from langchain.prompts import PromptTemplate

# MCQ Prompt Template for English
mcq_prompt_template_en = """
You are an AI assistant helping the user generate five multiple-choice questions (MCQs) based on the given context.

Each MCQ should include:
- A question related to the given context.
- Four options (A, B, C, D), one of which is the correct answer.
- The correct answer should be clearly marked.
- The Multiple Choice options are designed to be somewhat hard, with distractor answers that are plausible.
- Question and option in English.

Context: {context}

Generate the MCQs in this format:
1. Question: [Your question]
   Options:
   A. [Option A]
   B. [Option B]
   C. [Option C]
   D. [Option D]
   Answer: A

2. Question: [Your question]
   Options:
   A. [Option A]
   B. [Option B]
   C. [Option C]
   D. [Option D]
   Answer: B

... (repeat until 5 questions are generated)
"""
mcq_prompt_en = PromptTemplate(
    template=mcq_prompt_template_en, input_variables=["context"])

# MCQ Prompt Template for Indonesian
mcq_prompt_template_id = """
Anda adalah asisten AI yang membantu pengguna menghasilkan lima soal pilihan ganda (MCQ) berdasarkan konteks berikut.

Setiap MCQ harus mencakup:
- Sebuah pertanyaan yang relevan dengan konteks yang diberikan.
- Empat opsi jawaban (A, B, C, D), salah satu di antaranya adalah jawaban yang benar.
- Jawaban yang benar harus ditandai dengan jelas.
- Pilihan jawaban dirancang agar cukup sulit, dengan jawaban pengecoh yang masuk akal.
- Soal dan jawaban dalam Bahasa Indonesia.

Konteks: {context}

Hasilkan soal-soal MCQ dengan format berikut:
1. Pertanyaan: [Pertanyaan Anda]
   Pilihan:
   A. [Pilihan A]
   B. [Pilihan B]
   C. [Pilihan C]
   D. [Pilihan D]
   Jawaban: A

2. Pertanyaan: [Pertanyaan Anda]
   Pilihan:
   A. [Pilihan A]
   B. [Pilihan B]
   C. [Pilihan C]
   D. [Pilihan D]
   Jawaban: B

... (ulangi hingga 5 soal dibuat)
"""
mcq_prompt_id = PromptTemplate(
    template=mcq_prompt_template_id, input_variables=["context"])

# Essay Prompt Template for English
essay_prompt_template_en = """
You are an AI assistant helping the user generate five analytical essay questions based on the given context.

Each essay question should:
- Be challenging and require deep analysis.
- Be directly related to the provided context.
- Encourage critical thinking and detailed reasoning in the answer.
- Be written in English.

Context: {context}

Generate the essay questions in this format:
1. Question: [Your question]
   Answer: Short explanation correct answer

2. Question: [Your question]
   Answer: Short explanation correct answer

... (repeat until 5 questions are generated)
"""
essay_prompt_en = PromptTemplate(
    template=essay_prompt_template_en, input_variables=["context"])

# Essay Prompt Template for Indonesian
essay_prompt_template_id = """
Anda adalah asisten AI yang membantu pengguna menghasilkan lima soal essay analitis berdasarkan konteks yang diberikan.

Setiap soal essay harus:
- Bersifat menantang dan memerlukan analisis mendalam.
- Berhubungan langsung dengan konteks yang diberikan.
- Mendorong pemikiran kritis dan penalaran yang mendetail dalam jawaban.
- Ditulis dalam Bahasa Indonesia.

Konteks: {context}

Hasilkan soal essay dengan format berikut:
1. Pertanyaan: [Pertanyaan Anda]
   Jawaban: Penjelasan singkat jawaban benar

2. Pertanyaan: [Pertanyaan Anda]
   Jawaban: Penjelasan singkat jawaban benar

... (ulangi hingga 5 soal dibuat)
"""
essay_prompt_id = PromptTemplate(
    template=essay_prompt_template_id, input_variables=["context"])
