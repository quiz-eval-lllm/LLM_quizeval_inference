from langchain.prompts import PromptTemplate

mcq_en_prompt_template = """
### [INST] Instruction:
You are an AI assistant tasked with generating five advanced multiple-choice questions (MCQs) based on the given context.

Each MCQ must adhere to the following criteria:
- Questions should challenge the user's ability to analyze, apply, or evaluate the given context.
- Avoid questions that merely ask for definitions or simple facts ("What is...?"). Instead, focus on "Why", "How", or scenario-based questions that require deeper understanding.
- Use real-world scenarios or hypothetical situations that test the user's critical thinking skills.
- Provide four answer options (A, B, C, D), with one correct answer clearly marked.
- Distractors must be plausible, designed to test partial understanding or common misconceptions.
- Questions and answers must be written in English.
- Ensure no blank lines between questions and answers or between options.

Context: {context}

### Example of high-quality questions:
1. Question: Why do Gram-positive bacteria survive better in hypotonic environments compared to Gram-negative bacteria?
A) Their outer membrane acts as a selective barrier.
B) The thick peptidoglycan layer prevents osmotic lysis.
C) They lack teichoic acids, making them more flexible.
D) They have fewer porins, reducing water intake.
Answer: B. The thick peptidoglycan layer prevents osmotic lysis.

2. Question: In what ways could a mutation in the bacterial cell wall affect antibiotic resistance?
A) By reducing the binding affinity of antibiotics to the cell wall.
B) By increasing the permeability of the outer membrane to antibiotics.
C) By preventing the formation of biofilms on surfaces.
D) By accelerating the production of endospores.
Answer: A. By reducing the binding affinity of antibiotics to the cell wall.

### Now generate 5 analytical MCQs with the following format:
1. Question: [Your advanced question]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]
Answer: A. Option A
2. Question: [Your advanced question]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]
Answer: B. Option B
3. Question: [Your advanced question]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]
Answer: C. Option C
4. Question: [Your advanced question]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]
Answer: D. Option D
5. Question: [Your advanced question]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]
Answer: C. Option C
[/INST]
"""

mcq_prompt_en = PromptTemplate(
    template=mcq_en_prompt_template, input_variables=["context"]
)

# MCQ Prompt Template for Indonesian
mcq_id_prompt_template = """
### [INST] Instruction:
Anda adalah asisten AI yang bertugas untuk menghasilkan lima soal pilihan ganda (MCQ) tingkat lanjut berdasarkan konteks berikut.

Setiap MCQ harus mematuhi kriteria berikut:
- Pertanyaan harus menantang pemahaman pengguna dengan memerlukan analisis, penerapan, atau evaluasi dari konteks.
- Hindari pertanyaan yang hanya meminta definisi atau fakta sederhana ("Apa itu...?"). Sebaliknya, fokus pada "Mengapa", "Bagaimana", atau pertanyaan berbasis skenario yang membutuhkan pemahaman lebih dalam.
- Gunakan skenario dunia nyata atau situasi hipotetis untuk menguji keterampilan berpikir kritis pengguna.
- Berikan empat opsi jawaban (A, B, C, D), dengan satu jawaban yang benar ditandai dengan jelas.
- Distractor harus masuk akal dan dirancang untuk menguji pemahaman sebagian atau kesalahpahaman umum.
- Soal dan jawaban ditulis dalam Bahasa Indonesia.
- Tidak boleh ada baris kosong antara pertanyaan dan jawaban atau antara pilihan.

Konteks: {context}

### Contoh soal berkualitas tinggi:
1. Pertanyaan: Mengapa bakteri Gram-positif lebih tahan terhadap tekanan osmotik dibandingkan bakteri Gram-negatif?
A. Lapisan peptidoglikan tebal melindungi dari lisis osmotik.
B. Membran luar membantu mencegah tekanan.
C. Peptidoglikan mengurangi permeabilitas air.
D. Teichoic acid mengikat ion untuk mencegah lisis.
Jawaban: A. Lapisan peptidoglikan tebal melindungi dari lisis osmotik.

2. Pertanyaan: Bagaimana mekanisme kerja antibiotik seperti penisilin pada bakteri Gram-positif?
A. Mengganggu sintesis dinding sel dengan menghambat enzim transpeptidase.
B. Merusak membran plasma melalui ionisasi.
C. Menghambat sintesis protein di ribosom.
D. Memecah asam nukleat pada inti sel.
Jawaban: A. Mengganggu sintesis dinding sel dengan menghambat enzim transpeptidase.

### Sekarang buat 5 MCQ analitis dengan format berikut:
1. Pertanyaan: [Pertanyaan Anda yang tingkat lanjut]
A. [Pilihan A]
B. [Pilihan B]
C. [Pilihan C]
D. [Pilihan D]
Jawaban: B. Pilihan B
2. Pertanyaan: [Pertanyaan Anda yang tingkat lanjut]
A. [Pilihan A]
B. [Pilihan B]
C. [Pilihan C]
D. [Pilihan D]
Jawaban: C. Pilihan C
3. Pertanyaan: [Pertanyaan Anda yang tingkat lanjut]
A. [Pilihan A]
B. [Pilihan B]
C. [Pilihan C]
D. [Pilihan D]
Jawaban: D. Pilihan D
4. Pertanyaan: [Pertanyaan Anda yang tingkat lanjut]
A. [Pilihan A]
B. [Pilihan B]
C. [Pilihan C]
D. [Pilihan D]
Jawaban: A. Pilihan A
5. Pertanyaan: [Pertanyaan Anda yang tingkat lanjut]
A. [Pilihan A]
B. [Pilihan B]
C. [Pilihan C]
D. [Pilihan D]
Jawaban: C. Pilihan C
[/INST]
"""
mcq_prompt_id = PromptTemplate(
    template=mcq_id_prompt_template, input_variables=["context"]
)

essay_en_prompt_template = """
### [INST] Instruction:
You are an AI assistant tasked with helping users generate five different analytical essay questions based on the given context.

Each essay question must adhere to the following criteria:
- Questions should be challenging and require deep analysis.
- Questions must be directly related to the given context.
- Questions should encourage critical thinking and detailed reasoning in the answers.
- Questions and answers should be written in English.
- Each question must be unique, avoiding repetition in structure or focus (5 different questions).

Context:
{context}

### Generate 5 different analytical essay questions with a high level of difficulty or requiring deep analysis, along with concise answers in the following format:
1. Question: [Your Question]
Answer: Brief explanation of the correct answer
2. Question: [Your Question]
Answer: Brief explanation of the correct answer
3. Question: [Your Question]
Answer: Brief explanation of the correct answer
4. Question: [Your Question]
Answer: Brief explanation of the correct answer
5. Question: [Your Question]
Answer: Brief explanation of the correct answer
[/INST]
"""

essay_prompt_en = PromptTemplate(
    input_variables=["context"],
    template=essay_en_prompt_template,
)

essay_id_prompt_template = """
### [INST] Instruction:
Anda adalah asisten AI yang membantu pengguna menghasilkan lima soal essay berbeda yang analitis berdasarkan konteks yang diberikan.

Setiap soal essay harus mencakup ketentuan dibawah ini:
- Soal bersifat menantang dan memerlukan analisis mendalam.
- Soal berhubungan langsung dengan konteks yang diberikan.
- Soal mendorong pemikiran kritis dan penalaran yang mendetail dalam jawaban.
- Soal dan jawaban ditulis dalam Bahasa Indonesia.
- Setiap soal harus unik, tidak mengulang struktur, atau fokus dari pertanyaan lainnya (5 soal berbeda).
- **TIDAK BOLEH ada baris kosong antara pertanyaan dan jawaban. Semua soal dan jawaban harus berurutan tanpa baris kosong.**

Context:
{context}

### Generate 5 pertanyaan berbeda untuk tipe soal essay dengan level sulit atau membutuhkan analisis yang mendalam beserta dengan jawaban yang jelas berdasarkan konteks diatas dan detail dengan dengan format sebagai berikut:
1. Pertanyaan: [Pertanyaan Anda]
Jawaban: Penjelasan singkat jawaban benar
2. Pertanyaan: [Pertanyaan Anda]
Jawaban: Penjelasan singkat jawaban benar
3. Pertanyaan: [Pertanyaan Anda]
Jawaban: Penjelasan singkat jawaban benar
4. Pertanyaan: [Pertanyaan Anda]
Jawaban: Penjelasan singkat jawaban benar
5. Pertanyaan: [Pertanyaan Anda]
Jawaban: Penjelasan singkat jawaban benar
[/INST]
"""

essay_prompt_id = PromptTemplate(
    input_variables=["context"],
    template=essay_id_prompt_template,
)
