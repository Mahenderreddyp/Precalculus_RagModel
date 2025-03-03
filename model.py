import chromadb
import logging
import sys
import os
import torch
import functools
from typing import Dict, Any
import re

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate)
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

# Configure device - GPU if available
if torch.cuda.is_available():
    device = "cuda"
    # Set lower precision for faster computation
    torch.set_float32_matmul_precision('medium')
else:
    device = "cpu"

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Using device: {device}")

# Global variables
global query_engine
query_engine = None

# Response cache
response_cache: Dict[str, Any] = {}
CACHE_SIZE = 100  # Adjust based on memory constraints

# Critical formatting requirements to add to all prompts
FORMATTING_REQUIREMENTS = """
CRITICAL FORMATTING REQUIREMENTS:
1. NEVER place a # symbol alone on a line. Always put the heading text on the same line as the # symbol.
2. NEVER put numbers (like '1.') alone on a line. Always include the content on the same line.
3. Use '# Heading' format with a space between # and the heading text.
4. For headers: Write '# Title' not '#' followed by 'Title' on the next line.
5. For section titles like 'Explanations', write '# Explanations' not '#' followed by 'Explanations' on the next line.
6. For numbered steps: Write '1. First step' not '1.' followed by 'First step' on the next line.
7. For all headings, keep the heading symbol and the heading text together on the same line.
8. Avoid excessive line breaks - use at most one blank line between sections.
"""

def extract_requested_question_count(question_text):
    """Extract the number of questions requested by the student"""
    import re
    
    # Check for spelled-out numbers
    number_words = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    for word, number in number_words.items():
        pattern = r'\b' + word + r'\s*(question|problems|examples)\b'
        if re.search(pattern, question_text, re.IGNORECASE):
            return number
    
    # Check for digits
    digit_patterns = [
        r'\b(\d+)\s*(question|problems|examples)\b',  # "2 questions"
        r'([0-9]+)\s*q',  # "2q" or "2 q"
        r'give\s+me\s+([0-9]+)',  # "give me 2"
        r'provide\s+([0-9]+)'  # "provide 2"
    ]
    
    for pattern in digit_patterns:
        match = re.search(pattern, question_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    # Default to 1 if no number is found
    return 1

def fix_question_numbering(text):
    """Ensures questions are numbered sequentially starting from 1"""
    import re
    
    # Find all question headers
    question_pattern = r'(Question|QUESTION)\s+(\d+):'
    questions = re.findall(question_pattern, text)
    
    if not questions:
        return text
    
    # Check if numbering needs correction
    needs_correction = False
    if int(questions[0][1]) != 1:  # First question doesn't start with 1
        needs_correction = True
    
    # Check for sequential numbering
    for i in range(1, len(questions)):
        if int(questions[i][1]) != int(questions[i-1][1]) + 1:
            needs_correction = True
            break
    
    # If correction needed, fix the numbering
    if needs_correction:
        for i, (prefix, num) in enumerate(questions):
            old = f"{prefix} {num}:"
            new = f"{prefix} {i+1}:"
            text = text.replace(old, new)
    
    return text

def init_llm():
    """Initialize language model and embedding model with GPU acceleration if available"""
    # Use a smaller quantized model for faster inference
    llm = Ollama(model="llama3.2:3b", request_timeout=300.0)
    
    # Configure embedding model with appropriate device
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        device=device
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    logging.info(f"LLM and embedding models initialized on {device}")


def init_index(embed_model):
    """Initialize and return the vector index with persistence"""
    # Create persistent directory for ChromaDB
    os.makedirs("./chroma_db", exist_ok=True)
    
    # Check if we can load from cache
    chroma_cache_path = "./chroma_db"
    if os.path.exists(chroma_cache_path) and os.listdir(chroma_cache_path):
        logging.info("Loading index from persistent storage")
        chroma_client = chromadb.PersistentClient(path=chroma_cache_path)
        try:
            chroma_collection = chroma_client.get_collection("precalculus")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
            logging.info("Successfully loaded index from persistent storage")
            return index
        except Exception as e:
            logging.warning(f"Failed to load from persistent storage: {e}")
    
    # If loading from cache fails, create a new index
    logging.info("Creating new index")
    reader = SimpleDirectoryReader(input_dir="./precalculus_docs", recursive=True)
    documents = reader.load_data()

    logging.info(f"Creating index with {len(documents)} documents")

    chroma_client = chromadb.PersistentClient(path=chroma_cache_path)
    # Delete collection if it exists to avoid conflicts
    try:
        chroma_client.delete_collection("precalculus")
    except:
        pass
    
    chroma_collection = chroma_client.create_collection("precalculus")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create batch size for processing large document sets
    batch_size = 10
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        if i == 0:
            index = VectorStoreIndex.from_documents(batch, storage_context=storage_context, embed_model=embed_model)
        else:
            index.insert_nodes(VectorStoreIndex.from_documents(batch).as_nodes())
        
        logging.info(f"Processed {min(i + batch_size, len(documents))}/{len(documents)} documents")

    return index


def init_query_engine(index):
    global query_engine

    # Updated prompt template with better formatting instructions
    template = (
    "Imagine you are an AI teaching assistant for precalculus, with expertise in all relevant topics "
    "including Cartesian Plane and Functions, Lines, Polynomials and Rational Functions, "
    "Inequalities, Exponential and Logarithmic Functions, and Trigonometry.\n\n"
    "Here is some context related to the query:\n"
    "-----------------------------------------\n"
    "{context_str}\n"
    "-----------------------------------------\n"
    "Considering the above information, please respond to the following question:\n\n"
    "Question: {query_str}\n\n"
    "Provide helpful guidance without solving questions completely. Include steps that would help "
    "the student understand the concept. Reference the appropriate topics from precalculus that relate "
    "to this question. If the question is not about precalculus, politely redirect to precalculus topics.\n\n"
    "CRITICAL FORMATTING REQUIREMENTS:\n"
    "1. NEVER place a # symbol alone on a line. Always put the heading text on the same line as the # symbol.\n"
    "2. NEVER put numbers (like '1.') alone on a line. Always include the content on the same line.\n"
    "3. Use '# Heading' format with a space between # and the heading text.\n"
    "4. For headers: Write '# Title' not '#' followed by 'Title' on the next line.\n"
    "5. For section titles like 'Explanations', write '# Explanations' not '#' followed by 'Explanations' on the next line.\n"
    "6. For numbered steps: Write '1. First step' not '1.' followed by 'First step' on the next line.\n"
    "7. For all headings, keep the heading symbol and the heading text together on the same line.\n"
    "8. For mathematical points, use the format (x_1, y_1) for proper rendering, not (x₁, y₁).\n"
    "9. Minimize excessive spacing - use at most one blank line between paragraphs.\n"
    "10. For absolute values use |x|, not *x*.\n"
    "11. For inequalities, use ≤ and ≥ symbols instead of <= and >=\n"
    "12. For square roots, use √ instead of sqrt()\n"
    "13. For exponents, use superscript notation (x²) instead of x^2\n"
    )
    qa_template = PromptTemplate(template)

    # Build query engine with custom template
    query_engine = index.as_query_engine(
        text_qa_template=qa_template, 
        similarity_top_k=3,
        response_mode="compact"  # More concise responses
    )

    return query_engine


# Simple LRU cache implementation
@functools.lru_cache(maxsize=CACHE_SIZE)
def cached_query(prompt: str) -> str:
    """Cache query results to avoid recomputing common questions"""
    response = query_engine.query(prompt)
    return response.response

def post_process_math_formatting(text):
    """
    Post-process the response to fix formatting issues and improve text layout
    """
    import re
    
    # First, handle the standalone '#' issue by combining it with the next line
    lines = text.split('\n')
    result_lines = []
    skip_next = False
    
    for i in range(len(lines)):
        if skip_next:
            skip_next = False
            continue
            
        current_line = lines[i].strip()
        
        # If the current line is just a '#' and there's a next line
        if re.match(r'^#+$', current_line) and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            result_lines.append(f"{current_line} {next_line}")
            skip_next = True
        # If current line is just a section title word
        elif re.match(r'^(Explanations|Alternative Approach|Key Concepts( Being Applied)?|Solution)$', current_line) and i + 1 < len(lines):
            result_lines.append(f"# {current_line}")
            skip_next = True
        # If current line is just a number with a period (like "1.")
        elif re.match(r'^\d+\.$', current_line) and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            result_lines.append(f"{current_line} {next_line}")
            skip_next = True
        # If current line is "Step X:" without content
        elif re.match(r'^Step \d+:$', current_line) and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            result_lines.append(f"{current_line} {next_line}")
            skip_next = True
        else:
            result_lines.append(lines[i])
    
    text = '\n'.join(result_lines)
    
    # Fix standalone # that should precede content
    text = re.sub(r'(#+)\s*\n+\s*([A-Za-z][^\n]*)', r'\1 \2', text)
    
    # Fix headers and section titles that are split across lines
    text = re.sub(r'(Explanations)\s*\n+', r'# Explanations: ', text)
    text = re.sub(r'(Alternative Approach)\s*\n+', r'# Alternative Approach: ', text)
    text = re.sub(r'(Key Concepts Being Applied)\s*\n+', r'# Key Concepts Being Applied: ', text)
    text = re.sub(r'(Example \d+:)\s*\n+\s*([A-Za-z][^\n]*)', r'\1 \2', text)
    
    # Fix numbered list spacing - ensure number and content are on same line
    text = re.sub(r'(\d+)\.\s*\n+', r'\1. ', text)
    
    # Fix math notation for points - ensure proper MathJax formatting
    # Convert (x₁, y₁) to LaTeX format
    text = re.sub(r'\(x₁, y₁\)', r'(x_1, y_1)', text)
    text = re.sub(r'\(x₂, y₂\)', r'(x_2, y_2)', text)
    
    # Convert unicode subscripts to regular numbers for MathJax processing
    text = text.replace('x₁', 'x_1')
    text = text.replace('y₁', 'y_1')
    text = text.replace('x₂', 'x_2')
    text = text.replace('y₂', 'y_2')
    
    # Fix steps that are split across lines
    text = re.sub(r'(Step \d+:)\s*\n+', r'\1 ', text)
    
    # Fix question numbering
    text = fix_question_numbering(text)
    
    # Reduce excessive spacing - at most one blank line between paragraphs
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace single and double asterisks with HTML bold tags
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*([^*]+)\*', r'<strong>\1</strong>', text)
    
    # Ensure proper inequality symbols
    text = text.replace('<=', '≤')
    text = text.replace('>=', '≥')
    
    # Fix square root notation
    text = re.sub(r'sqrt\(([^)]+)\)', r'√(\1)', text)
    
    # Fix exponent notation
    text = re.sub(r'([a-zA-Z])(?:\^|\*\*)2', r'\1²', text)
    text = re.sub(r'([a-zA-Z])(?:\^|\*\*)3', r'\1³', text)
    
    return text

def clean_output_structure(text):
    """
    Special function to fix structural issues in the output format
    """
    import re
    
    # Replace numbered lists that are split across lines
    lines = text.split('\n')
    cleaned_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if this line is just a number followed by a period
        if re.match(r'^\d+\.$', line) and i+1 < len(lines):
            # Combine with next line
            cleaned_lines.append(f"{line} {lines[i+1].strip()}")
            i += 2
        # Check if line is just a hash symbol
        elif re.match(r'^#+$', line) and i+1 < len(lines):
            # Combine with next line
            cleaned_lines.append(f"{line} {lines[i+1].strip()}")
            i += 2
        # Check for section titles on their own line
        elif re.match(r'^(Explanations|Alternative Approach|Key Concepts( Being Applied)?|Solution)$', line) and i+1 < len(lines):
            cleaned_lines.append(f"# {line}")
            i += 1
        # Check for lines that just have "Step X:"
        elif re.match(r'^Step \d+:$', line) and i+1 < len(lines):
            next_line = lines[i+1].strip()
            cleaned_lines.append(f"{line} {next_line}")
            i += 2
        else:
            cleaned_lines.append(line)
            i += 1
    
    text = '\n'.join(cleaned_lines)
    
    # Fix headers and ensure they're on the same line as their content
    text = re.sub(r'(#+)\s*\n+\s*([A-Za-z])', r'\1 \2', text)
    
    # Ensure question numbers start at 1 and are sequential
    text = fix_question_numbering(text)
    
    # Reduce excessive spacing
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

def validate_question_count_and_numbering(response, requested_count):
    """Check if the response contains the correct number of properly numbered questions"""
    import re
    
    # Count question headers - check both formats (Question X: and # Question X:)
    question_patterns = [
        r'# Question \d+:', 
        r'Question \d+:',
        r'QUESTION \d+:'
    ]
    
    total_questions = 0
    for pattern in question_patterns:
        questions = re.findall(pattern, response)
        total_questions += len(questions)
    
    # Check if we have the right number of questions
    if total_questions < requested_count:
        # If not enough questions, add a request for more
        additional_prompt = f"""
        Your response only included {total_questions} question(s), but the student requested {requested_count}.
        Please provide {requested_count - total_questions} more question(s) with complete solutions.
        
        {FORMATTING_REQUIREMENTS}
        
        IMPORTANT: 
        - ALWAYS start question numbering from where you left off
        - If you provided Question 1, the next should be Question 2, and so on
        - Each question must be clearly labeled
        
        For each additional question:
        1. Start with "# Question X:" where X is the next question number
        2. Provide a relevant practice problem
        3. Follow with hints or a solution as requested
        
        SEPARATE EACH QUESTION with a line containing only "---"
        """
        return False, additional_prompt
    
    # Response already fixed by fix_question_numbering()
    return True, None

def process_precalculus_query(query_text, user_id):
    """
    Enhanced process function that provides better formatted mathematical content
    with caching for performance
    """
    # Parse the query format: "Mode: X | Topic: Y | Question: Z"
    parts = query_text.split('|')
    mode = parts[0].replace('Mode:', '').strip()
    topic = parts[1].replace('Topic:', '').strip() if len(parts) > 1 else ""
    question = parts[2].replace('Question:', '').strip() if len(parts) > 2 else ""
    
    # Extract the number of questions requested
    requested_count = extract_requested_question_count(question)
    
    # Statement to include in responses
    statement = ("\n\nA good reference for precalculus is Precalculus by R. Larson, 11th Edition, Cengage (2022). "
                "Please note that some problems may be inspired by problems in this textbook. "
                "If you have any questions or notice any errors, please let the instructor know about it. "
                "Thank you in advance for your feedback!")
    
    # Detect formatting issues
    has_formatting_issue = any(phrase in question.lower() for phrase in 
                          ["format", "asterisk", "paragraph", "line", "spacing", "layout", "display", 
                           "not rendering", "proper", "single paragraph", "differentiate", "visible", "new line"])
    
    # Detect if student is asking for step-by-step solution
    wants_solution = any(phrase in question.lower() for phrase in 
                         ["step by step", "solve this", "solution", "answer", "how to solve", 
                          "explain how", "show me how", "walk through"])
    
    # Check if the question is about error correction or layout
    is_error_question = any(phrase in question.lower() for phrase in
                            ["wrong", "incorrect", "error", "mistake", "fix", "correct", "layout"])
    
    # Process based on mode and query content
    if has_formatting_issue:
        prompt = f"""
        The student is having issues with formatting or layout in the chat interface.
        Their specific question is: "{question}"
        
        {FORMATTING_REQUIREMENTS}
        
        Please explain:
        1. How to properly format text with good layout and spacing
        2. How to use proper line breaks and paragraph spacing for mathematical content
        3. How to format problems so they are clearly separated from each other
        
        When writing your response, please use:
        - Clear section headers for each question (like "# Question 1", "# Question 2", etc.)
        - A blank line between the question and solution
        - Numbered steps on separate lines with proper indentation
        - Section headers for Explanations, Alternative Approaches, and Key Concepts
        - Triple dashes (---) between different problems to clearly separate them
        - Blank lines between different sections of content
        
        Demonstrate this with an example showing proper formatting for a practice problem.
        End with the statement: "{statement}"
        """
    elif is_error_question:
        prompt = f"""
        The student is pointing out a potential error or layout issue.
        Their specific request is: "{question}"
        
        {FORMATTING_REQUIREMENTS}
        
        Please provide:
        1. A careful analysis of any issues with the content
        2. A corrected version with proper formatting
        3. An explanation of the correct approach
        
        When formatting your response:
        - Use clear section headers for different parts (Question, Solution, Explanation)
        - Put each new step on a separate line
        - Separate different questions with triple dashes (---)
        - Use blank lines between different sections
        - Ensure all mathematical expressions are properly formatted
        
        Use proper mathematical notation:
        - Use |x| for absolute value (not *x*)
        - Use subscripts for variables (x₁, y₁) instead of x1, y1
        - Use proper notation for square roots (√)
        - Use proper notation for exponents (x²)
        - Format fractions clearly
        
        End with the statement: "{statement}"
        """
    elif mode == "practice":
        if wants_solution:
            prompt = f"""
            The student is asking for {requested_count} question(s) about {topic}.
            Their specific request is: "{question}"
            
            {FORMATTING_REQUIREMENTS}
            
            YOU MUST PROVIDE EXACTLY {requested_count} DIFFERENT QUESTIONS with step-by-step solutions.
            
            IMPORTANT QUESTION GENERATION INSTRUCTIONS:
            - Use the questions provided in the document as reference and generate the similar questions and also make sure you generate the hints on how to solve that question
            - make sure you provide the answers as well if the user has asked for the answers.
            - You must generate only the questions based on that particular topic
            
            For each question:
            1. Start with "# Question X:" where X is the question number (1, 2, etc.)
            2. Provide a relevant practice problem in the style of those in 131Win25PrecalculusProblems.pdf
            3. Follow with "# Step-by-Step Solution for Question X:"
            4. Give a complete, step-by-step solution with clearly numbered steps
            5. After all steps are listed, include "# Explanations for Question X:"
            6. If relevant, add "# Alternative Approach for Question X:"
            7. Finally, add "# Key Concepts for Question X:"
            
            IMPORTANT FORMATTING INSTRUCTIONS:
            - Ensure you provide EXACTLY {requested_count} questions and solutions
            - The first question MUST be labeled "# Question 1:" (not any other number)
            - Leave a blank line between the question and solution
            - Number each step and put it on its own line (1., 2., 3., etc.)
            - After the solution steps, include sections labeled "# Explanations", "# Alternative Approach", and "# Key Concepts Being Applied"
            - SEPARATE EACH QUESTION with a line containing only "---"
            - Use blank lines between different sections to improve readability
            - Ensure each solution follows a logical progression where each step builds on previous steps
            
            IMPORTANT FOR CALCULATIONS:
            - Double-check all calculations for accuracy
            - When working with coordinates, verify that all operations are done correctly
            - For midpoint calculations, use ((x₁+x₂)/2, (y₁+y₂)/2) applied precisely
            - For inequalities, be careful with sign changes when multiplying/dividing by negative numbers
            
            Use proper mathematical notation:
            - Use |x| for absolute value (not *x*)
            - Use subscripts for variables (x₁, y₁) instead of x1, y1
            - Use proper notation for square roots (√)
            - Use proper notation for exponents (x²)
            - Format fractions clearly
            
            Be thorough in your explanation and make it easy for a student to follow.
            DOUBLE-CHECK ALL CALCULATIONS FOR ACCURACY.
            End with the statement: "{statement}"
            """
        else:
            prompt = f"""
            The student is asking for {requested_count} practice problem(s) about {topic}.
            
            {FORMATTING_REQUIREMENTS}
            
            YOU MUST PROVIDE EXACTLY {requested_count} DIFFERENT PRACTICE PROBLEMS.
            
            IMPORTANT QUESTION GENERATION INSTRUCTIONS:
            - Use the questions provided in the document as reference and generate the similar questions and also make sure you generate the hints on how to solve that question
            - make sure you provide the answers as well if the user has asked for the answers.
            
            For each problem:
            1. Start with "# Question X:" where X is the question number (1, 2, etc.)
            2. Generate a practice problem similar to those in the 131Win25PrecalculusProblems.pdf document
            3. Provide guidance without completely solving the problem
            4. Give hints about the approach to use
            
            If the user has provided a specific question: "{question}", respond to that instead.
            
            IMPORTANT FORMATTING INSTRUCTIONS:
            - Ensure you provide EXACTLY {requested_count} questions with hints
            - The first question MUST be labeled "# Question 1:" (not any other number)
            - Leave a blank line after each question
            - Number any hint steps and put each on its own line
            - SEPARATE EACH QUESTION with a line containing only "---"
            - Use blank lines between different sections to improve readability
            
            Use proper mathematical notation:
            - Use |x| for absolute value (not *x*)
            - Use subscripts for variables (x₁, y₁) instead of x1, y1
            - Use proper notation for square roots (√)
            - Use proper notation for exponents (x²)
            - Format fractions clearly
                   
            
            Ensure your hints follow a logical progression where each hint builds on previous hints.
            End with the statement: "{statement}"
            """
    elif mode == "review":
        prompt = f"""
        I need a comprehensive review of the precalculus topic: {topic}.
        
        {FORMATTING_REQUIREMENTS}
        
        Include:
        1. Basic concepts and definitions
        2. Key formulas and theorems with proper mathematical notation
        3. Step-by-step examples with detailed explanations
        4. Common pitfalls and how to avoid them
        5. Connections to calculus
        
        IMPORTANT FORMATTING INSTRUCTIONS:
        - Use clear section headers for different parts of your review (like "# Basic Concepts", "# Key Formulas", etc.)
        - For examples, use clear labels like "# Example 1:", "# Example 2:", etc.
        - Put each step in examples on a new line
        - Number steps sequentially (1., 2., 3., etc.) and keep content on the same line
        - Use blank lines between different sections to improve readability
        - Clearly label and separate examples from explanations
        
        Use proper mathematical notation:
        - Use |x| for absolute value (not *x*)
        - Use subscripts for variables (x₁, y₁) instead of x1, y1
        - Use proper notation for square roots (√)
        - Use proper notation for exponents (x²)
        - Format fractions clearly
        
        Explain each step thoroughly and make sure formulas are clear and easy to understand.
        Ensure all mathematical calculations are accurate and properly formatted.
        End with the statement: "{statement}"
        """
    elif mode == "history":
        prompt = f"""
        Provide historical context about the precalculus topic: {topic}.
        
        {FORMATTING_REQUIREMENTS}
        
        Include key mathematicians who contributed to this area.
        Mention how these concepts evolved over time.
        
        IMPORTANT FORMATTING INSTRUCTIONS:
        - Use clear section headers for different parts of your response (like "# Early Development", "# Key Contributors", etc.)
        - Start each new concept or time period on a new line
        - Use blank lines between different sections to improve readability
        - For chronological information, use clear labels like "# Period 1:", "# Period 2:", etc.
        
        If mathematical formulas are mentioned, use proper notation:
        - Use |x| for absolute value (not *x*)
        - Use subscripts for variables (x₁, y₁) instead of x1, y1
        - Use proper notation for square roots (√)
        - Use proper notation for exponents (x²)
        - Format fractions clearly
        
        If no specific topic is mentioned, give a brief overview of the historical development of
        a random precalculus concept.
        End with the statement: "{statement}"
        """
    else:
        # Default handling
        prompt = question + f"""
        
        {FORMATTING_REQUIREMENTS}
        
        Please use proper mathematical notation and clear formatting in your response. 
        If multiple questions are requested, provide EXACTLY {requested_count} questions with clear solutions. 
        The first question MUST be labeled "# Question 1:" (not any other number).
        Use new lines for each step or concept, and separate sections with blank lines.
        """
    
    # Use the query engine to get a response
    try:
        # Use the cached query for better performance
        response = cached_query(prompt)
        
        # Fix question numbering first
        response = fix_question_numbering(response)
        
        # Apply post-processing for better formatting
        processed_response = post_process_math_formatting(response)
        
        # Apply additional structural cleaning
        final_response = clean_output_structure(processed_response)
        
        # Validate if the requested number of questions is present
        if requested_count > 1:
            is_valid, additional_prompt = validate_question_count_and_numbering(final_response, requested_count)
            
            if not is_valid:
                # Get additional questions
                additional_response = cached_query(additional_prompt)
                additional_response = fix_question_numbering(additional_response)
                processed_additional = post_process_math_formatting(additional_response)
                cleaned_additional = clean_output_structure(processed_additional)
                
                # Combine responses
                final_response += "\n\n---\n\n" + cleaned_additional
                
                # Do one final check on numbering for the combined response
                final_response = fix_question_numbering(final_response)
        
        return final_response
        
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return f"I apologize, but I encountered an error while processing your question. Please try again or simplify your question. Technical details: {str(e)}"

# Replace the original chat function with this enhanced one
def chat(input_question, user):
    return process_precalculus_query(input_question, user)

def chat_cmd():
    global query_engine

    while True:
        input_question = input("Enter your question (or 'exit' to quit): ")
        if input_question.lower() == 'exit':
            break

        response = query_engine.query(input_question)
        logging.info("got response from llm - %s", response)


if __name__ == '__main__':
    init_llm()
    index = init_index(Settings.embed_model)
    init_query_engine(index)
    chat_cmd()