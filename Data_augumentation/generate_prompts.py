import pandas as pd
import os
import re
import random
import time
from itertools import combinations
from embed_sample_prompt import search_text
from datetime import datetime

from util_func import generate_prompt_text



PROMPT = '''Generate **25 high-quality expectation prompts** for an **LLM**, focusing on **data validation checks** across the domains: {selected_domains}.

Each prompt should **follow the lexical style** of the provided examples and address **constraints** specified under {selected_constraints}.

---

### **Requirements:**
1. Each prompt must target a **specific field** in the selected domains.
2. Each prompt should align with one or more constraints defined in {selected_constraints_definition}.
3. The language of the prompts should be **clear, structured, and actionable**.

---

### **Constraint Categories:**
{constraints_section}

---

### **Examples of Prompts:**
{examples_section}

---

### **Output Format:**
Return the **25 prompts** as a **structured list**, ensuring:
- Adherence to the selected constraint categories.
- Alignment with the given examples.
- Clear, consistent, and actionable instructions in each prompt.

Ensure the prompts are tailored to reflect the nuances of the selected domains and constraints.
'''

DOMAINS = [ "E-commerce", "Healthcare", "Banking and Finance", "Social Media Platforms", "Education and Learning Management Systems", "Customer Relationship Management (CRM)",
    "Enterprise Resource Planning (ERP)",  "Travel and Hospitality", "Retail and Inventory Management",
    "Government and Public Services", "Real Estate Management", "Telecommunications", "Gaming and Entertainment", "Supply Chain Management",
    "Human Resources Management Systems (HRMS)", "Cybersecurity and Threat Analysis", "Weather Forecasting Systems", "Transportation and Logistics", "Online Streaming Platforms",
    "IoT (Internet of Things) Applications", "Research and Data Analysis", "Energy and Utilities Management", "Blockchain and Cryptocurrency Platforms",
    "Sports Analytics Platforms", "Fraud Detection Systems", "Content Management Systems (CMS)",
    "Email and Communication Platforms", "Voting and Election Systems", "Insurance Management Systems", "Legal Case Management Systems"
]



# --------------------------
# FUNCTIONS
# --------------------------

def process_excel_with_expectations(file_path):
    """
    Process an Excel file to extract categories, descriptions, and expectations.
    Combine descriptions with expectations into a final dictionary.
    """
    
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        raise FileNotFoundError(f"Error reading Excel file: {e}")
    
    # Create a key-pair dictionary with combined descriptions and expectations
    categories = {}
    for _, row in df.iterrows():
        if pd.notnull(row['Category']) and pd.notnull(row['category_explanation']):
            description = row['category_explanation']
            
            categories[row['Category']] = description
    
    return categories

def extract_prompts(text):
    """
    Extract and clean only the prompts from a structured text.

    Args:
        text (str): Raw text containing headers, numbering, and prompts.

    Returns:
        str: A cleaned string containing only the prompts.
    """
    # Split text into lines and filter lines that look like prompts
    lines = text.split("\n")
    prompts = []
    
    for line in lines:
        # Match lines that start with numbering or bullet point followed by text
        match = re.match(r'^\d+\.\s+"(.+)"$', line.strip())
        if match:
            prompts.append(match.group(1))  # Extract the content inside quotes
    
    return "\n".join(prompts)


def get_random_categories(categories, n=2):
    """
    Select n random categories from the dictionary.
    """
    return dict(random.sample(categories.items(), min(n, len(categories))))

def create_categories_combo(categories_key):
    """
    Create combinations of categories (2 to 5).
    """
    all_combos = []
    for i in range(2, 6):
        all_combos.extend(combinations(categories_key, i))
    return all_combos
    


def create_user_prompt(selected_constraints):
    """
    Dynamically generate a user prompt based on selected constraints and random domains.
    """
    # Select random domains
    num_domains = random.randint(2, 5)
    selected_domains = random.sample(DOMAINS, num_domains)
    
    # Extract constraint definitions
    selected_constraints_definition = [CATEGORIES.get(x, 'No description available') for x in selected_constraints]
    
    # Build constraints section
    constraints_section = ''.join(
        f'- **{constraint}**: {definition}\n'
        for constraint, definition in zip(selected_constraints, selected_constraints_definition)
    )
    
    # Generate examples dynamically (Simulating search_text functionality)
    examples_section = search_text(query=constraints_section, top_n=num_domains)
    
    # Generate the final prompt text
    response = generate_prompt_text(
        prompt_=PROMPT,
        selected_domains=', '.join(selected_domains),
        selected_constraints=', '.join(selected_constraints),
        selected_constraints_definition='\n'.join(selected_constraints_definition),
        examples_section=examples_section,
        constraints_section=constraints_section
    )
    
    # Clean the response to keep only the prompts
    cleaned_response = extract_prompts(response)
    
    if cleaned_response:
        write_to_user_prompt_file(cleaned_response)


def write_to_user_prompt_file(content):
    """
    Write or append content to a daily timestamped file.
    """
    # Generate daily filename
    today = datetime.now().strftime("%Y_%m_%d")
    directory = "./data/expectation_and_prompt_sample"
    os.makedirs(directory, exist_ok=True)
    file_name = os.path.join(directory, f"user_prompt_{today}.txt")
    
    # Write or append to the file
    write_mode = 'a' if os.path.exists(file_name) else 'w'
    with open(file_name, write_mode) as file:
        if write_mode == 'a':
            file.write("\n")  # Add a separator for new entries
        file.write(content)
    
    print(f"Content successfully written to {file_name}")

# --------------------------
# MAIN LOGIC
# --------------------------

            
            
def main():
    file_path = './data/expectation_and_prompt_sample/listExpectations.xlsx'
    try:
        global CATEGORIES
        CATEGORIES = process_excel_with_expectations(file_path)
    except FileNotFoundError as e:
        print(e)
        return
    
    categories_key = list(CATEGORIES.keys())
    categories_combo = create_categories_combo(categories_key)
    
    for combo in categories_combo:
        try:
            create_user_prompt(list(combo))
            time.sleep(5)
        except Exception as e:
            print(f"Error processing combo {combo}: {e}")


if __name__ == '__main__':
    main()
