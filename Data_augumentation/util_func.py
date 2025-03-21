from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
import os
import pandas as pd
import dotenv
import logging
import datetime

# Load environment variables
dotenv.load_dotenv()

def set_up_logging(take_log=True):
    if logging:
       # Set up logging
        log_file = 'expectation_generation.log'
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s : %(levelname)s - %(message)s')
        
def get_accepted_expectations():
    """
    Reads the list of accepted expectations from an Excel file and returns a dictionary.
    """
    try:
        expectation_list = pd.read_excel('./data/expectation_and_prompt_sample/listExpectations.xlsx', 
                                         sheet_name='Expectation_list',
                                         usecols=['Category', 'Expectations'])
        expectation_list['Category'] = expectation_list['Category'].ffill()  # Forward fill NaN values in 'Category'
        expectation_category_dict = (
            expectation_list.groupby('Category')['Expectations']
            .apply(list)
            .to_dict()
        )
        logging.info("Accepted expectations successfully loaded.")
        return expectation_category_dict
    except Exception as e:
        logging.error(f"Error loading accepted expectations: {e}")
        raise


def get_expectation_from_openai(input_text,model=ChatOpenAI(model="gpt-4o-mini", temperature=0.7)):
    """
    Sends a user input prompt to OpenAI's GPT model to generate expectations and references accepted expectations.
    Args:
        input_text (str): The user prompt describing data quality requirements.
    Returns:
        str: The cleaned response with only the generated expectations.
    """
    try:
        # Fetch the accepted expectations dictionary
        accepted_expectations = get_accepted_expectations()

        # Convert the accepted expectations dictionary to a string representation
        expectations_reference = "\n".join(
            [f"{category}: {', '.join(expectations)}" for category, expectations in accepted_expectations.items()]
        )

        examples = [
            {
                'user_prompt': "For column 'processed_timestamp': Ensure the column is required (not null). Ensure the column matches the type 'timestamp', Ensure this column exists.",
                'Expectations': '''expect_column_to_exist(column="processed_timestamp"),
                                    expect_column_values_to_not_be_null(column="processed_timestamp"),
                                    expect_column_values_to_be_of_type(column="processed_timestamp", type_="timestamp")'''
            },
            {
                'user_prompt': "For column 'postal_code': Ensure the column matches the type 'text' and the format 'ZIP'. Ensure this column exists.",
                'Expectations': '''expect_table_columns_to_match_set(column_set=["postal_code"], exact_match=False),
                                    expect_column_values_to_be_of_type(column="postal_code", type_="text"),
                                    expect_column_values_to_match_regex(column="postal_code", regex=r"^\d{5}(-\d{4})?$")'''
            }
        ]

        example_formatter_template = '''user_prompt: {user_prompt} 
                                        expectations: {Expectations}'''

        # Create example prompt
        example_prompt = PromptTemplate(
            input_variables=['user_prompt', 'Expectations'],
            template=example_formatter_template
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            input_variables=['user_prompt'],
            examples=examples,
            example_prompt=ChatPromptTemplate.from_messages([
                ("human", "{user_prompt}"), ("ai", "{Expectations}")
            ])
        )

        # Add the accepted expectations reference to the prompt
        prompt_instruction = f'''Convert the following data quality prompts to great_expectations in the form 
                                  expectation_type(columnName, params...).
                                  Use the accepted expectations reference below to get the available expectations. 
                                  Do not hallucinate or infer expectations from other sources.
                                  Accepted Expectations Reference:
                                  {expectations_reference}'''

        final_prompt = ChatPromptTemplate.from_messages([
            ('system', prompt_instruction),
            few_shot_prompt,
            ('human', '{user_prompt}')
        ])

        # Combine prompt with model
        chain = final_prompt | model

        # Generate raw response
        raw_response = chain.invoke({"user_prompt": input_text})

        logging.info(f"Successfully processed prompt: {input_text}")
        return raw_response.content
    except Exception as e:
        logging.error(f"Error processing prompt '{input_text}': {e}")
        return None        
        
def generate_prompt_text(prompt_, llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7), **kwargs,):
    """
    Generates a response from OpenAI via LangChain using the given prompt template.
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_)
    chain = prompt | llm
    result = chain.invoke(kwargs)
    return result.content
    
       