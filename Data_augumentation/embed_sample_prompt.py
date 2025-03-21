from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.prompts import PromptTemplate
import os
import dotenv
import argparse


dotenv.load_dotenv()
# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Database connection configuration
db_connection_args = {
    "user": os.getenv('POSTGRES_USER', 'GELMUSER'),
    "password": os.getenv('POSTGRES_PASSWORD', 'GELMPASSWORD###'),
    "host": "localhost",
    "port": 5432,
    "database": os.getenv('POSTGRES_DB', 'GELMDB')
}


# Create PGVector instance
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="sample_text_file",
    connection=f"postgresql+psycopg://{db_connection_args['user']}:{db_connection_args['password']}@{db_connection_args['host']}/{db_connection_args['database']}"
)


def search_text(query, top_n=3):
    """
    Perform a similarity search on the embedding table using LangChain's vector store.

    Args:
        query (str): The search query to match against the embeddings.
        top_n (int): The number of top documents to retrieve.

    Returns:
        list: A list of document contents from the embedding table.
    """
    try:
        # Perform similarity search in the vector store
        results = vector_store.similarity_search(query, k=top_n)
        
        # Extract document content
        documents = [result.page_content for result in results]
        
        return documents
    
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return []


        
def embed_sample_text(sample_prompt_file):
    with open(sample_prompt_file, 'r') as file:
        prompts = [line.strip() for line in file if line.strip()]
    
    # Add prompts to vector store
    vector_store.add_texts(
        texts=prompts,
        metadatas=[{"source": sample_prompt_file} for _ in prompts]
    )

    
def main(args):
    # Load prompts from file
    sample_prompt_file = args.filepath
    embed_sample_text(sample_prompt_file=sample_prompt_file)


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath',
                        help='Enter the filepath of the txt file to be encoded',
                        required=False,
                        default='./data/expectation_and_prompt_sample/sample_quality_check_prompts.txt')
    
    args = parser.parse_args()
    main(args)