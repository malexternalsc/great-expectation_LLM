import os
import pandas as pd

import logging
import datetime
from util_func import set_up_logging, get_expectation_from_openai

set_up_logging(take_log=True)

def call_openai(batch_size=10):
    """
    Reads user prompts from a text file and generates expectations for each line using OpenAI's model.

    Args:
        batch_size (int): Number of prompts to process in a single batch.
    """
    sample_prompt_file = './data/expectation_and_prompt_sample/sample_quality_check_prompts.txt'

        generated_expectations = []

    try:
        # Read input prompts from file
        with open(sample_prompt_file, 'r') as file:
            prompts = [line.strip() for line in file if line.strip()]

        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            logging.info(f"Processing batch {i // batch_size + 1} with {len(batch)} prompts.")
            print(f"Processing batch {i // batch_size + 1} with {len(batch)} prompts.")

            for prompt in batch:
                expectations = get_expectation_from_openai(input_text=prompt)
                if expectations:
                    generated_expectations.append({
                        'user_prompt': prompt,
                        'generated_expectations': expectations
                    })

        # Save results to a DataFrame with a timestamped filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'data/finetuning_dataset/generated/generated_expectations_{timestamp}.csv'
        df = pd.DataFrame(generated_expectations)
        df.to_csv(output_file, index=False)
        logging.info(f"Generated expectations saved to '{output_file}'.")
    except FileNotFoundError:
        logging.error(f"The file {sample_prompt_file} was not found.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


def encode_data_json():
    pass


if __name__ == '__main__':
    # Set the desired batch size
    call_openai(batch_size=100)