# Great-expectation_LLM

This repository contains the code for finetuning LLaMa for converting Natural Language data quality rules to Great Expectation rules.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Project Structure

```
.gitignore
LICENSE
README.md
requirements.txt

data/
    eval.jsonl
    test.jsonl
    train.jsonl
    expectation_and_prompt_sample/
    finetuning_dataset/
Data_augumentation/
    
    anotate_generated_dataset.py
    create_dataset.py
    docker-compose.yml
    dockerfile
    embed_sample_prompt.py
    generate_prompts.py
    RL_dataset.py
    util_func.py
    postgres-data/
finetuning/
    GE_Llama_3_2_1B+3B_analyse_baseline_experiment.ipynb
Results/
    eval_results_baseline_dataset.csv
```

## Installation

Instructions on how to install and set up the project.

```bash
# Clone the repository
git clone https://github.com/malexternalsc/great-expectation_LLM.git

# Navigate to the project directory
cd great-expectation_LLM

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Data Augumentation
- Create a .env file that contains 
```
OPENAI_API_KEY=sk-proj-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
POSTGRES_USER=GELMUSER
POSTGRES_PASSWORD='GELMPASSWORD###'
POSTGRES_DB=GELMDB
```
- Run the docker-compose file to set up the vector database
``` docker-compose up --build     ```
- Run the DRQ generation code
``` python Data_augumentation\generate_prompts.py```
- Generate corresponding GE prompts 
Run  ``` python Data_augumentation\create_dataset.py ```



#### To embed sample Data Quality rules
Run ``` python Data_augumentation\embed_sample_prompt.py```


## Finetune Model

###  Requirements

    - Access to Llama_3.2 model. Apply for access on [hugging face](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) or on [Meta](https://www.llama.com/llama-downloads/)
    - Device with at least 1 GPU and 16GB VRAM and adequte storage

### Instruction 
- Follow the notebook ```finetuning\GE_Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning_experiments.ipynb ```

Note: To run the baseline experiment that analysis the model before it was finetuned, follow the notebook ```finetuning\GE_Llama_3_2_1B+3B_analyse_baseline_experiment.ipynb```




## Contributing

Guidelines for contributing to the project.

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

