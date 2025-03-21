from openai import OpenAI

client = OpenAI()
import dotenv
import pandas as pd
import os

dotenv.load_dotenv()

# Set up OpenAI API key
openai_key = os.getenv("OPENAI_API_KEY")

# Load dataset (CSV example)
df = pd.read_csv("data/finetuning_dataset/confirmed/generated_expectations_20241231_185728.csv")  # Replace with your actual file

# Few-shot examples in GSM8K format
few_shot_examples = """
### Example 1:
Question: For field 'student_id': Ensure the expect column values to be in set {a,c,v,g,e,ta,b,f}; Ensure the expect column parameterized distribution ks test p value to be greater than 0.05; Ensure this field is a primary key with unique values and is required (not null).
Expected Answer:"expect_column_values_to_be_in_set(column=""student_id"", value_set=[""a"", ""c"", ""v"", ""g"", ""e"", ""ta"", ""b"", ""f""]),expect_column_parameterized_distribution_ks_test_p_value_to_be_greater_than(column=""student_id"", threshold=0.05),  # Example threshold, replace with actual value, expect_column_values_to_be_unique(column=""student_id""), expect_column_values_to_not_be_null(column=""student_id"")"

answer: 
1. The column 'student_id' must only contain values from the set {a, c, v, g, e, ta, b, f}. 
2. The Kolmogorov-Smirnov test should have a p-value greater than 0.05 to ensure distribution consistency. 
3. The field must be unique, ensuring it acts as a primary key.
4. The column must not contain null values.
\n####  "expect_column_values_to_be_in_set(column=""student_id"", value_set=[""a"", ""c"", ""v"", ""g"", ""e"", ""ta"", ""b"", ""f""]),expect_column_parameterized_distribution_ks_test_p_value_to_be_greater_than(column=""student_id"", threshold=0.05),  # Example threshold, replace with actual value, expect_column_values_to_be_unique(column=""student_id""), expect_column_values_to_not_be_null(column=""student_id"")"

### Example 2:
Question: Validate that the `order_date` column adheres to the `YYYY-MM-DD HH:MM:SS` format and does not contain future dates
Expected Answer: "expect_column_values_to_match_strftime_format(column=""order_date"", strftime_format=""%Y-%m-%d %H:%M:%S""), expect_column_values_to_be_dateutil_parseable(column=""order_date"")  expect_column_values_to_be_between(column=""order_date"",min= 1900-01-01, max=""todaY"")"

answer:
1. values in the order_date column must be strings in the date format `YYYY-MM-DD HH:MM:SS`.
2. The maximum date in the order date must not exceed today's date.
\n####  "expect_column_values_to_match_strftime_format(column=""order_date"", strftime_format=""%Y-%m-%d %H:%M:%S""), expect_column_values_to_be_dateutil_parseable(column=""order_date"") expect_column_values_to_be_between(column=""order_date"",min= 1900-01-01, max=""todaY"")
### Example 3:
Question: Validate that the `delivery_date` is always later than the `order_date`, and both are in `YYYY-MM-DD` format.
Expected Answer: "expect_column_values_to_match_strftime_format(column=""delivery_date"", strftime_format=""%Y-%m-%d""), expect_column_values_to_match_strftime_format(column=""order_date"", strftime_format=""%Y-%m-%d""), expect_column_pair_values_A_to_be_greater_than_B(column_A=""delivery_date"", column_B=""order_date"")"

answer:
1. The `delivery_date` must be in the format `YYYY-MM-DD`.
2. The order_date must be in the format `YYYY-MM-DD`.
3. The value in each row of the `delivery_date` must always be greater(after) than the `order_date`.
\n####  "expect_column_values_to_match_strftime_format(column=""delivery_date"", strftime_format=""%Y-%m-%d""), expect_column_values_to_match_strftime_format(column=""order_date"", strftime_format=""%Y-%m-%d""), expect_column_pair_values_A_to_be_greater_than_B(column_A=""delivery_date"", column_B=""order_date"")
"""

# Function to generate GSM8K-style responses
def generate_gsm8k_answer(row):
    question = row["user_prompt"]
    expected_answer = row["generated_expectations"]  # Cleaned expectations

    # Construct the few-shot prompt
    prompt = f"""{few_shot_examples}

    ### New Question:
    Question: {question}
    Expected Answer (Cleaned Expectations): {expected_answer}

    GSM8K-Formatted Answer:
    """

    response = client.chat.completions.create(model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are an AI that formats answers in GSM8K style with reasoning."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7)

    return response.choices[0].message.content.strip()

# Apply function to dataset
df["generated_expectations_gsm8k"] = df.apply(generate_gsm8k_answer, axis=1)

# Save updated dataset
df.to_csv("data/finetuning_dataset/generated/gsm8k_formatted_dataset.csv", index=False)

print("Dataset successfully updated in GSM8K format!")
