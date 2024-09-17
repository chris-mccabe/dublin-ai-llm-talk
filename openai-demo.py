import time
import json
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # required, but unused
)

# Dictionary to store the mappings
alias_to_root = {}


def generate_mappings(organization_name, model="llama3.1:70b"):
    prompt = f"""Generate a list of alternative names used for the organization '{organization_name}' with a score on 
    how likely they are to be used. Return a list with keys using the keys `name` and `score`."""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    return json.loads(response.choices[0].message.content)


def check_mappings(mappings, *names):
    """Check if the mappings contain the given names."""
    names = [name.lower() for name in names]
    if isinstance(mappings, dict):
        if mappings['name'].lower() in names:
            return True
    elif isinstance(mappings, list):
        for mapping in mappings:
            if 'name' in mapping and mapping['name'].lower() in names:
                return True
    return False


start_time = time.time()  # Record the start time

mappings = generate_mappings("Apple", model='llama3.1:70b')
print(mappings)



# Check if the returned data is a dictionary with a top-level key containing the list
if isinstance(mappings, dict) and 'alternative_names' in mappings:
    mappings = mappings['alternative_names']

end_time = time.time()  # Record the end time
elapsed_time = end_time - start_time  # Calculate the elapsed time

# Check if the mappings contain "apple" and "apple inc."
contains_apple = check_mappings(mappings, "apple", "apple inc.")
print(
    f"Model: 'llama3.1:70b', Time: {elapsed_time:.2f} seconds, Contains 'apple' or 'apple inc.': {contains_apple}")

print(mappings)


