import re
import json

def clean_text(text):
    """
    Remove Unicode bullet points and other special characters from text.
    """
    # Remove Unicode bullet points (\u00e2\u0080\u00a2)
    text = text.replace('\u00e2\u0080\u00a2', '')
    
    # Remove any other potential special characters
    # This regex will remove any non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Clean up any extra whitespace that might be left
    text = ' '.join(text.split())
    
    return text

def process_json_file(input_file, output_file):
    """
    Process a JSON file and write cleaned text to a new JSON file.
    """
    try:
        # Read and parse JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Function to recursively clean text in JSON
        def clean_json(obj):
            if isinstance(obj, dict):
                return {k: clean_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_json(item) for item in obj]
            elif isinstance(obj, str):
                return clean_text(obj)
            else:
                return obj
        
        # Clean the JSON data
        cleaned_data = clean_json(data)
        
        # Write the cleaned JSON data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
            
        print(f"Successfully cleaned JSON and saved to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Example usage
    input_file = "Text_Augmented_New.json"  # Replace with your input file
    output_file = "cleaned_text.json"       # Replace with your desired output file
    process_json_file(input_file, output_file) 