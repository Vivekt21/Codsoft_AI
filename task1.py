import re

def simple_chatbot(user_input):
    # Convert user input to lowercase for case-insensitive matching
    user_input = user_input.lower()

    # Define predefined rules and responses
    rules_responses = {
        'hello': 'Hi there! How can I help you?',
        'how are you': 'I am just a computer program, but thanks for asking!',
        'bye': 'Goodbye! Have a great day!',
        'default': "'I'm sorry, I don't understand. Can you please rephrase or ask something else?',"
    }

    # Pattern matching using regular expressions
    if re.search(r'\bhello\b', user_input):
        return rules_responses['hello']
    elif re.search(r'\bhow are you\b', user_input):
        return rules_responses['how are you']
    elif re.search(r'\bbye\b', user_input):
        return rules_responses['bye']
    else:
        return rules_responses['default']

# Example usage
while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = simple_chatbot(user_input)
    print("Chatbot:", response)
