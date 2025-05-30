def get_response(user_input):
    user_input = user_input.lower()

    if any(word in user_input for word in ['hello', 'hi', 'hey']):
        return "Hello! How can I assist you today?"

    if 'hours' in user_input or 'open' in user_input or 'close' in user_input:
        return "Our store is open from 9 AM to 9 PM, Monday to Saturday."

    if 'return' in user_input or 'refund' in user_input:
        return "You can return any product within 30 days of purchase with a valid receipt."

    if 'shipping' in user_input or 'delivery' in user_input:
        return "We offer free shipping for orders over $50. Delivery usually takes 3-5 business days."

    if 'contact' in user_input or 'phone' in user_input or 'email' in user_input:
        return "You can contact us at support@example.com or call us at 123-456-7890."

    if 'thank' in user_input:
        return "You're welcome! If you have any more questions, feel free to ask."

    return "I'm sorry, I didn't understand that. Can you please rephrase your question?"

def chat():
    print("Welcome to the Customer Service Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Thank you for visiting! Have a great day.")
            break
        response = get_response(user_input)
        print("Chatbot:", response)

if __name__ == "__main__":
    chat()
