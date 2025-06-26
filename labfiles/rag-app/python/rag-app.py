import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from openai import OpenAI

def main():
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        # Get configuration settings
        load_dotenv()
        open_ai_endpoint = os.getenv("OPEN_AI_ENDPOINT")
        open_ai_key = os.getenv("OPEN_AI_KEY")
        OPEN_API_KEY = os.getenv("OPEN_API_KEY")
        chat_model = os.getenv("CHAT_MODEL")
        embedding_model = os.getenv("EMBEDDING_MODEL")
        search_url = os.getenv("SEARCH_ENDPOINT")
        search_key = os.getenv("SEARCH_KEY")
        index_name = os.getenv("INDEX_NAME")

        # Get an Azure OpenAI chat client
        chat_client = AzureOpenAI(
            api_version = "2024-12-01-preview",
            azure_endpoint = open_ai_endpoint,
            api_key = open_ai_key
        )

        # Get an OpenAI client
        OpenAI_client = OpenAI(
            api_key = OPEN_API_KEY
        )

        #try:
        #    models = chat_client.models.list()
        #    print("API key works. Models:", [model.id for model in models.data])
        #except AzureOpenAI.AuthenticationError:
        #    print("Invalid API key.")
        #except AzureOpenAI.APIConnectionError as e:
        #    print("Connection error:", e)

        # Initialize prompt with system message
        prompt = [
            {"role": "system", "content": "You are a travel assistant that provides information on travel services available from Margie's Travel."}
        ]

        # Loop until the user types 'quit'
        while True:
            # Get input text
            input_text = input("Enter the prompt (or type 'quit' to exit): ")
            if input_text.lower() == "quit":
                break
            if len(input_text) == 0:
                print("Please enter a prompt.")
                continue

            # Add the user input message to the prompt
            prompt.append({"role": "user", "content": input_text})

            # Generate the query embedding
            try:
                response = OpenAI_client.embeddings.create(
                input=prompt,
                model="text-embedding-ada-002"
                )

                # Extract the embedding
                #query_embedding = response['data'][0]['embedding']
                query_embedding = response['data'][0].embedding

                # Inspect the embedding
                print(f"Prompt : {prompt}")
                print("Embedding length:", len(query_embedding))  # Should be 1536
                print("Embedding values (first 10):", query_embedding[:10])  # Print first 10 values for brevity
            except Exception as e:
                print("Error:", e)            
            #
            # Additional parameters to apply RAG pattern using the AI Search index
            rag_params = {
                "data_sources": [
                    {
                        #The following params are used to search the index
                        "type": "azure_search",
                        "parameters": {
                            "endpoint": search_url,
                            "index_name": index_name,
                            "authentication": {
                                "type": "api_key",
                                "key": search_key,
                            },
                            # The following params are used to vectorize the query
                            "embedding_dependency": {
                                "type": "deployment_name",
                                "deployment_name": embedding_model,
                            },
                            "query_type": "vector",
                            #"query_type": "vector_simple_hybrid",
                            #"in_scope": True,
                            #"role_information": "You are a travel assistant that provides information on travel services available from Margie's Travel.",
                            #"strictness": 3,
                            #"top_n_documents": 5

                        }
                    }
                ],
            }
            print(f"Prompt : {prompt}")
            print(f"Rag Parameters : {rag_params}")
            # Submit the prompt with the data source options and display the response
            response = chat_client.chat.completions.create(
                model=chat_model,
                messages=prompt,
                extra_body=rag_params,
                temperature=1,
                top_p=0.95,
                frequency_penalty=0.1,
                presence_penalty=0,
                stop=None,
                stream=False
            )
            completion = response.choices[0].message.content
            print(completion)

            # Add the response to the chat history
            prompt.append({"role": "assistant", "content": completion})

    except Exception as ex:
        print(ex)

if __name__ == '__main__':
    main()