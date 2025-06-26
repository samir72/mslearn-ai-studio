import os

# Add references
# from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage

def main(): 

    # Clear the console
    os.system('cls' if os.name=='nt' else 'clear')
        
    try: 
    
        # Get configuration settings 
        # load_dotenv()
        #project_connection = os.getenv("PROJECT_ENDPOINT")
        #model_deployment =  os.getenv("MODEL_DEPLOYMENT")
        project_connection="https://aifoundry2.services.ai.azure.com/api/projects/FoundryProject1"
        model_deployment="gpt-4o-standard"
        print(f"project_connection: {project_connection}, model_deployment: {model_deployment}")
   
        # Initialize the project client
        projectClient = AIProjectClient(            
         credential=DefaultAzureCredential(
             exclude_environment_credential=True,
             exclude_managed_identity_credential=True
         ),
         endpoint=project_connection
     )

        ## Get a chat client
        chat = projectClient.inference.get_chat_completions_client()

        # Initialize prompt with system message
        prompt=[
         SystemMessage("You are a helpful AI assistant that answers questions.")
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
            
            # Get a chat completion
            prompt.append(UserMessage(input_text))
            response = chat.complete(
            model=model_deployment,
            messages=prompt,
            temperature=0.1,
            top_p=0.95,
            frequency_penalty=0.1,
            presence_penalty=0,
            stop=None,
        stream=False)
            completion = response.choices[0].message.content
            print(completion)
            prompt.append(AssistantMessage(completion))

    except Exception as ex:
        print(ex)

if __name__ == '__main__': 
    #print(f"project_connection: {project_connection}, model_deployment: {model_deployment}")
    main()