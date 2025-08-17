import os
import json
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

# Load environment variables
load_dotenv()
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

class ToDoListAssistant:
    """
    An AI-powered To-Do List Assistant that processes natural language commands using OpenRouter.
    """
    def __init__(self):
        self.tasks = []
        self.chain = self._create_processing_chain()
        print("🤖 AI To-Do List Assistant (powered by OpenRouter) is ready! Type 'quit' to exit.")

    def _create_processing_chain(self):
        """
        Creates the LangChain pipeline to process user input using OpenRouter.
        """
        model = ChatOpenAI(
            model="openai/gpt-oss-20b:free",
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        parser = JsonOutputParser()

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an intelligent to-do list assistant. Your job is to analyze the user's request
and convert it into a structured JSON command. Today's date is {current_date}.

You must respond in a JSON format with two keys: 'action' and 'parameters'.

The possible 'action' values are:
1. 'add_task': When the user wants to add a new task.
   - The 'parameters' should be a dictionary with 'description', 'due_date' (in YYYY-MM-DD format if specified), and 'priority' ('high', 'medium', or 'low').
   - Infer the due_date from relative terms like 'tomorrow', 'next Friday', etc.
   - Infer priority from terms like 'urgent', 'ASAP', 'important'. Default to 'medium'.

2. 'list_tasks': When the user wants to see their current tasks.
   - 'parameters' should be an empty dictionary.

3. 'error': When the user's input is ambiguous, unclear, or not a command.
   - 'parameters' should contain a 'message' explaining why you couldn't process it.

Example 1:
User: Remind me to pay electricity bill tomorrow
Output: {{ "action": "add_task", "parameters": {{ "description": "Pay electricity bill", "due_date": "2025-08-18", "priority": "medium" }} }}

Example 2:
User: what's on my schedule this weekend?
Output: {{ "action": "list_tasks", "parameters": {{}} }}

Example 3:
User: please add the project submission on Friday, it's very important
Output: {{ "action": "add_task", "parameters": {{ "description": "Project submission", "due_date": "2025-08-22", "priority": "high" }} }}

Example 4:
User: hello there
Output: {{ "action": "error", "parameters": {{ "message": "I couldn't understand that as a to-do list command. Please try adding or listing tasks." }} }}
"""),
            ("human", "{user_input}")
        ])

        return prompt | model | parser

    def add_task(self, description, due_date=None, priority='medium'):
        """Adds a new task to the list."""
        task = {
            "id": len(self.tasks) + 1,
            "description": description,
            "due_date": due_date,
            "priority": priority,
            "status": "pending"
        }
        self.tasks.append(task)
        print(f"✅ Task added: '{description}' (Priority: {priority})")

    def list_tasks(self):
        """Displays all current tasks."""
        if not self.tasks:
            print("\n📋 Your to-do list is empty!")
            return

        print("\n--- 📋 Your To-Do List ---")
        for task in sorted(self.tasks, key=lambda x: (x.get('due_date') or 'z', x['id'])):
            date_str = f" (Due: {task['due_date']})" if task['due_date'] else ""
            print(f"  - [ID: {task['id']}] {task['description']}{date_str} [Priority: {task['priority']}]")
        print("--------------------------\n")

    def run(self):
        """The main loop to run the assistant."""
        while True:
            try:
                user_input = input("\n> ")
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break

                current_date = datetime.now().strftime('%Y-%m-%d')
                
                structured_command = self.chain.invoke({
                    "user_input": user_input,
                    "current_date": current_date
                })

                action = structured_command.get("action")
                params = structured_command.get("parameters", {})

                if action == "add_task":
                    self.add_task(**params)
                elif action == "list_tasks":
                    self.list_tasks()
                elif action == "error":
                    print(f"🤔 {params.get('message', 'Sorry, I did not understand that.')}")

            except OutputParserException:
                print("🧠 I had a little trouble understanding that. Could you please rephrase?")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    assistant = ToDoListAssistant()
    assistant.run()
