import os, requests
from serpapi import GoogleSearch
import re
import json
import bcrypt

from youtube_transcript_api import YouTubeTranscriptApi
from summa import summarizer

from fastapi import FastAPI,HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from groq import Groq

from model import *

import tiktoken
# Load tokenizer
tokenizer = tiktoken.get_encoding("r50k_base")

from dotenv import load_dotenv

#Load environment variables
load_dotenv()

#initialize search client API
SERPAPI_API_KEY=os.getenv("SERPAPI_API_KEY")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client=Groq()

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#function to get user login details alternative to OAuthentication
def get_user(password:str,username:str):
    db= SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    if not user or not bcrypt.checkpw(password.encode("utf-8"), user.hashed_password.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {"user_id":user.id,"username":user.username}

#Function below is use to count and chunk text to acceptable TPM according to model use
def count_tokens(text: str) -> int:
    """Estimates the number of tokens in the given text."""
    return len(tokenizer.encode(text, disallowed_special=()))

def chunk_text(text: str, max_tokens: int) -> list:
    """Splits text into chunks that do not exceed max_tokens."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0

    for word in words:
        token_count = count_tokens(word)  # Get token count for each word
        if current_tokens + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_tokens = token_count
        else:
            current_chunk.append(word)
            current_tokens += token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

#this function is use by the model to search for video on youtube
def get_youtube_search_results(query):
    #function to search youtube engine for videos according to user query
    #return five best match
    params = {
    "engine": "youtube",
    "search_query": query,
    "api_key":SERPAPI_API_KEY
    
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    _result = results["video_results"][:5]
    movie_results = []

    for result in _result:
        movie_results.append({
        "published_date": result["published_date"],
        "title": result["title"], 
        "link": result["link"],
        "video_id": re.search(r"v=([\w-]+)",result["link"]).group(1),
        "thumbnail": result["thumbnail"]["static"],
        "views": result.get("views", "N/A"),  # Use .get() to avoid KeyErrors
        "description": result.get("description", "No description available"),
        "length": result.get("length", "Unknown duration")
    })
        
    return movie_results

def get_video_transcription(video_id,language="en"):
    #use function to fetch video from youtube and transcript into english
    ytt_api = YouTubeTranscriptApi()
    try:
        resource=ytt_api.list(video_id)
        full_text=""
        for transcript in resource:
            if transcript.is_translatable:
                snipt= transcript.translate(language).fetch()
                full_text ="\n".join([snippet.text for snippet in snipt.snippets])
           
            else:
                original=transcript.fetch()
                full_text ="\n".join([snippet.text for snippet in original.snippets])
        def summarize_text(text, ratio=0.2):
            """
            Summarizes the transcript using TextRank before sending it to the LLM.
            """
            return summarizer.summarize(text, ratio=ratio)

        
        return chunk_text(summarize_text(full_text),6000)
    except KeyError as e:
        return "The video_id use is not correct"
        

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_video_transcription",
            "description": "Use it to transcribe video clip from youtube into text",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "Video_id to use to search for the video to transcribe on Youtube",
                    },
                    "language":{
                        "type": "string",
                        "description":"language the video should be translate into, default to english 'en' if not specified"
                    }
                },
                "required": ["video_id","language"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_youtube_search_results",
            "description": "Use it to search list of videos from youtube engine and select best match ",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "query to use to searched up video from Youtube",
                    }
                },
                "required": ["query"],
            },
        },
    }
    
]

@app.post("/search-and-summarize")
def get_summarized_video_transcript(request:Search):
    #get system message formated with user query
    query=request.text
    messages = [
        {
            "role": "system",
            "content": """You are a synopsizer that can perform video summarization through video search and video clip transcription.
                The extracted transcription should be summarized to capture key video messages, emotions, the storyline, 
                and individual character mentions. You have the ability to accurately search for and link to the intended video."""
        },
        {
            "role": "system",
            "content": """Automatically find the best and latest match video via YouTube search.
                Extract text transcription from the video clip using the `video_id`, and give a detail summary of the transcription. In the summarization, provide:
                
                - A **brief but detailed summary**
                - **Key themes or points** from the video
                - The **link to the actual YouTube video**.
                
                Search for user input inside the three backticks and fetch the transcription using the function call."""
        },
        {
            "role": "user",
            "content": f"```{query}```"
        }
    ]
    response=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=1,
        tools=tools,
        tool_choice="auto",
        max_tokens=4096
   
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        available_functions = {
            "get_youtube_search_results": get_youtube_search_results,
            "get_video_transcription": get_video_transcription
        }
    
    messages.append(response_message)
    
    video_id = None  # Store the video_id 

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions.get(function_name, None)
        function_args = json.loads(tool_call.function.arguments)

        
        # If the function was `get_youtube_search_results`, extract the first video_id
        if function_name == "get_youtube_search_results":
            # Call the function
            function_response = function_to_call(**function_args)
            video_id = function_response[0]["video_id"] # Extract first video ID
        
            # Append the function response to messages
            messages.append({
                "tool_call_id": tool_call.id, 
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_response),  # Ensure response is formatted correctly
            })
        elif function_name == "get_video_transcription":
            if video_id:
                function_args["video_id"] = video_id
                # Call the function
                function_response = function_to_call(**function_args)
                
                # Append transcription response to messages
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": "get_video_transcription",
                    "content": json.dumps(function_response),
                })
            else:
                continue
            
    # Make final LLM call with updated messages
    second_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    response = second_response

    return f"Final Response: {response.choices[0].message.content}\n\n"

#function that store user search query and summary recieved
def store_search_query_summary(
    user_id: int, 
    query: str|None,
    summary: str|None
    ):
    db = SessionLocal()
    try:
        if not summary.strip():  # Ensure ai advice message is not empty
            print("Warning: Attempted to store an empty Advice message.")
            return  # Exit function without storing
        
        # Create and add new chat entry
        _entry = SearchHistory(user_id=user_id,query=query, summary=summary)
        db.add(_entry)
        db.commit()
        db.refresh(_entry)  # Ensure the object is fully committed
        
        return _entry.id

          
    except Exception as e:
        db.rollback()  # Rollback transaction if there's an error
        print(f"Error storing advice message: {e}")
    
    finally:
        db.close()  # Ensure session is always closed.
        

#Retrieve previous conversation
@app.post("/get_search/history")
def retrieve_search_query_summary(user_id: int):
    db = SessionLocal()
    try:
        # Check if the user has previous advice
        getQuery = db.query(SearchHistory).filter(
            SearchHistory.user_id == user_id          
        )

        # Return empty response if no messages found
        if not getQuery:
            return {"response": ""}

        # Format and return history
        return {
            "response":[{"Query":x.query, "Summary":x.summary} for x in getQuery]
            
        }
    
    finally:
        db.close()  # Ensure the session is closed  
        
#function to authenticate user
@app.post("/login")
def login_user(form_data:OAuth2PasswordRequestForm =Depends()):
    #get user from database
    db = SessionLocal()
    user = db.query(User).filter(User.username == form_data.username).first()
    
    if not user or not bcrypt.checkpw(form_data.password.encode("utf-8"), user.hashed_password.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token_payload = {
        "username": user.username,
        "name": user.name,
        "user_id": user.id
        
    }
    #jwt can be use to manage data validity here
    
    return {"response": token_payload}

#Function to create new user
@app.post("/register")
def register_user(request:CreateUser):
    db = SessionLocal()
    hash_password = bcrypt.hashpw(request.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    db_user = db.query(User).filter(User.username== request.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    new_user = User(username=request.username, hashed_password=hash_password, image=request.image,
                    name=request.name)
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    db.close()
    
    return {"message": "User registered successfully"}

@app.post("/user-search")
def get_user_summarized_video_transcript(request:SearchText):
    #get system message formated with user query
    _user = get_user(password=request.password, username=request.username)
    query=request.text
    messages = [
        {
            "role": "system",
            "content": """You are a synopsizer that can perform video summarization through video search and video clip transcription.
                The extracted transcription should be summarized to capture key video messages, emotions, the storyline, 
                and individual character mentions. You have the ability to accurately search for and link to the intended video."""
        },
        {
            "role": "system",
            "content": """Automatically find the best and latest match video via YouTube search.
                Extract text transcription from the video clip using the `video_id`, and give a detail summary of the transcription. In the summarization, provide:
                
                - A **brief but detailed summary**
                - **Key themes or points** from the video
                - The **link to the actual YouTube video**.
                
                Search for user input inside the three backticks and fetch the transcription using the function call."""
        },
        {
            "role": "user",
            "content": f"```{query}```"
        }
    ]
    response=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=1,
        tools=tools,
        tool_choice="auto",
        max_tokens=4096
   
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        available_functions = {
            "get_youtube_search_results": get_youtube_search_results,
            "get_video_transcription": get_video_transcription
        }
    
    messages.append(response_message)
    
    video_id = None  # Store the video_id 

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions.get(function_name, None)
        function_args = json.loads(tool_call.function.arguments)

        
        # If the function was `get_youtube_search_results`, extract the first video_id
        if function_name == "get_youtube_search_results":
            # Call the function
            function_response = function_to_call(**function_args)
            video_id = function_response[0]["video_id"] # Extract first video ID
        
            # Append the function response to messages
            messages.append({
                "tool_call_id": tool_call.id, 
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_response),  # Ensure response is formatted correctly
            })
        elif function_name == "get_video_transcription":
            if video_id:
                function_args["video_id"] = video_id
                # Call the function
                function_response = function_to_call(**function_args)
                
                # Append transcription response to messages
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": "get_video_transcription",
                    "content": json.dumps(function_response),
                })
            else:
                continue
            
    # Make final LLM call with updated messages
    second_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    response = second_response
    store_search_query_summary(user_id=_user['user_id'], query=query,
                                 summary=response.choices[0].message.content)
    #return final response
    return f"Final Response: {response.choices[0].message.content}\n\n"