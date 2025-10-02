from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import random
from google import genai
import faiss
import json
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import mysql.connector
import pandas as pd
import re

client = genai.Client(api_key='AIzaSyDaqirxo3mh7WOh1Udbjq6yZxbLRtgyxtg')
ID_MAPPING = """
ID 1: Riyadh (منطقة الرياض)
ID 2: Makkah (منطقة مكة المكرمة)
ID 3: Madinah (منطقة المدينة المنورة)
ID 4: Qassim (منطقة القصيم)
ID 5: Eastern Region (المنطقة الشرقية)
ID 6: Asir (منطقة عسير)
ID 7: Tabuk (منطقة تبوك)
ID 8: Hail (منطقة حائل)
ID 9: Northern Region (منطقة الحدود الشمالية)
ID 10: Jizan (منطقة جازان)
ID 11: Najran (منطقة نجران)
ID 12: Bahah (منطقة الباحة)
ID 13: Jawf (منطقة الجوف)"""
map_data_schema = {
    "type": "object",
    "description": "JSON object containing data to visualize on the map. Must be one of: district-level, region-level, or city-level data.",
    "properties": {
        "districts": {
            "type": "object",
            "description": "District-level data. Keys MUST be the actual district_id values from the database (not generated). Values should be the aggregated metric numbers.",
            "additionalProperties": {
                "type": "number"
            }
        },
        "regions": {
            "type": "object",
            "description": "Region-level data. Keys MUST be the actual region_id values from the database (not generated). Values should be the aggregated metric numbers.",
            "additionalProperties": {
                "type": "number"
            }
        },
        "cities": {
            "type": "object",
            "description": "City-level data. Keys MUST be the actual city_id values from the database (not generated). Values should be the aggregated metric numbers.",
            "additionalProperties": {
                "type": "number"
            }
        },
        "title": {
            "type": "string",
            "description": "The title describing what the data represents (e.g., 'Population', 'Average Age', 'Education Index')."
        }
    },
    "required": ["title"],
    "oneOf": [
        {"required": ["districts", "title"]},
        {"required": ["regions", "title"]},
        {"required": ["cities", "title"]}
    ]
}

# ----------------------------
# 1️⃣ Load FAISS index + docs
# ----------------------------
index = faiss.read_index("schema_index.faiss")

with open("docs.json", "r", encoding="utf-8") as f:
    docs = json.load(f) 

# ----------------------------
# 2️⃣ Load BGE-M3 model (cached from Docker build)
# ----------------------------
model_name = "BAAI/bge-m3"
print("Loading BGE-M3 model from cache...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print("BGE-M3 model loaded successfully!")

def embed_text(texts):
    """Embed text using BGE-M3 (CLS pooling + normalization)."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0]  
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

def strip_sql_markdown(sql_text):
    """Strip markdown formatting from SQL queries."""
    # Remove markdown code blocks (```sql ... ``` or ``` ... ```)
    sql_text = re.sub(r'```(?:sql)?\s*\n?(.*?)\n?```', r'\1', sql_text, flags=re.DOTALL)
    
    # Remove any leading/trailing whitespace
    sql_text = sql_text.strip()
    
    return sql_text

# -------------------------------
# 1️⃣ Database connection details
# -------------------------------
host = "database-1.cko66etrq98i.us-east-1.rds.amazonaws.com"
port = 3306
username = "mysqladmin"
password = "mysqladmin"
database = "yamamah_mod"

# Connect to database
connection = mysql.connector.connect(
    host=host,
    port=port,
    user=username,
    password=password,
    database=database
)

# Create cursor
cursor = connection.cursor()

app = FastAPI(
    title="Yamama Chatbot API",
    version="1.0.0",
    description="The Yamama Chatbot API processes user input and provides an AI-generated response"
)


class ChatRequest(BaseModel):
    sessionId: str
    inputMessage: str


class MapCoordinates(BaseModel):
    latitude: float
    longitude: float


class ChartData(BaseModel):
    chartType: str
    chartTitle: str
    xAxisTitle: str
    yAxisTitle: str
    x: List[str]
    y: List[float]
    dataLabel: str


class ChatResponse(BaseModel):
    sessionId: str
    outputMessage: str
    mapAction: Optional[str] = None
    mapCoordinates: Optional[MapCoordinates] = None
    regionId: Optional[str] = None
    cityId: Optional[str] = None
    districtId: Optional[str] = None
    mapData: Optional[Dict[str, Any]] = None
    chartData: Optional[ChartData] = None


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Get chatbot response and map instructions based on session ID
    """
    session_id = request.sessionId
    
    # Simple CHAT response
    if session_id == "CHAT":
        return ChatResponse(
            sessionId=session_id,
            outputMessage="Hello how can I help you?"
        )
    
    # CHART response with chart data
    elif session_id == "CHART":
        return ChatResponse(
            sessionId=session_id,
            outputMessage="Here is a visual displaying the population of all cities in the kingdom:",
            chartData=ChartData(
                chartType="bar",
                chartTitle="Population Distribution by Region",
                xAxisTitle="Regions",
                yAxisTitle="Population (millions)",
                x=["Riyadh", "Makkah", "Eastern Province", "Asir", "Jazan"],
                y=[8.2, 7.1, 4.9, 2.2, 1.6],
                dataLabel="Cities"
            )
        )
    
    # MAP response with map data
    elif session_id == "MAP":
        # Generate random numbers for 13 regions
        region_data = {
            f"r_{str(i+1).zfill(2)}": round(random.uniform(100000, 5000000), 2)
            for i in range(13)
        }
        
        return ChatResponse(
            sessionId=session_id,
            outputMessage="Here is a breakdown of the population counts for all the regions across the kingdom",
            mapAction="REFRESH",
            mapCoordinates=MapCoordinates(
                latitude=23.8859,
                longitude=45.0792
            ),
            mapData={
                "regions": region_data,
                "title": "Population"
            }
        )
    
    # Default response for any other session ID
    else:
        user_query = request.inputMessage
        q_vec = embed_text([user_query])

        D, I = index.search(q_vec, k=5)  # top-5 matches

        retrieved_docs = [docs[idx]["text"] for idx in I[0]]

        retrieved_tables = "\n".join(retrieved_docs)

        system_instruction = f"""
        You are an expert SQL Query Generator.
        Your task is to convert a user's natural language question into a single, syntactically correct SQL query.
        The query MUST ONLY use the tables provided in the 'AVAILABLE SCHEMA' below.
        DO NOT include any explanation, context, or surrounding text (like markdown ticks ```).
        The output must be ONLY the raw SQL query.
        The output should always be aggregated regardless of the user question. For example, 'patients who live in Riyadh and have diabetes' should be queried aggregated and not raw.
        If the user ask about a specific region, city, or district, return the region_id, city_id, or district_id (depending on the user question) as well as the name of the region, city or district as additional columns and proceed with structuring the query correctly with the joins and conditions. 

        AVAILABLE SCHEMA:
        {retrieved_tables}

        REGION ID MAPPING:
        {ID_MAPPING}

        USER QUESTION TO CONVERT:
        {user_query}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=system_instruction,
            config={
                "temperature": 0.1 
            }
        )

        generated_sql = response.text.strip()
        generated_sql = strip_sql_markdown(generated_sql)
        print(generated_sql)


        # Connect to database
        connection = mysql.connector.connect(
            host=host,
            port=port,
            user=username,
            password=password,
            database=database
        )

        # Create cursor
        cursor = connection.cursor()


        retrieved_table_names = [
            line.split(":")[1].split("(")[0].strip()
            for line in retrieved_tables.splitlines()
            if line.startswith("Table:")
        ]

        query = generated_sql.strip()  

        cursor.execute(query)

        # Fetch result into DataFrame
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        # print(rows, flush=True)
        df_result = pd.DataFrame(rows, columns=columns)
        print(df_result, flush=True)
        cursor.close()
        connection.close()

        system_instruction_chat_decode = f"""
        The user has asked a general question, you need to return a response to the user. You will be provided with the data in a JSON format.
        You will also return a map action to perform on the map based on the request of the user, the 4 options are:
        REFRESH: Refresh the map with the new data.
        ZOOMIN: Zoom in on the map.
        ZOOMOUT: Zoom out on the map.
        MOVE: Move the map to the location specified by the user.

        if there is no action to be performed, return "NONE".
    
        USER QUESTION:
        {user_query}

        DATA FROM DATABASE:
        {df_result.to_json(orient="records")}
        
        return with the following json format:
        {{'outputMessage': 'response to the user', 'mapAction': 'map action'}}.

        have mapAction default to "NONE" if there is no action to be performed.
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=system_instruction_chat_decode,
            config={
                "temperature": 0.1 
            }
        )

        # Use regex to extract JSON from the response
        json_pattern = r'\{[^{}]*"outputMessage"[^{}]*"mapAction"[^{}]*\}'
        json_match = re.search(json_pattern, response.text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            response_text_chat = json.loads(json_str)
        else:
            # Fallback: try to parse the entire response as JSON
            response_text_chat = json.loads(response.text)

        return ChatResponse(
            sessionId=session_id,
            outputMessage=response_text_chat.get('outputMessage'),
            mapAction=response_text_chat.get('mapAction') if response_text_chat.get('mapAction') is not None else "NONE",
            regionId=str(df_result.iloc[0].get('region_id')) if df_result.iloc[0].get('region_id') is not None else None,
            cityId=str(df_result.iloc[0].get('city_id')) if df_result.iloc[0].get('city_id') is not None else None,
            districtId=str(df_result.iloc[0].get('district_id')) if df_result.iloc[0].get('district_id') is not None else None
        )

        # system_instruction_decode = f"""The user has asked a quesiton, you need to decode it depending on the functionality, the 3 options are:
        
        # 1. CHAT: The user is asking a general question that doesn't require any specific data.
        # 2. CHART: The user is asking for a chart or graph of the data.
        # 3. MAP: The user is asking for a map of the data.

        # Please return only of these 3 options, no other text or explanation.
        # Return only the text options "CHAT", "CHART", or "MAP".

        # USER QUESTION:
        # {user_query}
        # """

        # response = client.models.generate_content(
        #     model="gemini-2.5-flash", 
        #     contents=system_instruction_decode,
        #     config={
        #         "temperature": 0.1 
        #     }
        # )

        # response_text_decode = response.text

        # if response_text_decode == "MAP":
        #     system_instruction_map_decode = f"""
        #     The user has asked for map visualizations, and data has been queried to display the new visuals. 
        #     You will be provided with the data in a JSON format and you need to transform it into the mapData format.

        #     CRITICAL Instructions:
        #     1. Analyze the database results to identify if it contains region_id, city_id, or district_id columns
        #     2. Use the ACTUAL ID values from the database - DO NOT generate or make up IDs
        #     3. Look for columns like region_id, city_id, or district_id in the data
        #     4. The ID column values should be the KEYS in the output object
        #     5. The aggregated metric (count, average, etc.) should be the VALUES
        #     6. Create a descriptive title based on what the data represents
        #     7. Return ONLY a valid JSON object matching the schema below

        #     Example transformation:
        #     If data has [{{ "region_id": 1, "patient_count": 500 }}, {{ "region_id": 2, "patient_count": 300 }}]
        #     Then output should be: {{ "regions": {{ "1": 500, "2": 300 }}, "title": "Patient Count by Region" }}

        #     USER QUESTION:
        #     {user_query}

        #     REGION ID MAPPING:
        #     {ID_MAPPING}

        #     DATA FROM DATABASE:
        #     {df_result.to_json(orient="records")}

        #     EXPECTED SCHEMA:
        #     {json.dumps(map_data_schema, indent=2)}

        #     Return ONLY the JSON object, no explanations or markdown. Use the actual IDs from the data!
        #     """
            
        #     response = client.models.generate_content(
        #         model="gemini-2.5-flash", 
        #         contents=system_instruction_map_decode,
        #         config={
        #             "temperature": 0.1,
        #             "response_mime_type": "application/json"
        #         }
        #     )
            
        #     map_data_result = json.loads(response.text)
            
        #     return ChatResponse(
        #         sessionId=session_id,
        #         outputMessage=f"Here's the visualization of the data across the requested geographic areas.",
        #         mapAction="REFRESH",
        #         mapCoordinates=MapCoordinates(
        #             latitude=23.8859,
        #             longitude=45.0792
        #         ),
        #         mapData=map_data_result
        #     )
        # else:
        #     system_instruction_chat_decode = f"""
        #     The user has asked a general question, you need to return a response to the user. You will be provided with the data in a JSON format.

        #     If the data is focused on a specific region, city, or district, return the region_id, city_id, or district_id (depending on the user question) in one of the regionId, cityId, or districtId fields.

        #     USER QUESTION:
        #     {user_query}

        #     DATA FROM DATABASE:
        #     {df_result.to_json(orient="records")}

        #     Return only the following json format:
        #     {{'outputMessage': 'response to the user', 'regionId': 'region_id', 'cityId': 'city_id', 'districtId': 'district_id'}}.
            
        #     have regionId, cityId, or districtId default to null if the question is not about a specific region, city, or district.
        #     """
            
        #     response = client.models.generate_content(
        #         model="gemini-2.5-flash", 
        #         contents=system_instruction_chat_decode,
        #         config={
        #             "temperature": 0.1 
        #         }
        #     )
            
        #     print(response, flush=True)

        #     response_text_chat = response.text
            
        #     # Parse the JSON response from the model
        #     try:
        #         # Extract JSON from markdown code blocks if present
        #         import re
        #         import json
                
        #         # Look for JSON content in code blocks
        #         json_match = re.search(r'```json\s*\n(.*?)\n```', response_text_chat, re.DOTALL)
        #         if json_match:
        #             json_str = json_match.group(1)
        #         else:
        #             # Try to find JSON without code blocks
        #             json_str = response_text_chat.strip()
                
        #         # Parse the JSON
        #         parsed_response = json.loads(json_str)
                
        #         return ChatResponse(
        #             sessionId=session_id,
        #             outputMessage=parsed_response.get('outputMessage', response_text_chat),
        #             regionId=str(parsed_response.get('regionId')) if parsed_response.get('regionId') is not None else None,
        #             cityId=str(parsed_response.get('cityId')) if parsed_response.get('cityId') is not None else None,
        #             districtId=str(parsed_response.get('districtId')) if parsed_response.get('districtId') is not None else None
        #         )
                
        #     except (json.JSONDecodeError, AttributeError) as e:
        #         # Fallback to original behavior if JSON parsing fails
        #         print(f"Failed to parse JSON response: {e}", flush=True)
        #         return ChatResponse(
        #             sessionId=session_id,
        #             outputMessage=response_text_chat
        #         )
        


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Yamama Chatbot API",
        "documentation": "/docs",
        "version": "1.0.0"
    }
