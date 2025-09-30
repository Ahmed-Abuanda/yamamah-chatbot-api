from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import random

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
        return ChatResponse(
            sessionId=session_id,
            outputMessage="I received your message. Please use session ID: CHAT, CHART, or MAP for demo responses."
        )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Yamama Chatbot API",
        "documentation": "/docs",
        "version": "1.0.0"
    }
