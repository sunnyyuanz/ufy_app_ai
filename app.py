from flask import Flask, request, jsonify
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json

# Initialize Flask app
app = Flask(__name__)

load_dotenv()

# Enable CORS
CORS(app)

# Configure LangChain with OpenAI API key
chat_model = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,  # Increased for more creative responses
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

def create_initial_itinerary_structure(origin, days, destinations, budget, stayPref, currency="USD", groupSize=2, 
                    comfortLevel="moderate", theme="general", additionalInfo=""):
    return {
        "title": f"{days}-Day Trip from {origin} to {destinations}" if theme == "general" else f"{days}-Day {theme} Trip from {origin} to {destinations}",
        "details": {
            "budget": budget,
            "currency": currency,
            "groupSize": groupSize,
            "comfortLevel": comfortLevel,
            "StayPref": stayPref,
            "theme": theme,
            "additionalInfo": additionalInfo
        }
    }

def populate_daily_activities(trip_details):
    """Generate detailed daily activities using the LLM"""
    days = trip_details["details"].get("days", 5)
    prompt = f"""
    Create a detailed day-by-day itinerary for a {days}-day trip with the costs calculation:
    - From: {trip_details['details'].get('origin')}
    - To: {trip_details['details'].get('destinations')}
    - Budget: {trip_details['details'].get('budget')} {trip_details['details'].get('currency')}
    - Stay Preference: {trip_details['details'].get('stayPref')}
    - Group Size: {trip_details['details'].get('groupSize')}
    - Comfort Level: {trip_details['details'].get('comfortLevel')}
    - Theme: {trip_details['details'].get('theme')}
    - Additional Info: {trip_details['details'].get('additionalInfo')}

    For each day, provide:
    1. Morning activities, MUST provide transportation details for each activity(for example, taxi to Airbnb address), includes estimated transportation time; if theme is not general and the activity is related to the theme, MUST provide the details of how the activities related to the theme.
    2. Afternoon activities, MUST provide transportation details for each activity(for example, taxi to Airbnb address), includes estimated transportation time; if theme is not general and the activity is related to the theme, MUST provide the details of how the activities related to the theme.
    3. Evening activities, MUST provide transportation details for each activity(for example, subway to dinning address), includes estimated transportation time; if theme is not general and the activity is related to the theme, MUST provide the details of how the activities related to the theme.
    4. Recommended restaurants/meals
    5. Estimated costs, must includes stay(hotel/airbnb) costs, transportation costs, meal costs and miscellaneous costs. 

    After day-by-day itinerary, also provide:
    1. Estimated Stay Total Costs
    2. Estimated Transportation Total Costs
    3. Estimated Meal Total Costs
    4. Estimated miscellaneous Costs
    5. Estimated Trip Total Costs

    Return the response as a JSON array where each element has 'day' and 'details' keys, the response also includes a 'costs' key for trip total costs details.
    """

    messages = [
        SystemMessage(content="You are a knowledgeable travel planner. Create detailed, realistic daily itineraries that fit the budget and preferences specified."),
        HumanMessage(content=prompt)
    ]

    functions = [{
        "name": "create_daily_itinerary",
        "description": "Create detailed daily itinerary",
        "parameters": {
            "type": "object",
            "properties": {
                "itinerary": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "day": {"type": "string"},
                            "details": {
                                "type": "object",
                                "properties": {
                                    "morning": {"type": "string"},
                                    "afternoon": {"type": "string"},
                                    "evening": {"type": "string"},
                                    "meals": {"type": "string"},
                                    "estimated_costs": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            },
            "required": ["itinerary"]
        }
    }
    ]

    response = chat_model.invoke(
        messages,
        functions=functions,
        function_call={"name": "create_daily_itinerary"}
    )

    if response.additional_kwargs.get("function_call"):
        function_args = json.loads(response.additional_kwargs["function_call"]["arguments"])
        return function_args["itinerary"]
    
    return []

def calculate_total_cost(itinerary):
    """calculate total trip costs using the model generated itinerary"""
    messages = [
        SystemMessage(content="You are a trip cost calculator. Calculate the total trip cost using the itinerary provided"),
        HumanMessage(content=f"Itinerary:{itinerary}")
    ]

    functions = [
    {
        "name": "total_costs_calculator",
        "description": "calculate the total costs based on detailed daily itinerary",
        "parameters": {
            "type": "object",
            "properties": {
                "itinerary_costs": {
                    "type": "object",
                    "properties": {
                        "total_stay_costs": {"type": "string"},
                        "total_transportation_costs": {"type": "string"},
                        "total_meal_costs": {"type": "string"},
                        "total_miscellaneous_costs": {"type": "string"},
                        "total_trip_cost":{"type": "string"},
                    }
                }
            },
            "required": ["itinerary_costs"]
        }
    },    
    ]

    response = chat_model.invoke(
        messages,
        functions=functions,
        function_call={"name": "total_costs_calculator"}
    )

    if response.additional_kwargs.get("function_call"):
        function_args = json.loads(response.additional_kwargs["function_call"]["arguments"])
        return function_args["itinerary_costs"]
    
    return {}

@app.route("/generate_itinerary", methods=["POST"])
def generate_itinerary():
    try:
        data = request.get_json()
        print("Received data:", data)

        first_destination = data.get("destinations", [])[0] if data.get("destinations") else None

        destinations = ";".join(data.get("destinations", [])) if len(data.get("destinations", []))>0 else ""
        # Create initial structure
        trip_details = {
            "details": {
                "origin": data.get("origin", ""),
                "destinations": destinations,
                "budget": data.get("budget", ""),
                "days": int(data.get("days", 5)),
                "currency": data.get("currency", "USD"),
                "groupSize": int(data.get("groupSize", 2)),
                "comfortLevel": data.get("comfortLevel", "moderate"),
                "theme": data.get("theme", "general"),
                "additionalInfo": data.get("additionalInfo", ""),
                "stayPref": data.get("stayPref", "Doesn't matter, base on my budget and comfort level")
            }
        }

        # Generate initial structure
        itinerary = create_initial_itinerary_structure(
            origin=trip_details["details"]["origin"],
            days=trip_details["details"]["days"],
            destinations=trip_details["details"]["destinations"],
            budget=trip_details["details"]["budget"],
            stayPref=trip_details["details"]["stayPref"],
            currency=trip_details["details"]["currency"],
            groupSize=trip_details["details"]["groupSize"],
            comfortLevel=trip_details["details"]["comfortLevel"],
            theme=trip_details["details"]["theme"],
            additionalInfo=trip_details["details"]["additionalInfo"],
        )

        # Populate daily activities
        itinerary["itinerary"] = populate_daily_activities(trip_details)
        itinerary["itinerary_costs"] = calculate_total_cost(itinerary["itinerary"])

        if (first_destination):
            # Initialize OpenAI client
            client = OpenAI()
            # Generate image
            response = client.images.generate(
                model="dall-e-3",
                prompt=f"Generate an image related to {first_destination}",
                size='1024x1024',
                quality='standard',
                n=1
            )
            itinerary["image"] = response.data[0].url

        
        return jsonify(itinerary)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/update_itinerary", methods=["POST"])
def update_itinerary():
    try:
        data = request.get_json()
        current_itinerary = data.get("current_itinerary", {})
        user_suggestion = data.get("user_suggestion", "")

        prompt = f"""
        Update this itinerary based on the following suggestion: {user_suggestion}
        
        Current itinerary:
        {json.dumps(current_itinerary, indent=2)}
        
        Please maintain the same JSON structure but modify the activities and details according to the suggestion.
        Make sure to keep the same level of detail for transportation, costs, and activities.
        """

        messages = [
            SystemMessage(content="You are a travel assistant. Update the provided itinerary based on user suggestions while maintaining the same structure and level of detail."),
            HumanMessage(content=prompt)
        ]

        functions = [{
            "name": "update_daily_itinerary",
            "description": "Update the daily itinerary",
            "parameters": {
                "type": "object",
                "properties": {
                    "itinerary": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "day": {"type": "string"},
                                "details": {
                                    "type": "object",
                                    "properties": {
                                        "morning": {"type": "string"},
                                        "afternoon": {"type": "string"},
                                        "evening": {"type": "string"},
                                        "meals": {"type": "string"},
                                        "estimated_costs": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                },
                "required": ["itinerary"]
            }
        }]

        response = chat_model.invoke(
            messages,
            functions=functions,
            function_call={"name": "update_daily_itinerary"}
        )

        if response.additional_kwargs.get("function_call"):
            function_args = json.loads(response.additional_kwargs["function_call"]["arguments"])
            
            # Update the itinerary
            current_itinerary["itinerary"] = function_args["itinerary"]
            
            # Update the title based on user suggestion
            title_update_prompt = f"""
            Based on this user suggestion: {user_suggestion}
            And the current title: {current_itinerary.get('title', '')}
            Should the title be updated? If yes, generate a new title that reflects the changes.
            If no changes are needed, return the original title.
            """
            
            title_messages = [
                SystemMessage(content="You are a travel assistant. Help update the trip title if the user's suggestions warrant a change."),
                HumanMessage(content=title_update_prompt)
            ]
            
            title_functions = [{
                "name": "update_title",
                "description": "Update the itinerary title if needed",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"}
                    },
                    "required": ["title"]
                }
            }]
            
            title_response = chat_model.invoke(
                title_messages,
                functions=title_functions,
                function_call={"name": "update_title"}
            )
            
            if title_response.additional_kwargs.get("function_call"):
                title_args = json.loads(title_response.additional_kwargs["function_call"]["arguments"])
                current_itinerary["title"] = title_args["title"]
            
            # Recalculate costs
            current_itinerary["itinerary_costs"] = calculate_total_cost(current_itinerary["itinerary"])
            print(current_itinerary["itinerary_costs"])
            
            return jsonify(current_itinerary)

        return jsonify({"error": "Failed to update itinerary"}), 400

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)