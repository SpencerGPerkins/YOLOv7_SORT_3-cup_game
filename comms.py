import json
import requests

def format_locations(in_data):
    """Format the data from detect.py to be sent to the llm"""
    formatted_objects = []

    for data in in_data:
        cls, center_x, center_y, section = data
        if cls == 0:
            object_class = "ball"
        elif cls == 1:
            object_class = "cup"
        formatted_objects.append(
            {
                "class": object_class,
                "center_coords": {"center_x": center_x, "center_y": center_y},
                "section": section
                }
            )
        
    return formatted_objects

def send_to_llm(data, channel_access_token, user_id):
    url = ''
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {channel_access_token}'
    }
    payload = {
        'to': user_id,
        'messages': [
            {
                'type': 'text',
                'text': json.dumps(data, indent=2)  # Convert data to JSON string
            }
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.status_code


