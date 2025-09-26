import os
from datetime import timedelta

class Config:
    """Configuration class for Aviation Weather System"""
    
    # Gemini AI Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your-gemini-api-key-here')
    
    # Weather Data Sources
    AVIATION_WEATHER_API = 'https://aviationweather.gov/api/data/'
    CHECKWX_API = 'https://api.checkwx.com/'
    CHECKWX_API_KEY = os.getenv('CHECKWX_API_KEY', '')  # Optional backup source
    
    # Safety Configuration
    MAX_ANALYSIS_TIMEOUT = 30  # seconds
    FALLBACK_ANALYSIS_ENABLED = True
    LOG_LEVEL = 'INFO'
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE = 60
    
    # Cache Configuration
    CACHE_TIMEOUT = timedelta(minutes=5)  # Weather data cache
    
    # Airport Database
    MAJOR_INDIAN_AIRPORTS = {
        'VOBL': {'name': 'Kempegowda International Airport', 'city': 'Bangalore', 'lat': 12.9716, 'lon': 77.5946},
        'VIDP': {'name': 'Indira Gandhi International Airport', 'city': 'Delhi', 'lat': 28.5562, 'lon': 77.1000},
        'VABB': {'name': 'Chhatrapati Shivaji International Airport', 'city': 'Mumbai', 'lat': 19.0896, 'lon': 72.8656},
        'VECC': {'name': 'Netaji Subhas Chandra Bose International Airport', 'city': 'Kolkata', 'lat': 22.6547, 'lon': 88.4467},
        'VOMM': {'name': 'Chennai International Airport', 'city': 'Chennai', 'lat': 13.0827, 'lon': 80.2707},
        'VOHS': {'name': 'Rajiv Gandhi International Airport', 'city': 'Hyderabad', 'lat': 17.2313, 'lon': 78.4298},
        'VEGT': {'name': 'Bagdogra Airport', 'city': 'Siliguri/Sikkim', 'lat': 27.4840, 'lon': 88.5419},
        'VAID': {'name': 'Sardar Vallabhbhai Patel International Airport', 'city': 'Ahmedabad', 'lat': 23.0775, 'lon': 72.6347},
        'VOTR': {'name': 'Trivandrum International Airport', 'city': 'Thiruvananthapuram', 'lat': 10.7669, 'lon': 76.4091},
        'VOBZ': {'name': 'Jay Prakash Narayan Airport', 'city': 'Patna', 'lat': 25.4479, 'lon': 85.3239},
        'VOCB': {'name': 'Coimbatore International Airport', 'city': 'Coimbatore', 'lat': 11.0297, 'lon': 77.0438},
        'VAGO': {'name': 'Dabolim Airport', 'city': 'Goa', 'lat': 15.3808, 'lon': 73.8314},
        'VEJH': {'name': 'Jharsuguda Airport', 'city': 'Jharsuguda', 'lat': 21.9133, 'lon': 84.0508},
    }
    
    # Common flight routes in India
    COMMON_ROUTES = [
        ['VOBL', 'VIDP'],  # Bangalore to Delhi
        ['VOBL', 'VABB'],  # Bangalore to Mumbai
        ['VIDP', 'VABB'],  # Delhi to Mumbai
        ['VOBL', 'VECC'],  # Bangalore to Kolkata
        ['VOBL', 'VEGT'],  # Bangalore to Sikkim
        ['VIDP', 'VEGT'],  # Delhi to Sikkim
        ['VABB', 'VEGT'],  # Mumbai to Sikkim
    ]
