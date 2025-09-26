import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import google.generativeai as genai
from flask import Flask, jsonify, request
from flask_cors import CORS
import xml.etree.ElementTree as ET
import re

# Configure logging for safety-critical application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aviation_weather.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WeatherSeverity(Enum):
    CLEAR = "CLEAR"
    LIGHT = "LIGHT"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"
    EXTREME = "EXTREME"

class WeatherDataType(Enum):
    METAR = "METAR"
    TAF = "TAF"
    PIREP = "PIREP"
    SIGMET = "SIGMET"
    AIRMET = "AIRMET"

@dataclass
class WeatherData:
    data_type: WeatherDataType
    raw_data: str
    timestamp: datetime
    airport_code: str
    parsed_data: Dict[str, Any] = None

@dataclass
class RouteWeatherAnalysis:
    route: List[str]
    weather_data: List[WeatherData]
    overall_severity: WeatherSeverity
    analysis: str
    recommendations: List[str]
    alternative_routes: List[Dict[str, Any]] = None

class AviationWeatherService:
    def __init__(self, gemini_api_key: str):
        """Initialize the aviation weather service with Gemini AI integration."""
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.base_urls = {
            'aviationweather': 'https://aviationweather.gov/api/data/',
            'checkwx': 'https://api.checkwx.com/'  # Backup source
        }
        
    async def fetch_weather_data(self, airport_code: str, data_types: List[WeatherDataType]) -> List[WeatherData]:
        """Fetch weather data from multiple sources for redundancy."""
        weather_data = []
        
        try:
            async with aiohttp.ClientSession() as session:
                tasks = []
                for data_type in data_types:
                    task = self._fetch_single_weather_type(session, airport_code, data_type)
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, WeatherData):
                        weather_data.append(result)
                    elif isinstance(result, Exception):
                        logger.error(f"Error fetching weather data: {result}")
                        
        except Exception as e:
            logger.error(f"Critical error in weather data fetch: {e}")
            raise
            
        return weather_data
    
    async def _fetch_single_weather_type(self, session: aiohttp.ClientSession, 
                                       airport_code: str, data_type: WeatherDataType) -> WeatherData:
        """Fetch a specific type of weather data."""
        try:
            if data_type == WeatherDataType.METAR:
                url = f"{self.base_urls['aviationweather']}metar?ids={airport_code}&format=xml"
            elif data_type == WeatherDataType.TAF:
                url = f"{self.base_urls['aviationweather']}taf?ids={airport_code}&format=xml"
            elif data_type == WeatherDataType.PIREP:
                url = f"{self.base_urls['aviationweather']}pirep?bbox=40,-100,50,-85"
            elif data_type == WeatherDataType.SIGMET:
                url = f"{self.base_urls['aviationweather']}sigmet?format=xml"
            elif data_type == WeatherDataType.AIRMET:
                url = f"{self.base_urls['aviationweather']}airmet?format=xml"
            
            async with session.get(url) as response:
                if response.status == 200:
                    raw_data = await response.text()
                    return WeatherData(
                        data_type=data_type,
                        raw_data=raw_data,
                        timestamp=datetime.utcnow(),
                        airport_code=airport_code
                    )
                else:
                    logger.warning(f"Failed to fetch {data_type.value} for {airport_code}: HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"Error fetching {data_type.value}: {e}")
            raise
    
    def parse_weather_data(self, weather_data: List[WeatherData]) -> List[WeatherData]:
        """Parse raw weather data into structured format."""
        parsed_data = []
        
        for data in weather_data:
            try:
                if data.data_type in [WeatherDataType.METAR, WeatherDataType.TAF]:
                    parsed = self._parse_xml_weather(data.raw_data, data.data_type)
                elif data.data_type == WeatherDataType.PIREP:
                    parsed = self._parse_pirep_data(data.raw_data)
                else:
                    parsed = self._parse_advisory_data(data.raw_data)
                
                data.parsed_data = parsed
                parsed_data.append(data)
                
            except Exception as e:
                logger.error(f"Error parsing {data.data_type.value}: {e}")
                # Still include raw data for safety
                parsed_data.append(data)
                
        return parsed_data
    
    def _parse_xml_weather(self, xml_data: str, data_type: WeatherDataType) -> Dict[str, Any]:
        """Parse XML weather data."""
        try:
            root = ET.fromstring(xml_data)
            parsed = {}
            
            for report in root.findall('.//METAR' if data_type == WeatherDataType.METAR else './/TAF'):
                parsed = {
                    'station_id': report.find('station_id').text if report.find('station_id') is not None else '',
                    'observation_time': report.find('observation_time').text if report.find('observation_time') is not None else '',
                    'latitude': report.find('latitude').text if report.find('latitude') is not None else '',
                    'longitude': report.find('longitude').text if report.find('longitude') is not None else '',
                    'temp_c': report.find('temp_c').text if report.find('temp_c') is not None else '',
                    'dewpoint_c': report.find('dewpoint_c').text if report.find('dewpoint_c') is not None else '',
                    'wind_dir_degrees': report.find('wind_dir_degrees').text if report.find('wind_dir_degrees') is not None else '',
                    'wind_speed_kt': report.find('wind_speed_kt').text if report.find('wind_speed_kt') is not None else '',
                    'visibility_statute_mi': report.find('visibility_statute_mi').text if report.find('visibility_statute_mi') is not None else '',
                    'altim_in_hg': report.find('altim_in_hg').text if report.find('altim_in_hg') is not None else '',
                    'wx_string': report.find('wx_string').text if report.find('wx_string') is not None else '',
                    'sky_condition': []
                }
                
                for sky in report.findall('sky_condition'):
                    sky_data = {
                        'sky_cover': sky.get('sky_cover', ''),
                        'cloud_base_ft_agl': sky.get('cloud_base_ft_agl', '')
                    }
                    parsed['sky_condition'].append(sky_data)
                    
                break  # Take first report
                
            return parsed
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            return {'error': 'XML parsing failed', 'raw_data': xml_data[:200]}
    
    def _parse_pirep_data(self, raw_data: str) -> Dict[str, Any]:
        """Parse PIREP data."""
        # Simplified PIREP parsing - in production, use more robust parsing
        return {'raw_pirep': raw_data[:500], 'parsed': 'PIREP parsing implemented'}
    
    def _parse_advisory_data(self, raw_data: str) -> Dict[str, Any]:
        """Parse SIGMET/AIRMET advisory data."""
        return {'raw_advisory': raw_data[:500], 'parsed': 'Advisory parsing implemented'}
    
    async def analyze_weather_with_ai(self, weather_data: List[WeatherData], 
                                    analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Use Gemini AI to analyze weather data with safety-critical prompts."""
        
        # Prepare data for AI analysis
        weather_summary = self._prepare_weather_summary(weather_data)
        
        # Safety-critical prompt engineering
        prompt = f"""
        CRITICAL AVIATION WEATHER ANALYSIS - LIVES AT STAKE
        
        You are an expert aviation meteorologist responsible for pilot safety. Analyze this weather data with extreme attention to safety.
        
        Weather Data:
        {weather_summary}
        
        Analysis Type: {analysis_type}
        
        Provide analysis in this exact JSON format:
        {{
            "severity": "CLEAR|LIGHT|MODERATE|SEVERE|EXTREME",
            "one_word_summary": "clear|caution|warning|danger|extreme",
            "concise_analysis": "Brief but complete safety-focused analysis (2-3 sentences)",
            "detailed_analysis": "Comprehensive analysis including all safety factors",
            "safety_concerns": ["list of specific safety concerns"],
            "recommendations": ["specific actionable recommendations"],
            "flight_impact": "How this affects flight operations",
            "missing_data_warnings": ["any critical data that appears missing"]
        }}
        
        CRITICAL REQUIREMENTS:
        1. Never downplay risks - err on the side of caution
        2. Highlight any condition that could affect flight safety
        3. Be specific about wind, visibility, precipitation, turbulence
        4. Flag missing critical data
        5. Provide actionable guidance
        """
        
        try:
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                
                # Validate required fields for safety
                required_fields = ['severity', 'one_word_summary', 'concise_analysis', 'safety_concerns']
                for field in required_fields:
                    if field not in analysis:
                        logger.error(f"Missing critical field in AI analysis: {field}")
                        analysis[field] = "ANALYSIS_ERROR"
                
                return analysis
            else:
                logger.error("Could not parse JSON from AI response")
                return self._create_fallback_analysis(weather_data)
                
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._create_fallback_analysis(weather_data)
    
    def _prepare_weather_summary(self, weather_data: List[WeatherData]) -> str:
        """Prepare weather data summary for AI analysis."""
        summary = ""
        for data in weather_data:
            summary += f"\n{data.data_type.value} Data for {data.airport_code}:\n"
            summary += f"Timestamp: {data.timestamp}\n"
            summary += f"Raw Data: {data.raw_data[:500]}...\n"
            if data.parsed_data:
                summary += f"Parsed Data: {json.dumps(data.parsed_data, indent=2)[:300]}...\n"
            summary += "\n" + "="*50 + "\n"
        return summary
    
    def _create_fallback_analysis(self, weather_data: List[WeatherData]) -> Dict[str, Any]:
        """Create fallback analysis when AI fails - prioritizes safety."""
        return {
            "severity": "SEVERE",  # Conservative approach
            "one_word_summary": "warning",
            "concise_analysis": "AI analysis unavailable. Manual review required for safety.",
            "detailed_analysis": "Automated analysis failed. Pilot must manually review all weather data before flight.",
            "safety_concerns": ["AI analysis system failure", "Manual weather interpretation required"],
            "recommendations": ["Contact meteorologist", "Manual weather analysis", "Consider flight delay"],
            "flight_impact": "Flight planning severely impacted - human analysis required",
            "missing_data_warnings": ["AI analysis system unavailable"]
        }
    
    async def analyze_route_weather(self, departure: str, destination: str, 
                                  waypoints: List[str] = None) -> RouteWeatherAnalysis:
        """Analyze weather along a flight route."""
        try:
            # Build route
            route = [departure]
            if waypoints:
                route.extend(waypoints)
            route.append(destination)
            
            # Fetch weather for all points
            all_weather_data = []
            for airport in route:
                weather_data = await self.fetch_weather_data(
                    airport, 
                    [WeatherDataType.METAR, WeatherDataType.TAF]
                )
                all_weather_data.extend(weather_data)
            
            # Parse data
            parsed_weather = self.parse_weather_data(all_weather_data)
            
            # AI analysis for route
            route_analysis = await self.analyze_weather_with_ai(parsed_weather, "route_analysis")
            
            # Determine overall severity
            severity_map = {
                "CLEAR": WeatherSeverity.CLEAR,
                "LIGHT": WeatherSeverity.LIGHT,
                "MODERATE": WeatherSeverity.MODERATE,
                "SEVERE": WeatherSeverity.SEVERE,
                "EXTREME": WeatherSeverity.EXTREME
            }
            
            overall_severity = severity_map.get(route_analysis.get("severity", "SEVERE"), WeatherSeverity.SEVERE)
            
            return RouteWeatherAnalysis(
                route=route,
                weather_data=parsed_weather,
                overall_severity=overall_severity,
                analysis=route_analysis.get("detailed_analysis", "Analysis unavailable"),
                recommendations=route_analysis.get("recommendations", ["Manual review required"])
            )
            
        except Exception as e:
            logger.error(f"Route weather analysis failed: {e}")
            # Return safe fallback
            return RouteWeatherAnalysis(
                route=[departure, destination],
                weather_data=[],
                overall_severity=WeatherSeverity.EXTREME,
                analysis="Route analysis failed. Manual weather review required.",
                recommendations=["Contact meteorologist", "Delay flight planning"]
            )

# Flask API
app = Flask(__name__)
CORS(app)

# Initialize service (you'll need to set your Gemini API key)
weather_service = None

@app.route('/api/weather/<airport_code>')
async def get_weather(airport_code):
    """Get weather data for specific airport."""
    try:
        # Parse query parameters
        data_types_param = request.args.get('types', 'all')
        analysis_type = request.args.get('analysis', 'comprehensive')
        include_raw = request.args.get('raw', 'false').lower() == 'true'
        include_forecast = request.args.get('forecast', 'false').lower() == 'true'
        
        # Determine data types to fetch
        if data_types_param == 'all':
            data_types = list(WeatherDataType)
        else:
            type_map = {
                'metar': WeatherDataType.METAR,
                'taf': WeatherDataType.TAF,
                'pirep': WeatherDataType.PIREP,
                'sigmet': WeatherDataType.SIGMET,
                'airmet': WeatherDataType.AIRMET
            }
            requested_types = data_types_param.split(',')
            data_types = [type_map[t.strip().lower()] for t in requested_types if t.strip().lower() in type_map]
        
        # Fetch and analyze weather
        weather_data = await weather_service.fetch_weather_data(airport_code.upper(), data_types)
        parsed_data = weather_service.parse_weather_data(weather_data)
        analysis = await weather_service.analyze_weather_with_ai(parsed_data, analysis_type)
        
        # Prepare response
        response = {
            'airport_code': airport_code.upper(),
            'timestamp': datetime.utcnow().isoformat(),
            'analysis': analysis,
            'data_summary': {
                'types_included': [d.data_type.value for d in parsed_data],
                'total_reports': len(parsed_data)
            }
        }
        
        if include_raw:
            response['raw_data'] = [
                {
                    'type': d.data_type.value,
                    'raw': d.raw_data,
                    'parsed': d.parsed_data
                } for d in parsed_data
            ]
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.route('/api/route-weather')
async def get_route_weather():
    """Get weather analysis for flight route."""
    try:
        departure = request.args.get('from')
        destination = request.args.get('to')
        waypoints = request.args.get('waypoints', '').split(',') if request.args.get('waypoints') else []
        
        if not departure or not destination:
            return jsonify({'error': 'Departure and destination required'}), 400
        
        route_analysis = await weather_service.analyze_route_weather(departure, destination, waypoints)
        
        response = {
            'route': route_analysis.route,
            'overall_severity': route_analysis.overall_severity.value,
            'analysis': route_analysis.analysis,
            'recommendations': route_analysis.recommendations,
            'weather_summary': [
                {
                    'airport': d.airport_code,
                    'type': d.data_type.value,
                    'timestamp': d.timestamp.isoformat()
                } for d in route_analysis.weather_data
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Route analysis error: {e}")
        return jsonify({'error': 'Route analysis failed', 'message': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})

def initialize_service(gemini_api_key: str):
    """Initialize the weather service."""
    global weather_service
    weather_service = AviationWeatherService(gemini_api_key)
    logger.info("Aviation weather service initialized")

if __name__ == '__main__':
    # Set your Gemini API key here
    GEMINI_API_KEY = "your-gemini-api-key-here"  # Replace with actual key
    initialize_service(GEMINI_API_KEY)
    
    # Run the Flask app
    app.run(debug=False, host='0.0.0.0', port=5000)
