from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from openai import error
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from newscript import simplify_flight_data_for_llm
from fastembed import TextEmbedding
from typing import List
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Initialize OpenAI client only for chat completions
openai.api_key = os.getenv('OPENAI_API_KEY')

class VectorStore:
    MAX_CHUNK_LENGTH = 8000

    def __init__(self):
        self.data = []
        self.embeddings = None
        self.chunks = []
        print("Initializing embedding model...")
        try:
            self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            print("Embedding model ready!")
        except Exception as e:
            print(f"Error loading primary model: {e}")
            print("Attempting to load fallback model...")
            try:
                # Try a different model as fallback
                self.embedding_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
                print("Fallback embedding model ready!")
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                raise Exception("Failed to initialize any embedding model. Please check your internet connection and try again.")

    def add_data(self, flight_data):
        print("\n[Embedding] Starting data processing...")
        
        # Simplify flight data for LLM
        simplified_data = simplify_flight_data_for_llm(flight_data)
        
        # Create chunks from simplified data
        self.chunks = []
        for msg_type, data in simplified_data.items():
            if msg_type in ["flight_log_summary"]:
                continue
                
            chunk = {
                "message_type": msg_type,
                "content": f"Message Type: {msg_type}\nDescription: {data.get('description', '')}\nRecord Count: {data.get('record_count', 0)}\n"
            }
            
            # Add numerical statistics
            if "numerical_statistics" in data:
                chunk["content"] += "\nNumerical Statistics:\n"
                for field, stats in data["numerical_statistics"].items():
                    chunk["content"] += f"{field}: min={stats['min']}, max={stats['max']}, avg={stats['avg']:.2f}\n"
            
            # Add sample data
            if "sample_data" in data:
                chunk["content"] += "\nSample Data:\n"
                for section in ["start", "mid", "end"]:
                    if section in data["sample_data"]:
                        chunk["content"] += f"\n{section.capitalize()} samples:\n"
                        for sample in data["sample_data"][section]:
                            chunk["content"] += f"{json.dumps(sample, indent=2)}\n"
            
            # Add messages for MSG type
            if msg_type == "MSG" and "messages" in data:
                chunk["content"] += "\nMessages:\n"
                for msg in data["messages"]:
                    chunk["content"] += f"{msg}\n"
            
            self.chunks.append(chunk)
            if len(self.chunks) <= 3: # Print only the first 3 chunks for visibility
                print(f"[Chunk Creation] Added chunk for {msg_type}: {chunk['content'][:150]}...") # Print first 150 chars of content
        
        # Extract content from chunks for embedding
        self.data = [chunk["content"] for chunk in self.chunks]
        
        # Generate embeddings with detailed progress
        print("[Embedding] Generating embeddings (this may take a while)...")
        
        # Batch processing with progress bar
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(self.data), batch_size), desc="Generating embeddings"):
            batch = self.data[i:i + batch_size]
            embeddings_batch = list(self.embedding_model.embed(batch))
            embeddings.extend(embeddings_batch)
            print(f"Processed batch {i//batch_size + 1}/{(len(self.data)-1)//batch_size + 1}")
        
        self.embeddings = np.array(embeddings)
        print(f"[Embedding] Completed! Generated {len(self.embeddings)} embeddings")

    def _get_embeddings(self, texts):
        print(f"[Embedding] Generating query embedding for: {texts[0][:50]}...")
        embeddings = list(self.embedding_model.embed(texts))
        print("[Embedding] Query embedding complete")
        return embeddings[0]

    def search(self, query, k=5, relevant_message_types=None):
        if self.embeddings is None or self.embeddings.size == 0:
            return []
        
        print(f"[VectorStore Search] Searching with query: {query[:100]}...")
        print(f"[VectorStore Search] Received relevant_message_types: {relevant_message_types}")

        # Get query embedding using local model
        query_embedding = self._get_embeddings([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity([query_embedding], self.embeddings).flatten()
        
        # Get top k most similar chunks
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        # Filter by relevant message types if specified
        filtered_chunks = []
        initial_top_k_count = 0
        for i in top_k_indices:
            initial_top_k_count += 1
            chunk = self.chunks[i]
            if relevant_message_types is None or not relevant_message_types or chunk['message_type'] in relevant_message_types:
                filtered_chunks.append(chunk)
        
        print(f"[VectorStore Search] Initial top {k} chunks: {initial_top_k_count}, Filtered chunks: {len(filtered_chunks)}")
        
        return filtered_chunks

class FlightDataAnalyzer:
    def __init__(self):
        self.conversation_history = []
        self.flight_data = None
        self.vector_store = VectorStore()

    def set_flight_data(self, data):
        self.flight_data = data
        self.vector_store.add_data(data)

    def _optimize_query_with_docs(self, query: str) -> str:
        optimization_prompt = f"""You are an expert in ArduPilot MAVLink log messages. Your task is to identify the most relevant MAVLink message types and their fields for answering a user's question.

        Here is the ArduPilot log messages documentation:
        ---
        {self._get_mavlink_docs_subset()}
        ---

        For the given query, identify ONLY the most relevant message types and their specific fields that would contain the information needed to answer the question. Be precise and concise.

        Format your response as: "Relevant messages and fields: [MESSAGE_TYPE].[FIELD], [MESSAGE_TYPE].[FIELD], ..."

        Examples:
        Query: "What was the highest altitude of the flight?"
        Response: "Relevant messages and fields: GPS.Alt, AHR2.Alt, POS.Alt, POS.RelAlt"

        Query: "Tell me about the battery performance."
        Response: "Relevant messages and fields: BAT.Volt, BAT.Curr, BAT.Consum"

        Query: "Why did the drone crash?"
        Response: "Relevant messages and fields: MSG.Message, EV.Id, STAT.Sensor, VIB.VibeX, VIB.VibeY, VIB.VibeZ"

        Query: "{query}"
        Response:"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": optimization_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error optimizing query with docs: {e}")
            return f"Error in query optimization: {e}. Original query: {query}"

    def _get_mavlink_docs_subset(self) -> str:
        # This is a subset of the documentation I have from the web search.
        # In a real scenario, this would ideally be loaded from a more structured knowledge base or a dedicated file.
        return """
# Onboard Message Log Messages

## ACC
IMU accelerometer data
| TimeUS | μs | Time since system startup |
| AccX | m/s/s | acceleration along X axis |
| AccY | m/s/s | acceleration along Y axis |
| AccZ | m/s/s | acceleration along Z axis |

## ADSB
Automatic Dependent Serveillance - Broadcast detected vehicle information
| Lat | 1e-7 deglatitude | Vehicle latitude |
| Lng | 1e-7 deglongitude | Vehicle longitude |
| Alt | mm | Vehicle altitude |
| Heading | cdegheading | Vehicle heading |

## AHR2
Backup AHRS data
| Roll | deg | Estimated roll |
| Pitch | deg | Estimated pitch |
| Yaw | degheading | Estimated yaw |
| Alt | m | Estimated altitude |
| Lat | deglatitude | Estimated latitude |
| Lng | deglongitude | Estimated longitude |
| Q1 | Estimated attitude quaternion component 1 |
| Q2 | Estimated attitude quaternion component 2 |
| Q3 | Estimated attitude quaternion component 3 |
| Q4 | Estimated attitude quaternion component 4 |

## ATT
Attitude (Roll, Pitch, Yaw, Roll Speed, Pitch Speed, Yaw Speed).
| Roll | deg | Estimated roll |
| Pitch | deg | Estimated pitch |
| Yaw | degheading | Estimated yaw |
| RollRate | deg/s | Roll angular velocity |
| PitchRate | deg/s | Pitch angular velocity |
| YawRate | deg/s | Yaw angular velocity |

## GPS
GPS Global Position (Latitude, Longitude, Altitude, Satellites Visible, Fix Type).
| Lat | 1e-7 deglatitude | Vehicle latitude |
| Lng | 1e-7 deglongitude | Vehicle longitude |
| Alt | mm | Vehicle altitude |
| NSats | # | Number of visible satellites |
| FixType | Type | GPS Fix Type |

## MODE
Current Flight Mode.
| Mode | uint8 | Flight Mode number |

## CMD
Command executed by the autopilot.
| Cmd | uint16 | Command ID |
| Prm1 | float | Parameter 1 |
| Prm2 | float | Parameter 2 |
| Prm3 | float | Parameter 3 |
| Prm4 | float | Parameter 4 |
| Prm5 | float | Parameter 5 |
| Prm6 | float | Parameter 6 |
| Prm7 | float | Parameter 7 |

## MSG
General messages or events from the autopilot.
| Message | char | Text message from autopilot |

## EV
Event messages.
| Id | uint8 | Event ID |

## PARM
Parameter values and changes.
| Name | char | Parameter Name |
| Value | float | Parameter Value |

## POS
Position (Altitude, Relative Altitude).
| Alt | m | Estimated altitude |
| RelAlt | m | Estimated relative altitude |

## BAT
Battery status (Voltage, Current, Consumed mAh).
| Volt | V | Battery voltage |
| Curr | A | Battery current |
| Consum | mAh | Battery consumption |

## VIB
Vibration levels.
| VibeX | m/s/s | Vibration on X axis |
| VibeY | m/s/s | Vibration on Y axis |
| VibeZ | m/s/s | Vibration on Z axis |

## RCIN
Raw RC input channels.
| C1 | uint16 | Channel 1 PWM |
| C2 | uint16 | Channel 2 PWM |
| C3 | uint16 | Channel 3 PWM |
| C4 | uint16 | Channel 4 PWM |
| C5 | uint16 | Channel 5 PWM |
| C6 | uint16 | Channel 6 PWM |
| C7 | uint16 | Channel 7 PWM |
| C8 | uint16 | Channel 8 PWM |

## RCOUT
Raw RC output channels.
| C1 | uint16 | Channel 1 PWM |
| C2 | uint16 | Channel 2 PWM |
| C3 | uint16 | Channel 3 PWM |
| C4 | uint16 | Channel 4 PWM |
| C5 | uint16 | Channel 5 PWM |
| C6 | uint16 | Channel 6 PWM |
| C7 | uint16 | Channel 7 PWM |
| C8 | uint16 | Channel 8 PWM |

## STAT
System status (CPU Load, Sensors health).
| Load | uint16 | CPU Load (%) |
| Sensor | uint16 | Sensor health bitmap |
"""

    def get_system_prompt(self):
        return """You are a specialized flight data analysis assistant focused on MAVLink protocol data. Your primary goal is to provide clear, accurate, and actionable insights from flight logs.

        For each analysis, you will receive:
        1. The user's specific question
        2. Optimized query context identifying relevant MAVLink messages and fields
        3. Retrieved flight data chunks containing:
           - Message Type and Description
           - Record Count
           - Numerical Statistics (min, max, avg)
           - Sample Data (start, mid, end of flight)
           - Messages (for MSG type)

        Format your response using the following structure:

        📊 **SUMMARY**
        [One-line direct answer to the user's question]

        📈 **KEY FINDINGS**
        • [Finding 1]
        • [Finding 2]
        • [Finding 3]

        🔍 **DETAILED ANALYSIS**
        [Detailed explanation of the findings, with specific data points and their significance
         "look for sudden changes in altitude, battery voltage, or inconsistent GPS lock" etc.)]

        ⚠️ **NOTES & CAVEATS**
        • [Any important limitations or considerations]
        • [Data quality issues or gaps]
        • [Assumptions made]

        💡 **RECOMMENDATIONS**
        • [Specific recommendation 1]
        • [Specific recommendation 2]

        Analysis Guidelines:
        1. **Direct Answer First**: Start with a clear, concise answer to the user's question.
        2. **Data Validation**: 
           - Cross-reference values across different message types
           - Note any discrepancies
           - Highlight data quality issues
        3. **Contextual Analysis**:
           - Explain relationships between message types
           - Identify patterns and anomalies
           - Highlight critical events
        4. **Clear Communication**:
           - Use precise technical terms
           - Explain assumptions
           - Support conclusions with data
        5. **Actionable Insights**:
           - Suggest specific next steps
           - Recommend additional analysis
           - Highlight critical findings

        Formatting Rules:
        1. Use **bold** for important terms and values
        2. Use `code` formatting for specific message types and fields
        3. Use bullet points (•) for lists
        4. Use emojis for section headers
        5. Use --- for separating major sections
        6. Highlight critical values with **bold** and `code` formatting

        Example Format:
        📊 **SUMMARY**
        The flight reached a maximum altitude of **125.5 meters** according to GPS data.

        📈 **KEY FINDINGS**
        • Maximum altitude: **125.5m** (`GPS.Alt`)
        • Minimum altitude: **0.8m** (`GPS.Alt`)
        • Average altitude: **45.2m** (`GPS.Alt`)

        🔍 **DETAILED ANALYSIS**
        The altitude data from `GPS` shows a steady climb phase followed by...

        ⚠️ **NOTES & CAVEATS**
        • Altitude data is available from both `GPS` and `AHR2` sensors
        • Some gaps in GPS data during mid-flight

        💡 **RECOMMENDATIONS**
        • Consider analyzing `AHR2` data for comparison
        • Review flight mode changes during altitude variations

        Reference: https://ardupilot.org/plane/docs/logmessages.html
        """

    def generate_plot_data(self, chunks, query):
        """Generate plot data based on the retrieved chunks and query."""
        try:
            print("[Plot Generation] Starting plot generation...")
            # Extract numerical data from chunks
            plot_data = {}
            for chunk in chunks:
                msg_type = chunk['message_type']
                print(f"[Plot Generation] Processing chunk for message type: {msg_type}")
                
                # Get the content from the chunk
                content = chunk['content']
                print(f"[Plot Generation] Raw content: {content[:200]}...")
                
                # Look for numerical statistics section
                if 'Numerical Statistics:' in content:
                    try:
                        # Extract the numerical statistics section
                        stats_section = content.split('Numerical Statistics:')[1]
                        if 'Sample Data:' in stats_section:
                            stats_section = stats_section.split('Sample Data:')[0]
                        
                        print(f"[Plot Generation] Found stats section: {stats_section[:200]}...")
                        
                        fields = {}
                        for line in stats_section.strip().split('\n'):
                            if ':' in line:
                                field, values = line.split(':', 1)
                                field = field.strip()
                                values = values.strip()
                                
                                # Parse min, max, avg values
                                if 'min=' in values and 'max=' in values and 'avg=' in values:
                                    try:
                                        # Extract values using more robust parsing
                                        min_str = values.split('min=')[1].split(',')[0]
                                        max_str = values.split('max=')[1].split(',')[0]
                                        avg_str = values.split('avg=')[1].split(',')[0]
                                        
                                        # Convert to float, handling scientific notation
                                        min_val = float(min_str)
                                        max_val = float(max_str)
                                        avg_val = float(avg_str)
                                        
                                        # Only add fields that are relevant to the query
                                        if any(keyword in field.lower() for keyword in ['alt', 'height', 'altitude']):
                                            fields[field] = {
                                                'min': min_val,
                                                'max': max_val,
                                                'avg': avg_val
                                            }
                                            print(f"[Plot Generation] Parsed field {field}: min={min_val}, max={max_val}, avg={avg_val}")
                                    except ValueError as e:
                                        print(f"[Plot Generation] Error parsing values for {field}: {e}")
                        
                        if fields:
                            plot_data[msg_type] = fields
                            print(f"[Plot Generation] Added data for {msg_type}: {fields}")
                    except Exception as e:
                        print(f"[Plot Generation] Error processing stats section: {e}")

            # Generate appropriate plot based on data and query
            if not plot_data:
                print("[Plot Generation] No numerical data found for plotting")
                return None

            print("[Plot Generation] Creating plot with data:", plot_data)
            # Create subplots based on available data
            fig = make_subplots(rows=len(plot_data), cols=1, 
                               subplot_titles=[f"{msg_type} Data" for msg_type in plot_data.keys()])

            row = 1
            for msg_type, fields in plot_data.items():
                for field, values in fields.items():
                    # Create box plot for each field
                    fig.add_trace(
                        go.Box(
                            y=[values['min'], values['avg'], values['max']],
                            name=f"{msg_type}.{field}",
                            boxpoints='all',
                            jitter=0.3,
                            pointpos=-1.8
                        ),
                        row=row, col=1
                    )
                row += 1

            # Update layout
            fig.update_layout(
                height=300 * len(plot_data),
                showlegend=True,
                title_text="Flight Data Analysis",
                template="plotly_white"
            )

            plot_json = fig.to_json()
            print("[Plot Generation] Successfully generated plot JSON")
            return plot_json

        except Exception as e:
            print(f"[Plot Generation] Error generating plot: {e}")
            return None

    def analyze_query(self, query):
        if not self.flight_data:
            return "No flight data available. Please upload a flight log first."

        try:
            # Step 1: Optimize the user query using the MAVLink documentation
            optimized_query_context = self._optimize_query_with_docs(query)
            print(f"[Query Optimization] Optimized Query Context: {optimized_query_context}")

            # Extract relevant message types from the optimized query context
            relevant_message_types = []
            if "Relevant messages and fields:" in optimized_query_context:
                parts = optimized_query_context.split("Relevant messages and fields:")[1].strip().split(', ')
                for part in parts:
                    if "." in part:
                        relevant_message_types.append(part.split('.')[0])
                    elif part: # For cases like 'overall_flight_duration' without a dot
                        relevant_message_types.append(part)
            relevant_message_types = list(set(relevant_message_types)) # Remove duplicates
            print(f"[Query Optimization] Identified Relevant Message Types: {relevant_message_types}")

            # Use the optimized query context for vector store search
            search_query = optimized_query_context
            print(f"[VectorStore Search] Using optimized query: {search_query}")

            # Get relevant chunks with metadata, now potentially filtered by message types
            relevant_chunks = self.vector_store.search(search_query, relevant_message_types=relevant_message_types)
            
            # Generate plot data if relevant
            plot_data = self.generate_plot_data(relevant_chunks, query)
            
            # Format chunks for context
            context_parts = []
            for chunk in relevant_chunks:
                context_parts.append(f"\nMessage Type: {chunk['message_type']}")
                context_parts.append("Content:")
                context_parts.append(chunk['content'])
            
            context_str = "\n".join(context_parts)
            print(f"[Context Retrieval] Context sent to LLM:\n---\n{context_str[:500]}...\n---") # Print first 500 chars of context

            # Add plot data to the response if available
            if plot_data:
                system_prompt_with_optimization = self.get_system_prompt() + "\n\nQuery Context:\n" + optimized_query_context + "\n\nRetrieved Flight Data:\n" + context_str + "\n\nNote: A plot has been generated for this data."
            else:
                system_prompt_with_optimization = self.get_system_prompt() + "\n\nQuery Context:\n" + optimized_query_context + "\n\nRetrieved Flight Data:\n" + context_str

            messages = [
                {"role": "system", "content": system_prompt_with_optimization},
                *self.conversation_history,
                {"role": "user", "content": query}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.2,
                max_tokens=500
            )

            assistant_response = response.choices[0].message.content
            self.conversation_history.extend([
                {"role": "user", "content": query},
                {"role": "assistant", "content": assistant_response}
            ])

            # Return both the response and plot data
            return {
                "response": assistant_response,
                "plot_data": plot_data
            }

        except Exception as e:
            return {"response": f"Analysis failed: {str(e)}", "plot_data": None}

analyzer = FlightDataAnalyzer()

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or not data.get('query'):
        return jsonify({"error": "Query parameter required"}), 400

    response = analyzer.analyze_query(data['query'].strip())
    return jsonify(response)

@app.route('/api/set-flight-data', methods=['POST', 'OPTIONS'])
def set_flight_data():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'preflight'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response

    try:
        data = request.json
        if not data:
            raise ValueError("No JSON data provided")
            
        analyzer.set_flight_data(data)
        return jsonify({"status": "success"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)



