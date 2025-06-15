import json
from collections import defaultdict
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

def convert_timestamp(ts: int) -> str:
    """Convert timestamp to readable format"""
    try:
        return datetime.fromtimestamp(ts / 1000000).strftime('%Y-%m-%d %H:%M:%S.%f')
    except:
        return str(ts)

def get_message_description(msg_type: str) -> str:
    """Provides a human-readable description for a given MAVLink message type."""
    descriptions = {
        "AHR2": "Attitude, Heading, and Relative Altitude (Roll, Pitch, Yaw, Altitude, Quaternion).",
        "ATT": "Attitude (Roll, Pitch, Yaw, Roll Speed, Pitch Speed, Yaw Speed).",
        "GPS": "GPS Global Position (Latitude, Longitude, Altitude, Satellites Visible, Fix Type).",
        "MODE": "Current Flight Mode.",
        "CMD": "Command executed by the autopilot.",
        "MSG": "General messages or events from the autopilot.",
        "EV": "Event messages.",
        "PARM": "Parameter values and changes.",
        "XKQ": "Quaternion attitude estimate from external source.",
        "POS": "Position (Altitude, Relative Altitude).",
        "BAT": "Battery status (Voltage, Current, Consumed mAh).",
        "VIB": "Vibration levels.",
        "RCIN": "Raw RC input channels.",
        "RCOUT": "Raw RC output channels.",
        "STAT": "System status (CPU Load, Sensors health).",
        "FMT": "Log format information.",
        "FILE": "File transfer or file-related operations, potentially containing data."
    }
    return descriptions.get(msg_type, f"Unknown message type: {msg_type}")

def get_important_keys(msg_type: str) -> List[str]:
    """Returns a list of important keys for a given message type for LLM understanding."""
    important_keys = {
        "AHR2": ['Roll', 'Pitch', 'Yaw', 'Alt', 'Q1', 'Q2', 'Q3', 'Q4', 'time_boot_ms'],
        "ATT": ['Roll', 'Pitch', 'Yaw', 'RollRate', 'PitchRate', 'YawRate', 'time_boot_ms'],
        "GPS": ['Lat', 'Lng', 'Alt', 'NSats', 'FixType', 'time_boot_ms'],
        "MODE": ['Mode', 'ModeNum', 'time_boot_ms'],
        "CMD": ['Cmd', 'Prm1', 'Prm2', 'Prm3', 'Prm4', 'Prm5', 'Prm6', 'Prm7', 'time_boot_ms'],
        "MSG": ['Message', 'time_boot_ms'],
        "EV": ['Id', 'time_boot_ms'],
        "PARM": ['Name', 'Value', 'time_boot_ms'],
        "XKQ": ['Q1', 'Q2', 'Q3', 'Q4', 'time_boot_ms'],
        "POS": ['Alt', 'RelAlt', 'time_boot_ms'],
        "BAT": ['Volt', 'Curr', 'Consum', 'time_boot_ms'],
        "VIB": ['VibeX', 'VibeY', 'VibeZ', 'time_boot_ms'],
        "RCIN": ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'time_boot_ms'],
        "RCOUT": ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'time_boot_ms'],
        "STAT": ['Load', 'Sensor', 'time_boot_ms'],
        "FMT": ['Type', 'Name', 'time_boot_ms'],
        "FILE": ['Data', 'FmtType', 'time_boot_ms']
    }
    return important_keys.get(msg_type, [])

def summarize_numerical_fields(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates min, max, avg, and std for numerical fields across all records of a message type."""
    stats = defaultdict(lambda: {'values': []})
    
    for record in records:
        for key, value in record.items():
            if isinstance(value, (int, float)):
                stats[key]['values'].append(value)
    
    result = {}
    for key, data in stats.items():
        if data['values']:
            values_array = np.array(data['values'])
            result[key] = {
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'avg': float(np.mean(values_array)),
                'std': float(np.std(values_array))
            }
    return result

def get_sample_records(records: List[Dict[str, Any]], msg_type: str, num_samples_per_segment: int = 5, max_value_length: int = 4000) -> Dict[str, List[Dict[str, Any]]]:
    """Extracts important keys and returns a few sample records from the beginning, middle, and end of the flight."""
    samples = {
        "start": [],
        "mid": [],
        "end": []
    }
    important_keys = get_important_keys(msg_type)
    total_records = len(records)

    if total_records == 0:
        return samples

    def format_record(record):
        formatted_record = {}
        for key in important_keys:
            if key in record:
                value = record[key]
                if isinstance(value, str) and len(value) > max_value_length:
                    formatted_record[key] = value[:max_value_length] + "... (truncated)"
                elif isinstance(value, list):
                    if len(value) > 0:
                        joined_value = "\n".join(str(item) for item in value)
                        if len(joined_value) > max_value_length:
                            formatted_record[key] = joined_value[:max_value_length] + "... (truncated)"
                        else:
                            formatted_record[key] = joined_value
                    else:
                        formatted_record[key] = "[]"
                else:
                    formatted_record[key] = value
        if 'time_boot_ms' in formatted_record and isinstance(record.get('time_boot_ms'), (int, float)):
             formatted_record['time_boot_ms'] = convert_timestamp(record['time_boot_ms'])
        return formatted_record

    for i in range(min(num_samples_per_segment, total_records)):
        samples["start"].append(format_record(records[i]))

    if total_records > num_samples_per_segment * 2:
        mid_start_index = total_records // 2 - num_samples_per_segment // 2
        mid_end_index = mid_start_index + num_samples_per_segment
        for i in range(mid_start_index, min(mid_end_index, total_records)):
            samples["mid"].append(format_record(records[i]))
    elif total_records > num_samples_per_segment:
        mid_start_index = total_records // 2 - min(num_samples_per_segment // 2, total_records // 4)
        mid_end_index = mid_start_index + min(num_samples_per_segment, total_records - mid_start_index)
        for i in range(mid_start_index, mid_end_index):
            samples["mid"].append(format_record(records[i]))

    if total_records > num_samples_per_segment:
        for i in range(max(0, total_records - num_samples_per_segment), total_records):
            samples["end"].append(format_record(records[i]))
    elif total_records > 0:
        for i in range(max(0, total_records - min(num_samples_per_segment, total_records)), total_records):
            if records[i] not in samples["start"] and records[i] not in samples["mid"]:
                samples["end"].append(format_record(records[i]))
    
    return samples

def simplify_flight_data_for_llm(data: Dict[str, Any]) -> Dict[str, Any]:
    """Simplifies the parsed flight data for better LLM understanding."""
    simplified_data = {"flight_log_summary": "This is a summary of a flight log, broken down by message type."}
    
    overall_timestamps = []
    for msg_type, records_data in data.items():
        if isinstance(records_data, dict):
            timestamps = records_data.get('time_boot_ms', {})
            if isinstance(timestamps, dict):
                overall_timestamps.extend([int(ts) for ts in timestamps.keys()])
        elif isinstance(records_data, list):
            for record in records_data:
                if isinstance(record, dict) and 'time_boot_ms' in record:
                    overall_timestamps.append(record['time_boot_ms'])

    if overall_timestamps:
        min_time = min(overall_timestamps)
        max_time = max(overall_timestamps)
        simplified_data["overall_flight_duration"] = {
            "start_time": convert_timestamp(min_time),
            "end_time": convert_timestamp(max_time),
            "duration_ms": max_time - min_time
        }
    
    for msg_type, records in data.items():
        if msg_type == "MSG" and isinstance(records, dict) and "Message" in records:
            type_summary = {
                "description": get_message_description(msg_type),
                "record_count": len(records.get("Message", [])),
                "messages": records["Message"]
            }
            if "time_boot_ms" in records:
                timestamps = records["time_boot_ms"]
                if isinstance(timestamps, dict):
                    ts_values = [float(ts) for ts in timestamps.keys()]
                    type_summary["numerical_statistics"] = {
                        "time_boot_ms": {
                            "min": float(min(ts_values)),
                            "max": float(max(ts_values)),
                            "avg": float(np.mean(ts_values)),
                            "std": float(np.std(ts_values))
                        }
                    }
            simplified_data[msg_type] = type_summary
            continue

        processed_records = []
        if isinstance(records, dict):
            timestamps = records.get('time_boot_ms', {})
            if isinstance(timestamps, dict):
                record_ids = sorted(timestamps.keys(), key=lambda x: int(x))
                for record_id in record_ids:
                    record_row = {}
                    for field, values in records.items():
                        if isinstance(values, dict) and record_id in values:
                            record_row[field] = values[record_id]
                    if record_row:
                        processed_records.append(record_row)
        elif isinstance(records, list):
            processed_records = records

        if not processed_records:
            continue

        type_summary = {
            "description": get_message_description(msg_type),
            "record_count": len(processed_records),
            "numerical_statistics": summarize_numerical_fields(processed_records),
            "sample_data": get_sample_records(processed_records, msg_type, num_samples_per_segment=7)
        }

        simplified_data[msg_type] = type_summary
        
    return simplified_data
