"""Debug script to check MELD dataset structure"""
import pickle
import numpy as np

print("Loading MELD data structure...")
try:
    data = pickle.load(open('MELD_features/MELD_features_raw1.pkl', 'rb'))
    print(f"Number of items unpacked: {len(data)}")
    print(f"Type of each item:")
    for i, item in enumerate(data):
        if isinstance(item, dict):
            print(f"  [{i}] dict with {len(item)} keys")
            # Print first key as example
            first_key = list(item.keys())[0] if item else None
            if first_key:
                value = item[first_key]
                print(f"       Example value type for key '{first_key}': {type(value)}")
                if isinstance(value, (list, np.ndarray)):
                    print(f"       Example value shape: {np.array(value).shape}")
                    print(f"       Example value: {value}")
        elif isinstance(item, list):
            print(f"  [{i}] list with {len(item)} items")
            if len(item) > 0:
                print(f"       First item: {item[0]}")
        else:
            print(f"  [{i}] {type(item).__name__}: {item if not isinstance(item, (np.ndarray, list)) else '...'}")
    
    print("\n\nMapping unpacked items to variable names:")
    print("Expected: videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid")
    
    if len(data) == 9:
        names = ['videoIDs', 'videoSpeakers', 'videoLabels', 'videoText', 'videoAudio', 'videoVisual', 'videoSentence', 'trainVid', 'testVid']
    elif len(data) == 10:
        names = ['videoIDs', 'videoSpeakers', 'videoLabels', 'videoText', 'videoAudio', 'videoVisual', 'videoSentence', 'trainVid', 'testVid', 'unknown']
    else:
        names = [f'item_{i}' for i in range(len(data))]
    
    for name, item in zip(names, data):
        if isinstance(item, dict) and len(item) > 0:
            first_key = list(item.keys())[0]
            first_val = item[first_key]
            if isinstance(first_val, (list, np.ndarray)):
                shape_info = f"shape {np.array(first_val).shape}"
            else:
                shape_info = f"type {type(first_val).__name__}"
            print(f"{name}: dict with {len(item)} keys, first value {shape_info}")
        else:
            print(f"{name}: {type(item).__name__}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
