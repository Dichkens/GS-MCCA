"""Debug script to check MELD dataset values in detail"""
import pickle
import numpy as np

print("Loading MELD data structure...")
try:
    data = pickle.load(open('MELD_features/MELD_features_raw1.pkl', 'rb'))
    
    # Get first key from each dict
    first_key = None
    for i, item in enumerate(data[:6]):
        if isinstance(item, dict) and len(item) > 0:
            first_key = list(item.keys())[0]
            break
    
    print(f"Using first key: {first_key}\n")
    
    # Check what each item contains for this first key
    for i, item in enumerate(data):
        if isinstance(item, dict):
            if first_key in item:
                value = item[first_key]
                print(f"Item {i}: type={type(value).__name__}, ", end="")
                if isinstance(value, np.ndarray):
                    print(f"shape={value.shape}, dtype={value.dtype}, value preview: {value if value.size < 20 else str(value.flatten()[:5])+'...'}")
                elif isinstance(value, list):
                    print(f"len={len(value)}, first elem: {value[0] if value else 'empty'}")
                else:
                    print(f"value: {value}")
        elif isinstance(item, set):
            print(f"Item {i}: set with {len(item)} elements, first few: {list(item)[:5]}")
            
    print("\n\nExpected structure based on IEMOCAP:")
    print("0: videoIDs - dict mapping video_id to sequence of IDs")
    print("1: videoSpeakers - dict mapping video_id to speaker info")
    print("2: videoLabels - dict mapping video_id to emotion labels")
    print("3: videoText - dict mapping video_id to text features")
    print("4: videoAudio - dict mapping video_id to audio features")
    print("5: videoVisual - dict mapping video_id to visual features")
    print("6: videoSentence - training or validation set IDs")
    print("7: trainVid or other - training video IDs")
    print("8: testVid or other - test video IDs")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
