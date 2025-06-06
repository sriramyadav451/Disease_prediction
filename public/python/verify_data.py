from pathlib import Path

def verify_data_files():
    base_dir = Path(__file__).parent.parent
    required_files = [
        base_dir / 'data' / 'cleaned_symptom_disease.csv',
        base_dir / 'data' / 'dis_sym_dataset_comb.csv', 
        base_dir / 'data' / 'dis_sym_dataset_norm.csv'
    ]
    
    for file in required_files:
        if file.exists():
            print(f"✅ Found: {file}")
            print(f"   Size: {file.stat().st_size} bytes")
        else:
            print(f"❌ Missing: {file}")

if __name__ == "__main__":
    verify_data_files()