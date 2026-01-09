"""Quick test of app classifier"""
from src.app_classifier import classify_app

# Test classification
result = classify_app('TallyPrime.exe')

print("\nClassification result:")
print(f"  Success: {result.get('success')}")
print(f"  Suggested name: {result.get('suggested_name')}")
print(f"  From cache: {result.get('from_cache')}")

if result.get('classification'):
    print(f"  Category: {result['classification'].get('app_category')}")
    print(f"  Work-related: {result['classification'].get('is_work_related')}")
