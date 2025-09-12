# inspector.py
# This script will search the d3rlpy library for the evaluation function.

import inspect
import d3rlpy
import d3rlpy.metrics
import d3rlpy.ope

# List of modules we want to search within the d3rlpy library
modules_to_inspect = [d3rlpy, d3rlpy.metrics, d3rlpy.ope]

# Keywords to identify the function we're looking for
search_keywords = ['estimat', 'scorer', 'value']

print("--- Starting Library Inspection ---")
found = False

for module in modules_to_inspect:
    print(f"\nSearching in module: {module.__name__}")
    try:
        for name, member in inspect.getmembers(module):
            # Check if the member is a function or a class
            if inspect.isfunction(member) or inspect.isclass(member):
                # Check if the name contains any of our keywords
                if any(keyword in name.lower() for keyword in search_keywords):
                    print(f"  [FOUND] Name: {name}, Type: {'Function' if inspect.isfunction(member) else 'Class'}")
                    print(f"          Full path to import: from {module.__name__} import {name}")
                    found = True
    except Exception as e:
        print(f"Could not inspect {module.__name__}: {e}")

if not found:
    print("\nCould not automatically find a suitable evaluation function.")
print("\n--- Inspection Finished ---")