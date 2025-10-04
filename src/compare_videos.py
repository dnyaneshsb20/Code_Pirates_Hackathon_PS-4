import json

golden = json.load(open("out_golden/verification_result.json"))
test = json.load(open("out_test/verification_result.json"))

for step, info in golden["verification"].items():
    test_status = test["verification"].get(step, {}).get("status", "missing")
    if test_status != "done":
        print(f"⚠️ {step}: {info['expected']} is MISSING/UNCLEAR in test video")
    else:
        print(f"✅ {step}: completed")
