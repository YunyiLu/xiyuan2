import time, sys
path = r"C:\Users\Easyy\AppData\Local\Temp\claude\C--FDU-Y4S2-xiyuan\88e524e9-689d-4af6-ab84-d42f0037deca\tasks\b1f14w4r3.output"
keywords = ["COMPLETE", "Error", "Traceback", "Monitored", "Model saved", "early"]
while True:
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            tail = lines[-5:] if len(lines) >= 5 else lines
            for line in tail:
                for kw in keywords:
                    if kw.lower() in line.lower():
                        print(line.strip())
                        sys.exit(0)
    except Exception:
        pass
    time.sleep(30)
