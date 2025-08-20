import subprocess
import time

while True:
    try:
        # Run the program and pass "3" as input
        process = subprocess.Popen(
            ["python", "debugCompetitionRunner.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        time.sleep(1)
        
        # Send "3" followed by newline
        stdout, stderr = process.communicate(input="3\n")
        
        # Print output and errors
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        
    except Exception as e:
        print(f"Error occurred: {e}")
    
    print("Program ended or crashed. Restarting in 2 seconds...")
    time.sleep(2)
