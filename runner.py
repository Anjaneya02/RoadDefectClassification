import subprocess

# List of Python files to run
python_files = ['mobilenetv2.py','vgg19.py', 'resnet50.py']

for file in python_files:
    # Run the Python file
    process = subprocess.run(['python', file])
    
    # Check if the process completed successfully
    if process.returncode != 0:
        print(f"Error running {file}")
        break
    else:
        print(f"Successfully ran {file}")
