import subprocess
import pandas as pd


run_records = []
try:
    for model in ['Mobilenet', 'Resnet18', 'Resnet34', 'Resnet50']: 
        for data in ['EUV304', 'HMI-CTnuum', 'HMI-Mag', 'Het']:
            
            # Run the command and get exit code
            try:
                result = subprocess.run(
                    ['python', '-m', 'Main_CP_Train', '--model', model, '--data', data],
                    check=True
                )
                exit_code = result.returncode
            except subprocess.CalledProcessError as e:
                exit_code = e.returncode
                print('Errors!')
                print(f'Model: {model}, Data: {data}')
                print(f"Error message: {e.stderr}")

            record = {
                'model': model,
                'data': data,
                'success': '1' if exit_code == 0 else '0'
            }
            run_records.append(record)
except KeyboardInterrupt:
    print("\nProcess interrupted by user. Exiting...")

# Convert all records to DataFrame
df = pd.DataFrame(run_records)
df.to_csv("Run_scripts_CP_check.csv", index=False)
