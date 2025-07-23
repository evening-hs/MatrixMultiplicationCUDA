
import re
test_results = [str(p) for p in Path('tests/').rglob("MatrixC*")]
import pandas as pd

data = {'run':[], 'time':[], 'size':[], 'pattern':[], 'sparsity':[],
        'algorithm': [], 'gflops': []}

for file in test_results:
    pattern = r'tests/MatrixC_(\d+)_([^_]+)_run(\d+)\.res'
    match = re.match(pattern, file)

    matrix_size = int(match.group(1))
    matrix_pattern = match.group(2)
    run_number = int(match.group(3))

    with open(file) as f:
        lines = f.readlines()
        for line in range(3, len(lines), 4):
            if lines[line].strip().startswith('CUDA'):
                line += 1
            algo = lines[line].split("time")[0].strip()
            time = float(lines[line].split(":")[1].strip())
            #result = list(map(float, lines[line+1].split()))
            #(2 × N³ - N²) / (time_in_ms × 10⁶)

            if time != 0:
                gflops = (2.0 * matrix_size ** 3) / (time * 10 ** 6)
            else:
                gflops = -1

            data['run'].append(run_number)
            data['time'].append(time)
            data['size'].append(matrix_size)
            data['pattern'].append(matrix_pattern)

            data['sparsity'].append(0.5 if pattern == "checkerboard" else 16
                                                                          / matrix_size)
            #data['result'].append(result)
            data['algorithm'].append(algo)
            data['gflops'].append(gflops)

df = pd.DataFrame(data)
df.to_json('results.json')
df.to_csv('results.csv')
#df.to_excel('results.xlsx')