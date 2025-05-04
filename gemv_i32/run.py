import numpy as np

# Parameters
num_time_steps = 20
DX = 16  # Num inputs
DY = 16  # Num outputs
DV = 8   # Num lanes
Q = 2    # Number of splits along DX
dtype = np.int32

mat_concat = ','.join([f'm[{i}]' for i in range(Q)])

# Generate matrix and input signals
mat_t = np.random.randint(0, 10, size=(DX, DY), dtype=dtype)
x = np.random.randint(0, 10, size=(num_time_steps, DX), dtype=dtype)
np.savetxt("data/x.txt", x.reshape(num_time_steps*4, DX//4), fmt='%d')

# Prepare matrix for C header
rows_per_mat = DX // Q

with open('aie/kernels/matrix.h', 'w') as f:
    f.write(f'''
#ifndef MATRIX_H
#define MATRIX_H
#define DTYPE int32
#define DX {DX}
#define DY {DY}
#define DV {DV}
#define Q {Q}
#define MQS {mat_concat}

alignas(32) const DTYPE matrix[{Q}][{rows_per_mat}][{DY}] = {{''')

    for q in range(Q):
        sub_mat = mat_t[q::Q, :]
        f.write(f'    {{ // matrix block {q}\n')
        for i in range(rows_per_mat):
            row_vals = ', '.join([f'{val}' for val in sub_mat[i]])
            end_char = ',' if i < rows_per_mat - 1 else ''
            f.write(f'        {{{row_vals}}}{end_char}\n')
        f.write('    }')
        f.write(',\n' if q < Q - 1 else '\n')

    f.write('};\n\n#endif // MATRIX_H\n')

# Compute expected output
# y_exp = np.zeros((num_time_steps, DY), dtype=dtype)
y_exp = (x @ mat_t).astype(np.int32)

np.savetxt("data/y_exp.txt", y_exp.reshape(num_time_steps*4, DY//4), fmt='%d')
