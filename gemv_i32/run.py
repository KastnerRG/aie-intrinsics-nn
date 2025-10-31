import numpy as np

# Parameters
while True:
    s = input("Enter DX (e.g., 16): ").strip()
    try:
        DX = int(s)
        break
    except ValueError:
        print("Please enter an integer (e.g., 16).")
        
while True:
    s = input("Enter DY (e.g., 16): ").strip()
    try:
        DY = int(s)
        break
    except ValueError:
        print("Please enter an integer (e.g., 16).")

while True:
    data_type = input("Enter data type (int8, int16, int32): ").strip()
    if data_type == "int8":
        while True:
            mac_type = input("Enter mac function to use (mac8 or mac16): ").strip()
            if mac_type == "mac8":
                x_col = 8
                break
            elif mac_type == "mac16":
                x_col = 16
                break
            else:
                print("Please enter an appropriate mac function.")             
        dtype = np.int8
        bit = 8
        z_col = 16         # AIE PLIO bitwidth = 128 bit. => 128 / 8 = 16
        break
    elif data_type == "int16":
        while True:
            mac_type = input("Enter mac function to use (mac8 or mac16): ").strip()
            if mac_type == "mac8":
                x_col = 8
                break
            elif mac_type == "mac16":
                x_col = 16
                break
            else:
                print("Please enter an appropriate mac function.")  
        dtype = np.int16
        bit = 16
        z_col = 8         # 128 / 16 = 8
        break
    elif data_type == "int32":
        while True:
            mac_type = input("Enter mac function to use (lmac4 or lmac8): ").strip()
            if mac_type == "lmac4":
                x_col = 4
                break
            elif mac_type == "lmac8":
                x_col = 8
                break
            else:
                print("Please enter an appropriate mac function.")  
        dtype = np.int32
        bit = 32
        z_col = 4         # 128 / 8 = 4
        break
    else:
        print("Please enter an appropriate data type.")

num_time_steps = 20
Q = 2    # Number of splits along DX

mat_concat = ','.join([f'm[{i}]' for i in range(Q)])


# Generate matrix and input signals
mat_t = np.random.randint(0, 10, size=(DX, DY), dtype=dtype)

x = np.random.randint(0, 10, size=(num_time_steps, DX), dtype=dtype)

tokens = x.ravel()                # flatten 2D numpy array into 1D array
pad = -DX % (z_col * 2)           # how many zeros to add

if pad:
    zeros = np.zeros((num_time_steps, pad), dtype=x.dtype)
    tokens = np.hstack([x, zeros]).ravel()

new_x = tokens.reshape(-1, z_col)      # reshape the zero-padded 1D array into an 2D array with z_col columns
np.savetxt("data/x.txt", new_x, fmt='%d')   # save the array into x.txt file

mat_tokens = mat_t.ravel()            # flatten 2D numpy array into 1D array
mat_pad = -DY % x_col           # how many zeros to add in each row of matrix
i = (DY - 1) // x_col + 1

if mat_pad:
    zeros = np.zeros((DX, mat_pad), dtype=x.dtype)
    mat_tokens = np.hstack([mat_t, zeros])

new_mat_t = mat_tokens.reshape(-1, x_col*i)  

rows_per_mat = (DX + Q - 1) // Q   

with open('aie/kernels/matrix.h', 'w') as f:
    f.write(f'''
#ifndef MATRIX_H
#define MATRIX_H
#define DTYPE {data_type}
#define DX {rows_per_mat*2}
#define DY {x_col * i}
#define Q {Q}
#define MQS {mat_concat}

alignas({x_col * bit // 8}) const DTYPE matrix[{Q}][{rows_per_mat}][{x_col * i}] = {{''')

    zero_row = ', '.join('0' for _ in range(x_col * i))
    for q in range(Q):
        sub_mat = new_mat_t[q::Q, :]                 
        f.write(f'    {{ // matrix block {q}\n')
        # real rows
        for i in range(sub_mat.shape[0]):
            row_vals = ', '.join(str(int(v)) for v in sub_mat[i])
            f.write(f'        {{{row_vals}}},\n')
        # pad rows if needed
        for _ in range(rows_per_mat - sub_mat.shape[0]):
            f.write(f'        {{{zero_row}}},\n')
        f.write('    }')
        f.write(',\n' if q < Q - 1 else '\n')

    f.write('};\n\n#endif // MATRIX_H\n')


# Compute expected output
# y_exp = np.zeros((num_time_steps, DY), dtype=dtype)
y_exp = np.matmul(x, mat_t)

np.savetxt("data/y_exp.txt", y_exp.reshape(-1, z_col), fmt='%d')





