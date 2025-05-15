m = 1 #if data=32 m*=2 if coeff_size=32 m*=2 if data_complex m*=2 if coeff_complex m*=2
lanes = 8
rows = lanes
offs = [0,0,0,0] 
xstart = 0
xstep = 16
buffer_size = 128



cols = int(128 / (m * lanes))
idx = [0] * (rows * cols)  # Initialize idx array

for i in range(rows * cols):
    c = i % cols
    r = i // cols

    rx = r // 2
    rr = r % 4

    if rr == 0:
        offset = offs[rx] * 4
    elif rr == 1:
        offset = offs[rx] * 4 + 1
    elif rr == 2:
        offset = offs[rx] * 4 + (offs[rx - 1] + 1) * 4
    elif rr == 3:
        offset = offs[rx] * 4 + (offs[rx - 1] + 1) * 4 + 1

    xstep_val = (c // 2) * xstep + (c % 2) * 2
    # ystep = -((c // 2) * xstep - (c % 2) * 2)

    idx[i] = (xstart + offset + xstep_val) % buffer_size

# Print the idx array
for i in range(rows):
    for j in range(cols):
        print(f"{idx[i*cols+j]:3d}", end=" ")
        if j == cols - 1:
            print()