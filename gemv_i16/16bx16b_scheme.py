def compute_scheme(rows, cols, start, offset, offset_hi, step, buffer_size):
    idx = []
    row = []

    for i in range(rows * cols):
        c = i % cols
        r = i // cols

        if r < 8:
            base = offset
        else:
            base = offset_hi

        if r % 2 == 0:
            offs = base[r % 8] * 2
        else:
            offs = base[r % 8] * 2 + (base[(r - 1) % 8] + 1) * 2

        xstep = (c // 2) * step + (c % 2)
        ystep = -(c // 2) * step + (c % 2)

        index = (start + offs + xstep) % buffer_size
        row.append(index)

        if c == cols - 1:
            idx.append(row)
            row = []

    return idx

rows = 16
cols = 2
offset = [0x0, 0x7, 0x1, 0x7, 0x2, 0x7, 0x3, 0x7]
offset_hi = [0x4, 0x7, 0x5, 0x7, 0x6, 0x7, 0x7, 0x7]
start = 0
step = 0
buffer_size = 32

indices = compute_scheme(rows, cols, start, offset, offset_hi, step, buffer_size)

print("16bx16b Scheme Indices:")
for row in indices:
    print(row)
