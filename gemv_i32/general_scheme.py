def general_scheme(rows, cols, start, offset, step, buffer_size):
	idx = []
	row = []
	for i in range(rows*cols):
		
		c = i % cols
		r = i // cols
		
		row.append(start + offset[r] + step*c)
		
		if (c == cols-1):
			idx.append(row)
			row = []
	return idx
	
# Set rows, cols, offset, start, step, and buffer size for desired mac intrinsic
rows = 4
cols = 2
offset = [0x0, 0x1, 0x2, 0x3, 0x0, 0x0, 0x0, 0x0,] # little endian 
start = 0
step = 16
buffer_size = 16

indices = general_scheme(rows, cols, start, offset, step, buffer_size)
print("General Scheme Indices:", indices)