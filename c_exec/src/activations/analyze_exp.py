import pandas as pd

MIN_LEN = 64  # Define minimum length for a range segment

def find_piecewise_functions(x, y):
    segments = []
    i = 0
    n = len(x)

    while i < n:
        start = i
        end = i

        # Check for y = x + k
        k = y[i] - x[i]
        while i < n and y[i] == x[i] + k:
            end = i
            i += 1
        if end - start + 1 >= MIN_LEN:
            segments.append((start, end, f"{k}+x" if k != 0 else "x"))
            continue

        # Reset i for checking y = k - x
        i = start
        k = y[i] + x[i]
        while i < n and y[i] == k - x[i]:
            end = i
            i += 1
        if end - start + 1 >= MIN_LEN:
            segments.append((start, end, f"{k}-x"))
            continue

        # Reset i for checking y = k
        i = start
        k = y[i]
        while i < n and y[i] == k:
            end = i
            i += 1
        if end - start + 1 >= MIN_LEN:
            segments.append((start, end, f"{k}"))
            continue

        # For single points or unclassified patterns
        segments.append((start, start, f"{y[start]}"))
        i = start + 1

    return segments

def format_segments(segments, x):
    formatted_segments = []
    for start, end, equation in segments:
        if start == end:
            formatted_segments.append(f"{x[start]},{equation}")
        elif end - start + 1 < MIN_LEN:
            for idx in range(start, end + 1):
                formatted_segments.append(f"{x[idx]},{equation}")
        else:
            formatted_segments.append(f"{x[start]}-{x[end]},{equation}")
    return formatted_segments

# Load data
file = "gelu_raw.txt"
df = pd.read_csv(file, header=None)
x = df[0].values.tolist()
y = df[1].values.tolist()

# Find and format piecewise functions
segments = find_piecewise_functions(x, y)

# merge pieces
pieces = [] 
current_segment = []
for start, end, eqn in segments:
    if start == end:
        current_segment.append((start, end, eqn))
    else:
        # finished lut segment
        if len(current_segment) > 0:
            start = current_segment[0][0]
            end = current_segment[-1][1]
            lut = []
            for i in range(start, end + 1):
                lut.append((x[i], y[i]))
            pieces.append((start, end, lut))
            current_segment = []
        pieces.append((start, end, eqn))

print("module GeLU(input [15:0] in, output [15:0] out);")

for i, p in enumerate(pieces):
    print(f"wire [15:0] pieces{i};")
    if type(p[2]) == list:
        # code gen case statement
        print(f"assign pieces{i} = ", end="")
        for j, (x, y) in enumerate(p[2]):
            print(f"    in == {x} ? {y} :")
        print(" 0;")
    else:
        eqn = p[2]
        eqn = eqn.replace("x", "in")
        print(f"assign pieces{i} = {eqn};")
    print()

# assign piece
print("assign out = ", end="")
for i, p in enumerate(pieces):
    if i == len(pieces) - 1:
        print(f"    pieces{i};")
    else:
        print(f"    in <= {p[1]} ? pieces{i} :")

print("endmodule")