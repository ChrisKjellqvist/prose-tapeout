import re

fname = "prose_e_unit.lst"

with open(fname) as f:
    lst = filter(lambda x: x.strip() != "", f.readlines())

funcs = []

is_in_header = False
start = 0
func_name = None

for q in lst:
    if is_in_header:
        header = re.match("[0-9a-f]+ <(.*)>:\w*$", q)
        if header:
            # beginning of a new function
            sstr = q.split(" <")[0].strip()
            sstart = int(sstr, 16)
            ln = sstart - start
            funcs.append((func_name, start, ln))
            start = sstart
            func_name = header.group(1)
    else:
        header = re.match("[0-9a-f]+ <(.*)>:\w*$", q)
        if header:
            func_name = header.group(1)
            sstr = q.split(" <")[0].strip()
            start = int(sstr, 16)
            is_in_header = True

# print out funcs by length
funcs.sort(key=lambda x: -x[2])
for f in funcs:
    print(f"name({f[0]}): {hex(f[2])} @ {hex(f[1])}")
