
# first, read the input.tcl and find the `set toplevel "X"`

with open('input.tcl', 'r') as f:
    lines = f.readlines()

found_toplevel = False
for line in lines:
    if 'set toplevel' in line:
        toplevel = line.split()[-1].strip()[1:-1]
        found_toplevel = True
        break

if not found_toplevel:
    print('Error: toplevel not found')
    exit(1)

# then, read the ../syn_out/$toplevel.v and find the toplevel module

lines_out = []

has_seen_module_declaration = False
has_seen_module_decEnd = False
has_written = False

with open(f'../syn_out/{toplevel}.v', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if not has_seen_module_declaration:
            if f"module {toplevel}" in line:
                has_seen_module_declaration = True
            lines_out.append(line)
            continue
        elif not has_seen_module_decEnd:
            if ');' in line:
                has_seen_module_decEnd = True
            lines_out.append(line)
            continue
        elif not has_written:
            lines_out.append(f'\tinitial begin\n\t\t$sdf_annotate("../../{toplevel}/syn_out/{toplevel}.sdf");\n\tend\n')
            lines_out.append(line)
            has_written = True
        else:
            lines_out.append(line)

with open(f'../syn_out/{toplevel}.v', 'w') as f:
    f.writelines(lines_out)

print('Done')



