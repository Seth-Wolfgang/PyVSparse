import os
import subprocess

group_1 = ["int8_t", "uint8_t", "int16_t", "uint16_t", "int32_t", "uint32_t", "int64_t", "uint64_t", "float", "double"]
group_2 = [2, 3]

# Create the directory if it doesn't exist
directory_name = "classFiles"
os.makedirs(directory_name, exist_ok=True)

# List to store function names and file paths
function_names = []
file_paths = []

for g1 in group_1:
    for g2 in group_2:
        file_path = os.path.join(directory_name, f"{g1}_{g2}.cpp")
        function_name = f"init_{g1}_{g2}"
        function_names.append(function_name)
        file_paths.append(file_path)
        with open(file_path, 'w') as file:
            file.write(f'#include "../PyVSparse.hpp"\n\n')
            file.write(f'void {function_name}(py::module& m) {{\n')
            file.write(f'    generateForEachIndexType<{g1}, {g2}>(m);\n')
            file.write('}\n')

        # Optionally, you can print a message indicating the file creation
        print(f"File created: {file_path}")

        # Optionally, you can run the touch command after writing to the file
        command = f"touch {file_path};"
        subprocess.run(command, shell=True)


