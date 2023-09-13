import csv
import os
import re


def compile_data(base_folder, file_name="summary.txt",
                 variable=None):
        '''
        to compile all the data from different folders into one csv file
        options to include the line name and the variable name(s) to look for; 
        otherwise the programs finds the average amplitude in all the summary.txt

        parameters:
        ------------
        base_folder: str. the name of the base folder
        compile_name: str. optional.
            the name of the final file that has all the data
            default to compiled_file.txt
        folder_prefix: str. optional
            the prefix of the folders that has data to analyze
            e.g. NC230802
        file_name: str. optional
            the name of the file to pull data from
            default to summary.txt
        variable: list of str. optional. Be specific!
            a list of str that the user wants from each data file
            default to average amplitude

        returns:
        ------------
        None
        '''
        print("hello")
        if variable is None:
            variable = ["Total ROI", "Percent Active ROI", "Average Amplitude",
                        "Amplitude Standard Deviation", "Average Max Slope",
                        "Max Slope Standard Deviation", "Average Time to Rise",
                        "Time to Rise Standard Deviation",
                        "Average Interevent Interval (IEI)",
                        "IEI Standard Deviation", "Average Number of events",
                        "Number of events Standard Deviation",
                        "Frequency", "Global Connectivity"]

        dir_list = []

        for (dir_path, _dir_names, file_names) in os.walk(base_folder):
            if file_name in file_names:
                dir_list.append(dir_path)

        files = []
        frequency_unit =""

        # traverse through all the matching files
        for dir_name in dir_list:
            result = open(dir_name + "/" + file_name)
            data = {}
            data['name'] = dir_name.split(os.path.sep)[-1][:-4]

            # find the variable in the file
            for line in result:
                for var in variable:
                    if var.lower().strip() in line.lower():
                        if var not in data:
                            data[var] = []

                        items = line.split(":")
                        values = items[1].strip()
                        value = values.split(" ")

                        num = ""
                        for i in value[0]:
                            if i.isdigit():
                                num += i
                        print(num, "type: ", type(num))
                        data[var] = float(num)

                        if var == "Frequency":
                            frequency_unit = str(value[1:])

                        # for item in items:
                        #     print("item in the line: ", item)
                        #     if any(char.isdigit() for char in item):
                        #         print(float(item))
                        #         data[var] = float(item)

            if len(data) > 1:
                files.append(data)
            else:
                print(f'There is no {var} mentioned in the {dir_name}. Please check again.')

        if len(files) > 0:
            # write into a new csv file
            field_names = ["name"]

            for i in range(len(variable)):
                if variable[i] == "Percent Active ROI":
                    variable[i] += " (%)"
                elif variable[i] == "Average Time to Rise" or\
                    variable[i] == "Average Interevent Interval (IEI)":
                    variable[i] += " (seconds)"
                elif variable[i] == "Frequency":
                    variable[i] += frequency_unit

            field_names.extend(variable)

            print(os.getcwd())
            compile_name = os.getcwd() + "_compile_file.csv"

            with open(base_folder + "/" + compile_name, 'w', newline='') as c_file:
                writer = csv.DictWriter(c_file, fieldnames=field_names)
                writer.writeheader()
                writer.writerows(files)
        else:
            print('no data was found. please check the folder to see if there is any matching file')  # noqa: E501

if __name__ == "__main__":
    base_folder = input("base folder name: ").strip()

    # file_name = input("file name (optional; default to summary.txt):")
    # if len(file_name) < 1:
    #     file_name = "summary.txt"
    # variable = list(input("variable name (optional; default to average amplitude; use ',' to separate each variable): ").split(","))  # noqa: E501
    # if "all" in variable[0]:
    #     variable = ["Total ROI", "Percent Active ROI", "Average Amplitude",
    #                 "Amplitude Standard Deviation", "Average Max Slope",
    #                 "Max Slope Standard Deviation", "Average Time to Rise",
    #                 "Time to Rise Standard Deviation", "Average Interevent Interval (IEI)",
    #                 "IEI Standard Deviation", "Average Number of events",
    #                 "Number of events Standard Deviation", "Frequency",
    #                 "Global Connectivity"]
    # elif "average" in variable[0]:
    #     variable = ["Average Amplitude", "Average Max Slope",
    #         "Average Time to Rise", "Average Interevent Interval (IEI)",
    #         "Average Number of events"]
    # elif len(variable[0]) < 1:
    #     variable = ["Average Amplitude"]
    # output_name = input("Output file name (default: compile_data.csv): ")
    # if len(output_name) < 1:
    #     output_name = "compile_data.csv"
    compile_data(base_folder)
