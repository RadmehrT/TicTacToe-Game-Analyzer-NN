import csv

input_csv_filename = 'C:/Users/radme/Desktop/Python Test/Neural Network stuff/TicTacToeAI\Data.csv'
output_csv_filename = 'converted_Data.csv'  

with open(input_csv_filename, newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  

    converted_rows = []

    for row in reader:
        converted_row = [float(value) if value.replace('.', '', 1).isdigit() else value for value in row]

        Space_Mapping = {'x': 1, 'o': 0, 'b': -1}
        converted_row[0] = Space_Mapping.get(converted_row[0], converted_row[0])

        for i in range(0, 9):
            converted_row[i] = Space_Mapping.get(converted_row[i], converted_row[i])
            converted_rows.append(converted_row)

        result_Mapping = {'positive': 1, 'negative': 0}
        converted_row[9] = result_Mapping.get(converted_row[9], converted_row[9])
        
        converted_rows.append(converted_row)

with open(output_csv_filename, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(header)

    writer.writerows(converted_rows)

print(f"Converted data has been saved to {output_csv_filename}")
