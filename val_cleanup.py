with open('validated.tsv', 'r') as input_file, open('output_file.tsv', 'w') as output_file:
    for line in input_file:
        new_line = line.replace('.mp3', '.wav')
        output_file.write(new_line)