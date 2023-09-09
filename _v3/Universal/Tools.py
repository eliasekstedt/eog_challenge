
def file_it(file_name, message):
    with open(file_name, 'a') as file:
        file.write(f'{message}\n')
