import mol

while True:
    text = input('MOL > ')
    result, error = mol.run(text, '<stdin>')

    if error: 
        print(error.as_string())
    elif result:
        print(result)