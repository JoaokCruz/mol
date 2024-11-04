import basic

while True:
    text = input('MOL > ')
    result, error = basic.run(text, '<stdin>')

    if error: 
        print(error.as_string())
    else:
        print(result)